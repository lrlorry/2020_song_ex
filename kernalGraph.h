#pragma once
#include <vector>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <cuda_runtime.h>
#include "config.h"
#include "distance.h"
#include "gpuHeap.h"

using namespace std;
using KP = pair<dist_t, int>;  // (distance, node_id) pair for priority queues

// GPU NSW graph with L2 distance.
// CPU arrays (data, adj) are used during build and kept for reference.
// GPU arrays (d_data, d_adj) are populated by upload_to_gpu() for kernel search.
struct KernelGraph {
    int num_nodes;           // number of nodes
    int dim;                 // vector dimensionality
    vector<dist_t> data;    // CPU dataset, row-major [num_nodes x dim]
    vector<int>    adj;     // CPU adjacency list, adj[i*BLOCK_SIZE+j] = j-th neighbor of i, -1 if empty

    float* d_data = nullptr; // GPU dataset pointer [num_nodes x dim]
    int*   d_adj  = nullptr; // GPU adjacency list pointer [num_nodes x BLOCK_SIZE]

    KernelGraph(int n, int d, vector<dist_t> data_)
        : num_nodes(n), dim(d), data(move(data_)), adj(n * BLOCK_SIZE, -1) {}

    void add_vertex(int u, const vector<int>& neighbors) {
        int m = min((int)neighbors.size(), FIXED_DEGREE);
        for (int j = 0; j < m; j++) adj[u * BLOCK_SIZE + j] = neighbors[j];
    }

    // Copy data and adj to GPU. Call once after build; not counted in search timing.
    void upload_to_gpu() {
        cudaMalloc(&d_data, (size_t)num_nodes * dim        * sizeof(float));
        cudaMalloc(&d_adj,  (size_t)num_nodes * BLOCK_SIZE * sizeof(int));
        cudaMemcpy(d_data, data.data(), (size_t)num_nodes * dim        * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_adj,  adj.data(),  (size_t)num_nodes * BLOCK_SIZE * sizeof(int),   cudaMemcpyHostToDevice);
    }

    ~KernelGraph() { if (d_data) { cudaFree(d_data); cudaFree(d_adj); } }
    KernelGraph(const KernelGraph&)            = delete;
    KernelGraph& operator=(const KernelGraph&) = delete;
};

// ── NSW build ─────────────────────────────────────────────────────────────────

// Greedy BFS to find EF_CONSTRUCTION nearest already-inserted neighbors of node u.
// Returns candidates sorted nearest-first.
static vector<KP> kg_search_layer(const KernelGraph& g, int u) {
    const dist_t* data = g.data.data();
    const dist_t* q    = data + (size_t)u * g.dim; // q: vector of the node being inserted

    priority_queue<KP, vector<KP>, greater<KP>> candidates; // min-heap: BFS frontier
    priority_queue<KP> results;                              // max-heap: best EF_CONSTRUCTION so far
    unordered_set<int> visited;

    int    ep      = rand() % u;                                   // ep: random entry point among 0..u-1
    dist_t ep_dist = l2(q, data + (size_t)ep * g.dim, g.dim);
    candidates.push({ep_dist, ep});
    results.push({ep_dist, ep});
    visited.insert(ep);

    while (!candidates.empty()) {
        auto [cd, cu] = candidates.top(); candidates.pop();
        if ((int)results.size() >= EF_CONSTRUCTION && cd > results.top().first) break;

        for (int j = 0; j < BLOCK_SIZE; j++) {
            int v = g.adj[cu * BLOCK_SIZE + j];
            if (v < 0 || v >= u || visited.count(v)) continue;
            visited.insert(v);
            dist_t d = l2(q, data + (size_t)v * g.dim, g.dim);
            if ((int)results.size() < EF_CONSTRUCTION || d < results.top().first) {
                candidates.push({d, v});
                results.push({d, v});
                if ((int)results.size() > EF_CONSTRUCTION) results.pop();
            }
        }
    }

    vector<KP> ret; // ret: EF_CONSTRUCTION best candidates, nearest-first
    while (!results.empty()) { ret.push_back(results.top()); results.pop(); }
    reverse(ret.begin(), ret.end());
    return ret;
}

static vector<int> kg_select_neighbors(const vector<KP>& candidates) {
    vector<int> result;
    result.reserve(FIXED_DEGREE);
    for (auto& [d, e] : candidates) {
        if ((int)result.size() >= FIXED_DEGREE) break;
        result.push_back(e);
    }
    return result;
}

// Add reverse edge v→u. If v is full, evict the farthest neighbor if u is closer.
static void kg_add_reverse_edge(KernelGraph& g, int v, int u) {
    const dist_t* data = g.data.data();
    for (int j = 0; j < BLOCK_SIZE; j++) {
        if (g.adj[v * BLOCK_SIZE + j] < 0) { g.adj[v * BLOCK_SIZE + j] = u; return; }
    }
    const dist_t* qv     = data + (size_t)v * g.dim;
    dist_t        dist_u = l2(qv, data + (size_t)u * g.dim, g.dim); // distance v→u
    int    fj = 0;                                                    // fj: index of farthest neighbor
    dist_t fd = l2(qv, data + (size_t)g.adj[v * BLOCK_SIZE] * g.dim, g.dim); // fd: its distance
    for (int j = 1; j < BLOCK_SIZE; j++) {
        dist_t d = l2(qv, data + (size_t)g.adj[v * BLOCK_SIZE + j] * g.dim, g.dim);
        if (d > fd) { fd = d; fj = j; }
    }
    if (dist_u < fd) g.adj[v * BLOCK_SIZE + fj] = u;
}

// Build KernelGraph from raw float data, row-major [n x dim].
inline KernelGraph build_kernel_graph(vector<dist_t> data, int n, int dim) {
    KernelGraph g(n, dim, move(data));
    for (int u = 1; u < n; u++) {
        auto candidates = kg_search_layer(g, u);
        auto neighbors  = kg_select_neighbors(candidates);
        g.add_vertex(u, neighbors);
        for (int v : neighbors) kg_add_reverse_edge(g, v, u);
    }
    return g;
}

// ── GPU search kernel ──────────────────────────────────────────────────────────

// Warp-level L2: all 32 lanes each sum their stripe of (sq[i]-b[i])^2,
// then reduce across the warp; result valid in lane 0.
// sq  : query vector cached in shared memory [dim floats]
// b   : one node's vector in global memory   [dim floats]
__device__ inline float warp_l2(
    const float* __restrict__ sq,
    const float* __restrict__ b,
    int dim)
{
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += 32) { // each lane handles every 32nd element
        float d = sq[i] - b[i];
        sum += d * d;
    }
    for (int off = 16; off >= 1; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off); // butterfly reduction → lane 0 has total
    return sum;
}

// One block = 32 threads (1 warp) per query.
//
// Shared memory layout (contiguous, no padding):
//   float              sq[dim]            — query vector cached from global memory
//   Candidates<pq_size>                 — BFS exploration min-heap
//   Results<pq_size>                    — best-so-far max-heap (top = worst = pruning threshold)
//
// data    : base dataset on GPU, row-major [n x dim]
// adj     : adjacency list on GPU [n x BLOCK_SIZE]
// queries : query vectors on GPU, row-major [nq x dim]
// out     : output k-NN node ids [nq x k]
// vis     : per-query visited flags [nq x n], must be zeroed before kernel launch
__global__ void kg_search_kernel(
    const float* __restrict__ data,
    const int*   __restrict__ adj,
    const float* __restrict__ queries,
    int*         out,
    bool*        vis,
    int n, int dim, int k, int nq)
{
    const int qid  = blockIdx.x; // qid: query index — one block per query
    if (qid >= nq) return;
    const int lane = threadIdx.x; // lane: thread index within warp (0..31)

    extern __shared__ char smem[];
    float*               sq         = (float*)smem;                         // sq: cached query [dim]
    Candidates<pq_size>* candidates = (Candidates<pq_size>*)(sq + dim); // candidates: BFS frontier
    Results<pq_size>*    results   = (Results<pq_size>*)(candidates + 1); // results: best nodes so far

    bool* visited = vis + (size_t)qid * n; // visited: per-query slice of the global vis array [n bools]

    // Load query into shared memory (each lane loads its stripe)
    for (int i = lane; i < dim; i += 32)
        sq[i] = queries[(size_t)qid * dim + i];
    if (lane == 0) { candidates->init(); results->init(); }
    __syncthreads();

    // Seed BFS from N_MULTIPROBE evenly-spaced entry points
    for (int p = 0; p < N_MULTIPROBE; p++) {
        int ep = (int)((size_t)p * n / N_MULTIPROBE); // ep: p-th evenly-spaced entry point
        float dp = warp_l2(sq, data + (size_t)ep * dim, dim); // all lanes compute, result valid in lane 0
        if (lane == 0 && !visited[ep]) {
            visited[ep] = true;
            candidates->push(dp, ep);
            results->try_push(dp, ep);
        }
        __syncthreads();
    }

    for (;;) {
        // Lane 0 pops the closest candidate; result broadcast to all lanes
        int   cn    = -1;    // cn: current node id to expand (-1 → heap empty → stop)
        float c_min = 1e30f; // c_min: distance of cn
        if (lane == 0 && !candidates->empty()) cn = candidates->pop_min(&c_min);
        cn    = __shfl_sync(0xffffffff, cn,    0);
        c_min = __shfl_sync(0xffffffff, c_min, 0);
        if (cn < 0) break;

        // Early stop: results full and current node is worse than the worst result
        int   rs_full  = 0;    // rs_full: 1 if results heap is at capacity
        float rs_worst = 0.0f; // rs_worst: distance of worst entry in results
        if (lane == 0) { rs_full = results->full() ? 1 : 0; rs_worst = rs_full ? results->worst() : 0.0f; }
        rs_full  = __shfl_sync(0xffffffff, rs_full,  0);
        rs_worst = __shfl_sync(0xffffffff, rs_worst, 0);
        if (rs_full && c_min > rs_worst) break;

        for (int j = 0; j < BLOCK_SIZE; j++) {
            // Stage 1: lane 0 reads neighbor id and broadcasts to warp
            int nb = -1; // nb: neighbor node id (-1 = empty slot)
            if (lane == 0) nb = adj[cn * BLOCK_SIZE + j];
            nb = __shfl_sync(0xffffffff, nb, 0);
            if (nb < 0) continue;

            int skip = 0; // skip: 1 if nb already visited
            if (lane == 0) { skip = visited[nb] ? 1 : 0; if (!skip) visited[nb] = true; }
            skip = __shfl_sync(0xffffffff, skip, 0);
            if (skip) continue;

            // Stage 2: all 32 lanes cooperate to compute L2(query, nb)
            float nb_d = warp_l2(sq, data + (size_t)nb * dim, dim); // nb_d: distance query→nb

            // Stage 3: lane 0 updates candidates (always) and results (only if it improves)
            if (lane == 0) { candidates->push(nb_d, nb); results->try_push(nb_d, nb); }
        }
        __syncthreads();
    }

    // Trim results to k, write ids best-first into output
    if (lane == 0) {
        while (results->size() > k) results->pop_worst();
        int cnt = results->drain_best(out + (size_t)qid * k, k); // cnt: number of results written
        for (int i = cnt; i < k; i++) out[(size_t)qid * k + i] = -1; // pad with -1 if fewer than k
    }
}

// GPU batch search — graph must have been uploaded with g.upload_to_gpu().
// Queries are processed in batches so visited flags never exceed N_MULTIQUERY * n bytes.
// queries    : row-major float queries [nq x dim]
// N_MULTIQUERY : queries per kernel launch; controls peak GPU memory for visited flags
//              (visited = N_MULTIQUERY * n bytes — tune down if OOM)
// returns    : flat k-NN ids [nq x k]; timing covers query H2D + kernel + result D2H
inline vector<int> gpu_search(const KernelGraph& g, const dist_t* queries, int nq, int k,
                               int N_MULTIQUERY = 512) {
    const int n = g.num_nodes, dim = g.dim;

    float* d_q;   cudaMalloc(&d_q,   (size_t)N_MULTIQUERY * dim * sizeof(float)); // d_q: GPU query buffer (one batch)
    int*   d_res; cudaMalloc(&d_res, (size_t)N_MULTIQUERY * k   * sizeof(int));   // d_res: GPU result buffer (one batch)
    // visited: N_MULTIQUERY * n bytes — batching keeps this from OOM on large n
    bool*  d_vis; cudaMalloc(&d_vis, (size_t)N_MULTIQUERY * n   * sizeof(bool));  // d_vis: GPU visited flags (one batch)

    // smem: shared memory bytes per block = sq + Candidates + Results
    size_t smem = sizeof(float) * dim
                + sizeof(Candidates<pq_size>)
                + sizeof(Results<pq_size>);

    vector<int> result((size_t)nq * k);

    for (int off = 0; off < nq; off += N_MULTIQUERY) {
        int bs = min(N_MULTIQUERY, nq - off); // bs: actual queries in this batch

        cudaMemcpy(d_q, queries + (size_t)off * dim, (size_t)bs * dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_vis, 0, (size_t)bs * n * sizeof(bool));

        kg_search_kernel<<<bs, 32, smem>>>(g.d_data, g.d_adj, d_q, d_res, d_vis, n, dim, k, bs);
        cudaDeviceSynchronize();

        cudaMemcpy(result.data() + (size_t)off * k, d_res, (size_t)bs * k * sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_q); cudaFree(d_res); cudaFree(d_vis);
    return result;
}
