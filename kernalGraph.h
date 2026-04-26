#pragma once
#include <vector>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include "config.h"
#include "distance.h"
#include "gpuHeap.h"
#include "telemetry.h"
#include "bloomFilter.h"

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
    KernelGraph(KernelGraph&& other) noexcept
        : num_nodes(other.num_nodes), dim(other.dim), data(std::move(other.data)),
          adj(std::move(other.adj)), d_data(other.d_data), d_adj(other.d_adj) {
        other.d_data = nullptr;
        other.d_adj = nullptr;
    }
    KernelGraph& operator=(KernelGraph&& other) noexcept {
        if (this != &other) {
            if (d_data) {
                cudaFree(d_data);
                cudaFree(d_adj);
            }
            num_nodes = other.num_nodes;
            dim = other.dim;
            data = std::move(other.data);
            adj = std::move(other.adj);
            d_data = other.d_data;
            d_adj = other.d_adj;
            other.d_data = nullptr;
            other.d_adj = nullptr;
        }
        return *this;
    }
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

    int    ep      = 0;                                            // official SONG build search starts from node 0
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
            candidates.push({d, v});
            results.push({d, v});
            if ((int)results.size() > EF_CONSTRUCTION) results.pop();
        }
    }

    vector<KP> ret; // ret: EF_CONSTRUCTION best candidates, nearest-first
    while (!results.empty()) { ret.push_back(results.top()); results.pop(); }
    reverse(ret.begin(), ret.end());
    return ret;
}

static vector<int> kg_select_neighbors(const vector<KP>& candidates) {
    vector<int> result;
    result.reserve(SEARCH_DEGREE);
    for (auto& [d, e] : candidates) {
        if ((int)result.size() >= SEARCH_DEGREE) break;
        result.push_back(e);
    }
    return result;
}

// Add reverse edge v→u. If v is full, evict the farthest neighbor if u is closer.
static void kg_add_reverse_edge(KernelGraph& g, int v, int u) {
    const dist_t* data = g.data.data();
    for (int j = 0; j < FIXED_DEGREE; j++) {
        if (g.adj[v * BLOCK_SIZE + j] < 0) { g.adj[v * BLOCK_SIZE + j] = u; return; }
    }
    const dist_t* qv     = data + (size_t)v * g.dim;
    dist_t        dist_u = l2(qv, data + (size_t)u * g.dim, g.dim); // distance v→u
    int    fj = 0;                                                    // fj: index of farthest neighbor
    dist_t fd = l2(qv, data + (size_t)g.adj[v * BLOCK_SIZE] * g.dim, g.dim); // fd: its distance
    for (int j = 1; j < FIXED_DEGREE; j++) {
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
//   Candidates<pq_size>                  — BFS exploration min-heap
//   Results<pq_size>                     — best-so-far max-heap (top = worst = pruning threshold)
//   uint64_t           bloom[BF_WORDS]   — Bloom filter for visited tracking (no global vis array)
//
// data    : base dataset on GPU, row-major [n x dim]
// adj     : adjacency list on GPU [n x BLOCK_SIZE]
// queries : query vectors on GPU, row-major [nq x dim]
// out     : output k-NN node ids [nq x k]
__global__ void kg_search_kernel(
    const float* __restrict__ data,
    const int*   __restrict__ adj,
    const float* __restrict__ queries,
    int*         out,
    QueryTrace*  trace_out,
    int n, int dim, int k, int nq, int ef_search)
{
    const int qid  = blockIdx.x; // qid: query index — one block per query
    if (qid >= nq) return;
    const int lane = threadIdx.x; // lane: thread index within warp (0..31)

    extern __shared__ char smem[];
    float*               sq         = (float*)smem;
    Candidates<pq_size>* candidates = (Candidates<pq_size>*)(sq + dim);
    Results<pq_size>*    results    = (Results<pq_size>*)(candidates + 1);
    uint64_t*            bloom      = (uint64_t*)(results + 1); // bloom: Bloom filter for visited nodes

    QueryTrace trace{};

    // Load query into shared memory; initialize heaps and Bloom filter
    for (int i = lane; i < dim; i += 32)
        sq[i] = queries[(size_t)qid * dim + i];
    if (lane == 0) {
        candidates->init(ef_search);
        results->init(ef_search);
        for (int w = 0; w < BF_WORDS; w++) bloom[w] = 0ULL;
    }
    __syncthreads();

    // Seed BFS from N_MULTIPROBE evenly-spaced entry points
    for (int p = 0; p < N_MULTIPROBE; p++) {
        int ep = (int)((size_t)p * n / N_MULTIPROBE); // ep: p-th evenly-spaced entry point
        float dp = warp_l2(sq, data + (size_t)ep * dim, dim); // all lanes compute, result valid in lane 0
        if (lane == 0 && !bf_test(bloom, ep)) {
            bf_set(bloom, ep);
            bool frontier_added = candidates->push(dp, ep);
            bool result_added = results->try_push(dp, ep);
            trace.seed_count++;
            trace.visited_nodes++;
            trace.distance_evals++;
            trace.frontier_pushes += frontier_added ? 1 : 0;
            trace.result_updates += result_added ? 1 : 0;
            trace.peak_frontier = trace.peak_frontier < candidates->size() ? candidates->size() : trace.peak_frontier;
            trace.peak_results = trace.peak_results < results->size() ? results->size() : trace.peak_results;
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
        if (lane == 0) trace.expanded_nodes++;

        // Early stop: results full and current node is worse than the worst result
        int   rs_full  = 0;    // rs_full: 1 if results heap is at capacity
        float rs_worst = 0.0f; // rs_worst: distance of worst entry in results
        if (lane == 0) { rs_full = results->full() ? 1 : 0; rs_worst = rs_full ? results->worst() : 0.0f; }
        rs_full  = __shfl_sync(0xffffffff, rs_full,  0);
        rs_worst = __shfl_sync(0xffffffff, rs_worst, 0);
        if (rs_full && c_min > rs_worst) {
            if (lane == 0) trace.early_stopped = 1;
            break;
        }

        for (int j = 0; j < BLOCK_SIZE; j++) {
            if (lane == 0) trace.neighbor_slots_scanned++;
            // Stage 1: lane 0 reads neighbor id and broadcasts to warp
            int nb = -1; // nb: neighbor node id (-1 = empty slot)
            if (lane == 0) nb = adj[cn * BLOCK_SIZE + j];
            nb = __shfl_sync(0xffffffff, nb, 0);
            if (nb < 0) continue;

            int skip = 0; // skip: 1 if nb possibly already visited (Bloom filter test)
            if (lane == 0) {
                skip = bf_test(bloom, nb) ? 1 : 0;
                if (!skip) {
                    bf_set(bloom, nb);
                    trace.visited_nodes++;
                    trace.candidate_nodes++;
                }
            }
            skip = __shfl_sync(0xffffffff, skip, 0);
            if (skip) continue;

            // Stage 2: all 32 lanes cooperate to compute L2(query, nb)
            float nb_d = warp_l2(sq, data + (size_t)nb * dim, dim); // nb_d: distance query→nb
            if (lane == 0) trace.distance_evals++;

            // Stage 3: lane 0 updates candidates (always) and results (only if it improves)
            if (lane == 0) {
                bool frontier_added = candidates->push(nb_d, nb);
                bool result_added = results->try_push(nb_d, nb);
                trace.frontier_pushes += frontier_added ? 1 : 0;
                trace.result_updates += result_added ? 1 : 0;
                trace.peak_frontier = trace.peak_frontier < candidates->size() ? candidates->size() : trace.peak_frontier;
                trace.peak_results = trace.peak_results < results->size() ? results->size() : trace.peak_results;
            }
        }
        __syncthreads();
    }

    // Trim results to k, write ids best-first into output
    if (lane == 0) {
        while (results->size() > k) results->pop_worst();
        int cnt = results->drain_best(out + (size_t)qid * k, k); // cnt: number of results written
        for (int i = cnt; i < k; i++) out[(size_t)qid * k + i] = -1; // pad with -1 if fewer than k
        trace.result_count = cnt;
        if (trace_out) trace_out[qid] = trace;
    }
}

// GPU batch search — graph must have been uploaded with g.upload_to_gpu().
// Queries are processed in batches of N_MULTIQUERY; each block uses a Bloom filter
// in shared memory for visited tracking (no global visited array).
// queries    : row-major float queries [nq x dim]
// N_MULTIQUERY : queries per kernel launch (tune down if kernel smem is too large)
// returns    : flat k-NN ids [nq x k]; timing covers query H2D + kernel + result D2H
inline vector<int> gpu_search(const KernelGraph& g, const dist_t* queries, int nq, int k,
                              int N_MULTIQUERY = 512, int ef_search = pq_size,
                              GpuSearchTrace* trace = nullptr) {
    using clk = std::chrono::high_resolution_clock;
    auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const int n = g.num_nodes, dim = g.dim;
    if (ef_search < k) ef_search = k;
    if (ef_search > pq_size) ef_search = pq_size;

    if (trace) {
        *trace = GpuSearchTrace{};
        trace->ef_search = ef_search;
        trace->batch_limit = N_MULTIQUERY;
        trace->queries.resize(nq);
    }

    auto t_alloc0 = clk::now();
    float* d_q;   cudaMalloc(&d_q,   (size_t)N_MULTIQUERY * dim * sizeof(float)); // d_q: GPU query buffer (one batch)
    int*   d_res; cudaMalloc(&d_res, (size_t)N_MULTIQUERY * k   * sizeof(int));   // d_res: GPU result buffer (one batch)
    // no d_vis: visited tracking uses a Bloom filter in shared memory
    QueryTrace* d_trace = nullptr;
    if (trace) cudaMalloc(&d_trace, (size_t)N_MULTIQUERY * sizeof(QueryTrace));
    auto t_alloc1 = clk::now();
    if (trace) trace->alloc_ms = ms(t_alloc0, t_alloc1);

    // smem per block: sq + Candidates + Results + Bloom filter
    size_t smem = sizeof(float) * dim
                + sizeof(Candidates<pq_size>)
                + sizeof(Results<pq_size>)
                + sizeof(uint64_t) * BF_WORDS;

    vector<int> result((size_t)nq * k);
    auto t_total0 = clk::now();

    for (int off = 0; off < nq; off += N_MULTIQUERY) {
        int bs = min(N_MULTIQUERY, nq - off); // bs: actual queries in this batch
        BatchTiming batch;
        batch.batch_index = off / N_MULTIQUERY;
        batch.batch_offset = off;
        batch.batch_size = bs;

        auto t0 = clk::now();
        cudaMemcpy(d_q, queries + (size_t)off * dim, (size_t)bs * dim * sizeof(float), cudaMemcpyHostToDevice);
        auto t1 = clk::now();

        kg_search_kernel<<<bs, 32, smem>>>(g.d_data, g.d_adj, d_q, d_res, d_trace, n, dim, k, bs, ef_search);
        cudaDeviceSynchronize();
        auto t2 = clk::now();

        cudaMemcpy(result.data() + (size_t)off * k, d_res, (size_t)bs * k * sizeof(int), cudaMemcpyDeviceToHost);
        auto t3 = clk::now();

        batch.h2d_ms = ms(t0, t1);
        batch.memset_ms = 0.0; // no memset; Bloom filter is zero-initialized by lane 0 inside the kernel
        batch.kernel_ms = ms(t1, t2);
        batch.d2h_ms = ms(t2, t3);
        if (trace) {
            trace->h2d_ms += batch.h2d_ms;
            trace->memset_ms += batch.memset_ms;
            trace->kernel_ms += batch.kernel_ms;
            trace->d2h_ms += batch.d2h_ms;
            trace->batches.push_back(batch);
            std::vector<QueryTrace> host_trace(bs);
            cudaMemcpy(host_trace.data(), d_trace, (size_t)bs * sizeof(QueryTrace), cudaMemcpyDeviceToHost);
            for (int i = 0; i < bs; ++i) trace->queries[off + i] = host_trace[i];
        }
    }
    auto t_total1 = clk::now();
    if (trace) trace->total_ms = ms(t_total0, t_total1);

    cudaFree(d_q); cudaFree(d_res);
    if (d_trace) cudaFree(d_trace);
    return result;
}
