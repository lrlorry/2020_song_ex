#pragma once
#include <vector>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include "config.h"
#include "bitHash.h"
#include "gpuHeap.h"

using namespace std;

using HP = pair<int, int>;  // (hamming_distance, node_id) pair for priority queues

// GPU NSW graph with Hamming distance on random-projected bit vectors.
// CPU arrays (adj, hashed) are used during build and kept for CPU search.
// GPU arrays (d_hashed, d_adj) are populated by upload_to_gpu() for kernel search.
struct HashKernelGraph {
    int num_nodes;              // number of nodes
    BitHash          bh;        // projection matrix — stored so queries use the same random hyperplanes
    vector<int>      adj;       // CPU adjacency list, adj[i*BLOCK_SIZE+j] = j-th neighbor of i, -1 if empty
    vector<uint32_t> hashed;   // bit-hashed dataset, row-major [num_nodes x HASH_WORDS]

    uint32_t* d_hashed = nullptr; // GPU pointer to hashed dataset [num_nodes x HASH_WORDS]
    int*      d_adj    = nullptr; // GPU pointer to adjacency list [num_nodes x BLOCK_SIZE]

    HashKernelGraph(int num_nodes, BitHash bh, vector<uint32_t> hashed)
        : num_nodes(num_nodes), bh(move(bh)), hashed(move(hashed)),
          adj(num_nodes * BLOCK_SIZE, -1) {}

    void add_vertex(int u, const vector<int>& neighbors) {
        int m = min((int)neighbors.size(), FIXED_DEGREE);
        for (int j = 0; j < m; j++) adj[u * BLOCK_SIZE + j] = neighbors[j];
    }

    // CPU search: project raw float query → bit vector, then Hamming search.
    // query : raw float vector, length = bh.dim
    vector<int> search(const float* query, int k) const {
        vector<uint32_t> qhash(HASH_WORDS); // qhash: bit-projected query
        bh.project(query, qhash.data());
        return search(qhash.data(), k);
    }

    // CPU search on bit vectors.
    // query     : bit vector [HASH_WORDS uint32_t]
    // ef_search : exploration width (>= k)
    vector<int> search(const uint32_t* query, int k, int ef_search = 0) const {
        if (ef_search < k) ef_search = k;
        auto cmp = [](const HP& a, const HP& b) { return a.first > b.first; };
        priority_queue<HP, vector<HP>, decltype(cmp)> q(cmp); // q: BFS frontier min-heap
        priority_queue<HP> topk;                               // topk: best ef_search results, max-heap
        unordered_set<int> visited;

        // Seed BFS from N_MULTIPROBE evenly-spaced entry points
        for (int p = 0; p < N_MULTIPROBE; p++) {
            int ep = (int)((size_t)p * num_nodes / N_MULTIPROBE); // ep: p-th evenly-spaced entry point
            if (visited.count(ep)) continue;
            int d = hamming(query, hashed.data() + (size_t)ep * HASH_WORDS);
            q.push({d, ep});
            topk.push({d, ep});
            visited.insert(ep);
        }

        while (!q.empty()) {
            auto [cd, cu] = q.top(); q.pop(); // cd: Hamming distance of current node, cu: its id
            if ((int)topk.size() == ef_search && cd > topk.top().first) break;

            for (int j = 0; j < BLOCK_SIZE; j++) {
                int v = adj[cu * BLOCK_SIZE + j];
                if (v < 0 || visited.count(v)) continue;
                visited.insert(v);
                int d = hamming(query, hashed.data() + (size_t)v * HASH_WORDS);
                q.push({d, v}); // always explore — bridge nodes matter
                if ((int)topk.size() < ef_search || d < topk.top().first) {
                    topk.push({d, v});
                    if ((int)topk.size() > ef_search) topk.pop();
                }
            }
        }

        while ((int)topk.size() > k) topk.pop();
        vector<int> result;
        result.reserve(k);
        while (!topk.empty()) { result.push_back(topk.top().second); topk.pop(); }
        return result;
    }

    // Copy hashed and adj to GPU. Call once after build; not counted in search timing.
    void upload_to_gpu() {
        cudaMalloc(&d_hashed, (size_t)num_nodes * HASH_WORDS * sizeof(uint32_t));
        cudaMalloc(&d_adj,    (size_t)num_nodes * BLOCK_SIZE * sizeof(int));
        cudaMemcpy(d_hashed, hashed.data(), (size_t)num_nodes * HASH_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_adj,    adj.data(),    (size_t)num_nodes * BLOCK_SIZE * sizeof(int),       cudaMemcpyHostToDevice);
    }

    ~HashKernelGraph() { if (d_hashed) { cudaFree(d_hashed); cudaFree(d_adj); } }
    HashKernelGraph(const HashKernelGraph&)            = delete;
    HashKernelGraph& operator=(const HashKernelGraph&) = delete;
    HashKernelGraph(HashKernelGraph&& other) noexcept
        : num_nodes(other.num_nodes), bh(std::move(other.bh)), adj(std::move(other.adj)),
          hashed(std::move(other.hashed)), d_hashed(other.d_hashed), d_adj(other.d_adj) {
        other.d_hashed = nullptr;
        other.d_adj = nullptr;
    }
    HashKernelGraph& operator=(HashKernelGraph&& other) noexcept {
        if (this != &other) {
            if (d_hashed) {
                cudaFree(d_hashed);
                cudaFree(d_adj);
            }
            num_nodes = other.num_nodes;
            bh = std::move(other.bh);
            adj = std::move(other.adj);
            hashed = std::move(other.hashed);
            d_hashed = other.d_hashed;
            d_adj = other.d_adj;
            other.d_hashed = nullptr;
            other.d_adj = nullptr;
        }
        return *this;
    }
};

// ── NSW build ─────────────────────────────────────────────────────────────────

// Greedy BFS to find EF_CONSTRUCTION nearest already-inserted neighbors of node u (by Hamming).
// Returns candidates sorted nearest-first.
static vector<HP> hkg_search_layer(const HashKernelGraph& g, int u) {
    const uint32_t* q = g.hashed.data() + (size_t)u * HASH_WORDS; // q: bit vector of node being inserted

    auto cmp = [](const HP& a, const HP& b) { return a.first > b.first; };
    priority_queue<HP, vector<HP>, decltype(cmp)> candidates(cmp); // min-heap: BFS frontier
    priority_queue<HP> results;                                     // max-heap: best EF_CONSTRUCTION so far
    unordered_set<int> visited;

    int ep      = rand() % u;                                                // ep: random entry point among 0..u-1
    int ep_dist = hamming(q, g.hashed.data() + (size_t)ep * HASH_WORDS);
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
            int d = hamming(q, g.hashed.data() + (size_t)v * HASH_WORDS);
            if ((int)results.size() < EF_CONSTRUCTION || d < results.top().first) {
                candidates.push({d, v});
                results.push({d, v});
                if ((int)results.size() > EF_CONSTRUCTION) results.pop();
            }
        }
    }

    vector<HP> ret; // ret: EF_CONSTRUCTION best candidates, nearest-first
    while (!results.empty()) { ret.push_back(results.top()); results.pop(); }
    reverse(ret.begin(), ret.end());
    return ret;
}

static vector<int> hkg_select_neighbors(const vector<HP>& candidates) {
    vector<int> result;
    result.reserve(FIXED_DEGREE);
    for (auto& [d, e] : candidates) {
        if ((int)result.size() >= FIXED_DEGREE) break;
        result.push_back(e);
    }
    return result;
}

// Add reverse edge v→u. If v is full, evict the farthest neighbor (by Hamming) if u is closer.
static void hkg_add_reverse_edge(HashKernelGraph& g, int v, int u) {
    for (int j = 0; j < BLOCK_SIZE; j++) {
        if (g.adj[v * BLOCK_SIZE + j] < 0) { g.adj[v * BLOCK_SIZE + j] = u; return; }
    }
    // v is full — evict farthest neighbor if u is closer
    const uint32_t* qv     = g.hashed.data() + (size_t)v * HASH_WORDS;
    int             dist_u = hamming(qv, g.hashed.data() + (size_t)u * HASH_WORDS); // distance v→u
    int fj = 0;                                                                      // fj: index of farthest neighbor
    int fd = hamming(qv, g.hashed.data() + (size_t)g.adj[v * BLOCK_SIZE] * HASH_WORDS); // fd: its distance
    for (int j = 1; j < BLOCK_SIZE; j++) {
        int d = hamming(qv, g.hashed.data() + (size_t)g.adj[v * BLOCK_SIZE + j] * HASH_WORDS);
        if (d > fd) { fd = d; fj = j; }
    }
    if (dist_u < fd) g.adj[v * BLOCK_SIZE + fj] = u;
}

// Build HashKernelGraph: project raw float data to bit vectors, then build NSW on Hamming distance.
// raw_data : original dataset, row-major [n x orig_dim]
inline HashKernelGraph build_hashed_graph(const float* raw_data, int n, int orig_dim) {
    BitHash bh(orig_dim);                        // bh: random projection matrix
    auto hashed = bh.project_all(raw_data, n);   // hashed: bit-projected dataset [n x HASH_WORDS]
    HashKernelGraph g(n, move(bh), move(hashed));
    for (int u = 1; u < n; u++) {
        auto candidates = hkg_search_layer(g, u);
        auto neighbors  = hkg_select_neighbors(candidates);
        g.add_vertex(u, neighbors);
        for (int v : neighbors) hkg_add_reverse_edge(g, v, u);
    }
    return g;
}

// ── GPU search kernel ──────────────────────────────────────────────────────────

// Warp-level Hamming: each lane processes one word (lane < HASH_WORDS → __popc, else 0),
// then reduce across warp; result valid in lane 0.
// sq : query bit vector cached in shared memory [HASH_WORDS uint32_t]
// b  : one node's bit vector in global memory   [HASH_WORDS uint32_t]
__device__ inline int warp_hamming(
    const uint32_t* __restrict__ sq,
    const uint32_t* __restrict__ b)
{
    int lane = threadIdx.x & 31;
    int dist = (lane < HASH_WORDS) ? __popc(sq[lane] ^ b[lane]) : 0; // dist: popcount for this lane's word
    for (int off = 16; off >= 1; off >>= 1)
        dist += __shfl_down_sync(0xffffffff, dist, off); // butterfly reduction → lane 0 has total
    return dist;
}

// One block = 32 threads (1 warp) per query.
//
// Shared memory layout (contiguous, no padding):
//   uint32_t           sq[HASH_WORDS]     — query bit vector cached from global memory
//   Candidates<pq_size>                 — BFS exploration min-heap
//   Results<pq_size>                    — best-so-far max-heap (top = worst = pruning threshold)
//
// hashed  : bit-hashed dataset on GPU [n x HASH_WORDS]
// adj     : adjacency list on GPU [n x BLOCK_SIZE]
// queries : bit-hashed query vectors on GPU [nq x HASH_WORDS]
// out     : output k-NN node ids [nq x k]
// vis     : per-query visited flags [nq x n], must be zeroed before kernel launch
__global__ void hkg_search_kernel(
    const uint32_t* __restrict__ hashed,
    const int*      __restrict__ adj,
    const uint32_t* __restrict__ queries,
    int*            out,
    bool*           vis,
    int n, int k, int nq)
{
    const int qid  = blockIdx.x; // qid: query index — one block per query
    if (qid >= nq) return;
    const int lane = threadIdx.x; // lane: thread index within warp (0..31)

    extern __shared__ char smem[];
    uint32_t*              sq         = (uint32_t*)smem;                            // sq: cached query [HASH_WORDS]
    Candidates<pq_size>* candidates = (Candidates<pq_size>*)(sq + HASH_WORDS);  // candidates: BFS frontier
    Results<pq_size>*    results    = (Results<pq_size>*)(candidates + 1);       // results: best nodes so far

    bool* visited = vis + (size_t)qid * n; // visited: per-query slice of global vis array [n bools]

    // Load query bit vector (each lane loads one word if lane < HASH_WORDS)
    if (lane < HASH_WORDS) sq[lane] = queries[(size_t)qid * HASH_WORDS + lane];
    if (lane == 0) { candidates->init(); results->init(); }
    __syncthreads();

    // Seed BFS from N_MULTIPROBE evenly-spaced entry points
    for (int p = 0; p < N_MULTIPROBE; p++) {
        int ep = (int)((size_t)p * n / N_MULTIPROBE); // ep: p-th evenly-spaced entry point
        float dp = (float)warp_hamming(sq, hashed + (size_t)ep * HASH_WORDS); // all lanes compute, result valid in lane 0
        if (lane == 0 && !visited[ep]) {
            visited[ep] = true;
            candidates->push(dp, ep);
            results->try_push(dp, ep);
        }
        __syncthreads();
    }

    for (;;) {
        // Lane 0 pops the closest candidate; broadcast to all lanes
        int   cn    = -1;    // cn: current node id to expand (-1 → heap empty → stop)
        float c_min = 1e30f; // c_min: Hamming distance of cn
        if (lane == 0 && !candidates->empty()) cn = candidates->pop_min(&c_min);
        cn    = __shfl_sync(0xffffffff, cn,    0);
        c_min = __shfl_sync(0xffffffff, c_min, 0);
        if (cn < 0) break;

        // Early stop: results full and current node is worse than the worst result
        int   rs_full  = 0;    // rs_full: 1 if results heap is at capacity
        float rs_worst = 0.0f; // rs_worst: Hamming distance of worst entry in results
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

            // Stage 2: all 32 lanes cooperate to compute Hamming(query, nb)
            float nb_d = (float)warp_hamming(sq, hashed + (size_t)nb * HASH_WORDS); // nb_d: Hamming distance query→nb

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
// queries    : bit-hashed row-major [nq x HASH_WORDS]
// N_MULTIQUERY : queries per kernel launch; controls peak GPU memory for visited flags
// returns    : flat k-NN ids [nq x k]; timing covers query H2D + kernel + result D2H
inline vector<int> gpu_search(const HashKernelGraph& g, const uint32_t* queries, int nq, int k,
                               int N_MULTIQUERY = 512) {
    const int n = g.num_nodes;

    uint32_t* d_q;   cudaMalloc(&d_q,   (size_t)N_MULTIQUERY * HASH_WORDS * sizeof(uint32_t)); // d_q: GPU query buffer (one batch)
    int*      d_res; cudaMalloc(&d_res, (size_t)N_MULTIQUERY * k           * sizeof(int));      // d_res: GPU result buffer (one batch)
    // visited: N_MULTIQUERY * n bytes — batching keeps this from OOM on large n
    bool*     d_vis; cudaMalloc(&d_vis, (size_t)N_MULTIQUERY * n           * sizeof(bool));     // d_vis: GPU visited flags (one batch)

    // smem: shared memory bytes per block = sq + Candidates + Results
    size_t smem = sizeof(uint32_t) * HASH_WORDS
                + sizeof(Candidates<pq_size>)
                + sizeof(Results<pq_size>);

    vector<int> result((size_t)nq * k);

    for (int off = 0; off < nq; off += N_MULTIQUERY) {
        int bs = min(N_MULTIQUERY, nq - off); // bs: actual queries in this batch

        cudaMemcpy(d_q, queries + (size_t)off * HASH_WORDS, (size_t)bs * HASH_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemset(d_vis, 0, (size_t)bs * n * sizeof(bool));

        hkg_search_kernel<<<bs, 32, smem>>>(g.d_hashed, g.d_adj, d_q, d_res, d_vis, n, k, bs);
        cudaDeviceSynchronize();

        cudaMemcpy(result.data() + (size_t)off * k, d_res, (size_t)bs * k * sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_q); cudaFree(d_res); cudaFree(d_vis);
    return result;
}
