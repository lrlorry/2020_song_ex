#pragma once
#include <vector>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include "config.h"
#include "distance.h"

// CPU NSW proximity graph with three-stage batch search (SONG paper, Algorithm 1).
struct Graph {
    int num_nodes;           // number of nodes in the graph
    int dim;                 // vector dimensionality
    std::vector<idx_t>  adj; // flat adjacency list, row-major [num_nodes x BLOCK_SIZE];
                             //   adj[i*BLOCK_SIZE + j] = j-th neighbor of node i, -1 if empty
    std::vector<dist_t> data;// base dataset, row-major [num_nodes x dim]

    Graph(int num_nodes, int dim, std::vector<dist_t> data)
        : num_nodes(num_nodes), dim(dim), data(std::move(data)),
          adj(num_nodes * BLOCK_SIZE, -1) {}

    // Write up to FIXED_DEGREE neighbors for node u into adj.
    void add_vertex(idx_t u, const std::vector<idx_t>& neighbors) {
        int m = std::min((int)neighbors.size(), FIXED_DEGREE);
        for (int j = 0; j < m; j++)
            adj[u * BLOCK_SIZE + j] = neighbors[j];
    }

    // ANN search: three-stage batch approach from SONG paper.
    // query     : query vector, length = dim
    // k         : number of neighbors to return
    // ef_search : exploration width (>= k); larger = better recall, slower
    std::vector<idx_t> search(const dist_t* query, int k, int ef_search = 0) const {
        if (ef_search < k) ef_search = k;
        using P = std::pair<dist_t, idx_t>;

        // q    : exploration frontier, min-heap — always expand the closest unvisited node
        // topk : best ef_search results so far, max-heap — top = worst result = pruning threshold
        std::priority_queue<P, std::vector<P>, std::greater<P>> q;
        std::priority_queue<P>                                  topk;
        std::unordered_set<idx_t>                               visited; // nodes already evaluated

        // Seed BFS from N_MULTIPROBE evenly-spaced entry points
        for (int p = 0; p < N_MULTIPROBE; p++) {
            idx_t  ep = (idx_t)((size_t)p * num_nodes / N_MULTIPROBE); // ep: p-th evenly-spaced entry point
            if (visited.count(ep)) continue;
            dist_t d = l2(query, data.data() + (size_t)ep * dim, dim);
            q.push({d, ep});
            topk.push({d, ep});
            visited.insert(ep);
        }

        while (!q.empty()) {
            auto [cd, cu] = q.top(); q.pop(); // cd: distance of current node, cu: its id
            if ((int)topk.size() == ef_search && cd > topk.top().first) break;

            // Stage 1: Candidate Locating — collect all unvisited neighbors of cu
            std::vector<idx_t> cands;
            cands.reserve(BLOCK_SIZE);
            for (int j = 0; j < BLOCK_SIZE; j++) {
                idx_t v = adj[cu * BLOCK_SIZE + j];
                if (v < 0 || visited.count(v)) continue;
                visited.insert(v);
                cands.push_back(v);
            }

            // Stage 2: Bulk Distance Computation — compute all distances in one pass
            std::vector<dist_t> dists(cands.size()); // dists[i]: distance from query to cands[i]
            for (int i = 0; i < (int)cands.size(); i++)
                dists[i] = l2(query, data.data() + (size_t)cands[i] * dim, dim);

            // Stage 3: Data Structure Maintenance
            // q is pushed unconditionally — bridge nodes (mediocre themselves, good neighbors)
            // must remain explorable even when they don't improve topk
            for (int i = 0; i < (int)cands.size(); i++) {
                q.push({dists[i], cands[i]});
                if ((int)topk.size() < ef_search || dists[i] < topk.top().first) {
                    topk.push({dists[i], cands[i]});
                    if ((int)topk.size() > ef_search) topk.pop();
                }
            }
        }

        while ((int)topk.size() > k) topk.pop();

        std::vector<idx_t> result;
        result.reserve(k);
        while (!topk.empty()) { result.push_back(topk.top().second); topk.pop(); }
        return result;
    }
};

// ── NSW build ─────────────────────────────────────────────────────────────────

using P = std::pair<dist_t, idx_t>;  // (distance, node_id) pair for priority queues

// Greedy BFS to find EF_CONSTRUCTION nearest already-inserted neighbors of node u.
// Returns candidates sorted nearest-first.
static std::vector<P> graph_search_layer(const Graph& g, idx_t u) {
    const dist_t* data = g.data.data();
    const dist_t* q    = data + (size_t)u * g.dim; // q: vector of the node being inserted

    std::priority_queue<P, std::vector<P>, std::greater<P>> candidates; // min-heap: BFS frontier
    std::priority_queue<P> results;                                      // max-heap: best EF_CONSTRUCTION so far
    std::unordered_set<idx_t> visited;

    idx_t  ep      = 0;                                              // official SONG build search starts from node 0
    dist_t ep_dist = l2(q, data + (size_t)ep * g.dim, g.dim);
    candidates.push({ep_dist, ep});
    results.push({ep_dist, ep});
    visited.insert(ep);

    while (!candidates.empty()) {
        auto [cd, cu] = candidates.top(); candidates.pop();
        if ((int)results.size() >= EF_CONSTRUCTION && cd > results.top().first) break;

        for (int j = 0; j < BLOCK_SIZE; j++) {
            idx_t v = g.adj[cu * BLOCK_SIZE + j];
            if (v < 0 || v >= u || visited.count(v)) continue; // only look at already-inserted nodes
            visited.insert(v);
            dist_t d = l2(q, data + (size_t)v * g.dim, g.dim);
            candidates.push({d, v});
            results.push({d, v});
            if ((int)results.size() > EF_CONSTRUCTION) results.pop();
        }
    }

    std::vector<P> ret; // ret: EF_CONSTRUCTION best candidates, nearest-first
    while (!results.empty()) { ret.push_back(results.top()); results.pop(); }
    std::reverse(ret.begin(), ret.end());
    return ret;
}

// Select the nearest SEARCH_DEGREE nodes from the candidate list.
static std::vector<idx_t> graph_select_neighbors(const std::vector<P>& candidates) {
    std::vector<idx_t> result;
    result.reserve(SEARCH_DEGREE);
    for (auto& [d, e] : candidates) {
        if ((int)result.size() >= SEARCH_DEGREE) break;
        result.push_back(e);
    }
    return result;
}

// Add reverse edge v→u. If v's neighbor list is full, evict the farthest neighbor
// only if u is closer (maintains graph quality in both directions).
static void graph_add_reverse_edge(Graph& g, idx_t v, idx_t u) {
    const dist_t* data = g.data.data();
    for (int j = 0; j < FIXED_DEGREE; j++) {
        if (g.adj[v * BLOCK_SIZE + j] < 0) { g.adj[v * BLOCK_SIZE + j] = u; return; }
    }
    // v is full — evict farthest neighbor if u is closer
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

// Build NSW graph from raw float data, row-major [n x dim].
inline Graph build_graph(std::vector<dist_t> data, int n, int dim) {
    Graph g(n, dim, std::move(data));
    for (int u = 1; u < n; u++) {
        auto candidates = graph_search_layer(g, u);
        auto neighbors  = graph_select_neighbors(candidates);
        g.add_vertex(u, neighbors);
        for (idx_t v : neighbors) graph_add_reverse_edge(g, v, u);
    }
    return g;
}
