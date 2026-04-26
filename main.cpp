#include <iostream>
#include <vector>
#include <chrono>
#include <queue>
#include "parser.h"
#include "graph.h"

using clk = std::chrono::high_resolution_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Standard recall@k: avg over queries of |result ∩ top-k GT| / k
static float recall(
    const std::vector<std::vector<idx_t>>& results, // results[i]: k neighbors returned for query i
    const std::vector<std::vector<idx_t>>& gt,      // gt[i]: exact top-k for query i
    int nq, int k)
{
    float total = 0.0f;
    for (int i = 0; i < nq; i++) {
        int hit = 0;
        for (int j = 0; j < k; j++) {
            idx_t id = results[i][j];
            for (int r = 0; r < k && r < (int)gt[i].size(); r++)
                if (gt[i][r] == id) { hit++; break; }
        }
        total += (float)hit / k;
    }
    return total / nq;
}

// Brute-force exact k-NN for query against base[0..n-1].
// Returns k nearest neighbor ids, nearest-first.
static std::vector<idx_t> brute_force(
    const float* query,          // query vector [dim]
    const std::vector<dist_t>& base, // base dataset, row-major [n x dim]
    int n, int dim, int k)
{
    // max-heap: (dist, id), top = worst result = pruning threshold
    std::priority_queue<std::pair<dist_t, idx_t>> pq;
    for (int i = 0; i < n; i++) {
        dist_t d = l2(query, base.data() + (size_t)i * dim, dim);
        pq.push({d, i});
        if ((int)pq.size() > k) pq.pop();
    }
    std::vector<idx_t> result(pq.size());
    for (int i = (int)pq.size() - 1; i >= 0; i--) { result[i] = pq.top().second; pq.pop(); }
    return result;
}

int main() {
    // ── load SIFT1M (use first n base vectors) ────────────────────────────────
    auto base  = read_fvecs("data/sift/sift_base.fvecs");
    auto query = read_fvecs("data/sift/sift_query.fvecs");
    // Note: the .ivecs GT is against the full 1M base; we recompute brute-force GT for the n-vector subset

    const int n   = 10000;
    const int nq  = (int)query.size();
    const int dim = (int)base[0].size();
    const int k   = TOPK;

    std::cout << "base=" << n << "  queries=" << nq << "  dim=" << dim << "  k=" << k << "\n\n";

    // Flatten base vectors to row-major float array
    std::vector<dist_t> data((size_t)n * dim);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < dim; j++)
            data[(size_t)i * dim + j] = base[i][j];

    // ── brute-force GT against first n base vectors ───────────────────────────
    std::cout << "computing brute-force GT for " << nq << " queries × " << n << " base... ";
    std::cout.flush();
    std::vector<std::vector<idx_t>> gt(nq);
    for (int i = 0; i < nq; i++)
        gt[i] = brute_force(query[i].data(), data, n, dim, k);
    std::cout << "done\n\n";

    // ── build Graph ───────────────────────────────────────────────────────────
    auto t0 = clk::now();
    Graph g = build_graph(std::vector<dist_t>(data), n, dim);
    auto t1 = clk::now();
    std::cout << "[Graph] build  " << ms(t0, t1) << " ms\n";

    // Sanity check: every node except 0 must have at least one neighbor
    int total_edges = 0;
    for (int u = 1; u < n; u++) {
        int cnt = 0;
        for (int j = 0; j < BLOCK_SIZE; j++)
            if (g.adj[u * BLOCK_SIZE + j] >= 0) cnt++;
        if (cnt == 0) { std::cerr << "node " << u << " has no neighbors\n"; return 1; }
        total_edges += cnt;
    }
    std::cout << "[Graph] avg degree: " << (float)total_edges / n << "\n";

    // ── CPU search ────────────────────────────────────────────────────────────
    std::vector<std::vector<idx_t>> results(nq);
    auto t2 = clk::now();
    for (int i = 0; i < nq; i++)
        results[i] = g.search(query[i].data(), k, pq_size);
    auto t3 = clk::now();
    double t_search = ms(t2, t3);

    std::cout << "[Graph] search " << t_search << " ms"
              << "  QPS=" << (int)(nq / (t_search / 1000.0))
              << "  recall@" << k << "=" << recall(results, gt, nq, k) << "\n";

    return 0;
}
