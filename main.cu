#include <iostream>
#include <vector>
#include <chrono>
#include "parser.h"
#include "kernalGraph.h"
#include "hashKernelGraph.h"

using clk = std::chrono::high_resolution_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Standard recall@k: for each query, fraction of its true top-k that appear in results.
// Averaged over all queries.
// results : flat k-NN ids returned by search, row-major [nq x k]
// gt      : ground-truth top-k per query (gt[i][0] = nearest, gt[i][1] = 2nd nearest, …)
// nq      : number of queries
// k       : number of neighbors
static float recall(
    const vector<int>& results,
    const vector<vector<int>>& gt,
    int nq, int k)
{
    float total = 0.0f; // total: sum of per-query recall values
    for (int i = 0; i < nq; i++) {
        int hit = 0; // hit: how many of this query's k results appear in its true top-k
        for (int j = 0; j < k; j++) {
            int id = results[(size_t)i * k + j]; // id: one returned neighbor
            for (int r = 0; r < k && r < (int)gt[i].size(); r++)
                if (gt[i][r] == id) { hit++; break; }
        }
        total += (float)hit / k;
    }
    return total / nq;
}

int main() {
    // ── load full SIFT1M ──────────────────────────────────────────────────────
    auto base  = read_fvecs("data/sift/sift_base.fvecs");       // base: 1M base vectors
    auto query = read_fvecs("data/sift/sift_query.fvecs");      // query: 10K query vectors
    auto gt    = read_ivecs("data/sift/sift_groundtruth.ivecs"); // gt: ground-truth k-NN ids (against full 1M)

    const int n   = (int)base.size();   // n: full dataset size (SIFT1M = 1,000,000)
    const int nq  = (int)query.size(); // nq: number of queries
    const int dim = (int)base[0].size();// dim: vector dimensionality (128 for SIFT)
    const int k   = TOPK;              // k: number of neighbors to retrieve

    cout << "base=" << n << "  queries=" << nq << "  dim=" << dim << "  k=" << k << "\n\n";

    // Flatten base vectors to row-major float array
    vector<dist_t> data((size_t)n * dim); // data: base dataset, row-major [n x dim]
    for (int i = 0; i < n; i++)
        for (int j = 0; j < dim; j++)
            data[(size_t)i * dim + j] = base[i][j];

    // Flatten query vectors to row-major float array
    vector<float> qdata((size_t)nq * dim); // qdata: queries, row-major [nq x dim]
    for (int i = 0; i < nq; i++)
        for (int j = 0; j < dim; j++)
            qdata[(size_t)i * dim + j] = query[i][j];

    // ── KernelGraph: L2 distance, GPU search ─────────────────────────────────
    auto t0 = clk::now();
    KernelGraph kg = build_kernel_graph(vector<dist_t>(data), n, dim);
    auto t1 = clk::now();
    cout << "[KernelGraph]     build  " << ms(t0, t1) << " ms\n";

    kg.upload_to_gpu(); // upload index to GPU — not counted in search timing

    auto t2 = clk::now();
    auto res_kg = gpu_search(kg, qdata.data(), nq, k); // res_kg: k-NN results [nq x k]
    auto t3 = clk::now();
    double t_kg = ms(t2, t3); // t_kg: search latency in ms
    cout << "[KernelGraph GPU] search " << t_kg << " ms"
         << "  QPS=" << (int)(nq / (t_kg / 1000.0))
         << "  recall@" << k << "=" << recall(res_kg, gt, nq, k) << "\n\n";

    // ── HashKernelGraph: Hamming distance on bit projections, GPU search ──────
    auto t4 = clk::now();
    HashKernelGraph hkg = build_hashed_graph(data.data(), n, dim);
    auto t5 = clk::now();
    cout << "[HashKernelGraph] build  " << ms(t4, t5) << " ms\n";

    // Project queries using the same BitHash used at build time
    vector<uint32_t> qhash((size_t)nq * HASH_WORDS); // qhash: bit-projected queries [nq x HASH_WORDS]
    for (int i = 0; i < nq; i++)
        hkg.bh.project(query[i].data(), qhash.data() + (size_t)i * HASH_WORDS);

    hkg.upload_to_gpu(); // upload index to GPU — not counted in search timing

    auto t6 = clk::now();
    auto res_hkg = gpu_search(hkg, qhash.data(), nq, k); // res_hkg: k-NN results [nq x k]
    auto t7 = clk::now();
    double t_hkg = ms(t6, t7); // t_hkg: search latency in ms
    cout << "[HashKernelGraph GPU] search " << t_hkg << " ms"
         << "  QPS=" << (int)(nq / (t_hkg / 1000.0))
         << "  recall@" << k << "=" << recall(res_hkg, gt, nq, k) << "\n";

    return 0;
}
