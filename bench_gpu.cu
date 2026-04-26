#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "parser.h"
#include "kernalGraph.h"
#include "hashKernelGraph.h"

using clk = std::chrono::high_resolution_clock;
namespace fs = std::filesystem;

static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Args {
    std::string base = "data/sift/sift_base.fvecs";
    std::string query = "data/sift/sift_query.fvecs";
    std::string raw_ids_out = "results/repro_gpu_raw.ids";
    std::string hash_ids_out = "results/repro_gpu_hash.ids";
    std::string json_out = "results/repro_gpu.json";
    int n = 10000;
    int nq = 1000;
    int k = TOPK;
};

static Args parse_args(int argc, char** argv) {
    Args args;
    std::unordered_map<std::string, std::string> kv;
    for (int i = 1; i + 1 < argc; i += 2) kv[argv[i]] = argv[i + 1];
    if (kv.count("--base")) args.base = kv["--base"];
    if (kv.count("--query")) args.query = kv["--query"];
    if (kv.count("--raw-ids-out")) args.raw_ids_out = kv["--raw-ids-out"];
    if (kv.count("--hash-ids-out")) args.hash_ids_out = kv["--hash-ids-out"];
    if (kv.count("--json-out")) args.json_out = kv["--json-out"];
    if (kv.count("--n")) args.n = std::stoi(kv["--n"]);
    if (kv.count("--nq")) args.nq = std::stoi(kv["--nq"]);
    if (kv.count("--k")) args.k = std::stoi(kv["--k"]);
    return args;
}

static void write_flat_ids(const std::string& path, const std::vector<int>& ids, int nq, int k) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream out(path);
    for (int i = 0; i < nq; ++i) {
        for (int j = 0; j < k; ++j) {
            if (j) out << ' ';
            out << ids[(size_t)i * k + j];
        }
        out << '\n';
    }
}

static void write_json(const std::string& path,
                       int n, int nq, int dim, int k,
                       double raw_build_ms, double raw_upload_ms, double raw_search_ms,
                       double hash_build_ms, double hash_upload_ms, double hash_search_ms) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream out(path);
    out << "{\n";
    out << "  \"n\": " << n << ",\n";
    out << "  \"nq\": " << nq << ",\n";
    out << "  \"dim\": " << dim << ",\n";
    out << "  \"k\": " << k << ",\n";
    out << "  \"pq_size\": " << pq_size << ",\n";
    out << "  \"hash_bits\": " << hash_bits << ",\n";
    out << "  \"raw_build_ms\": " << raw_build_ms << ",\n";
    out << "  \"raw_upload_ms\": " << raw_upload_ms << ",\n";
    out << "  \"raw_search_ms\": " << raw_search_ms << ",\n";
    out << "  \"raw_qps\": " << (nq / (raw_search_ms / 1000.0)) << ",\n";
    out << "  \"hash_build_ms\": " << hash_build_ms << ",\n";
    out << "  \"hash_upload_ms\": " << hash_upload_ms << ",\n";
    out << "  \"hash_search_ms\": " << hash_search_ms << ",\n";
    out << "  \"hash_qps\": " << (nq / (hash_search_ms / 1000.0)) << "\n";
    out << "}\n";
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    auto base = read_fvecs(args.base);
    auto query = read_fvecs(args.query);
    if (base.empty() || query.empty()) {
        std::cerr << "empty base/query\n";
        return 1;
    }
    if (args.n > (int)base.size()) args.n = (int)base.size();
    if (args.nq > (int)query.size()) args.nq = (int)query.size();
    const int dim = (int)base[0].size();

    std::vector<dist_t> data((size_t)args.n * dim);
    for (int i = 0; i < args.n; i++)
        for (int j = 0; j < dim; j++)
            data[(size_t)i * dim + j] = base[i][j];

    std::vector<float> qdata((size_t)args.nq * dim);
    for (int i = 0; i < args.nq; i++)
        for (int j = 0; j < dim; j++)
            qdata[(size_t)i * dim + j] = query[i][j];

    auto t0 = clk::now();
    KernelGraph kg = build_kernel_graph(std::vector<dist_t>(data), args.n, dim);
    auto t1 = clk::now();
    auto t2 = clk::now();
    kg.upload_to_gpu();
    auto t3 = clk::now();
    auto t4 = clk::now();
    auto res_kg = gpu_search(kg, qdata.data(), args.nq, args.k, N_MULTIQUERY);
    auto t5 = clk::now();

    auto t6 = clk::now();
    HashKernelGraph hkg = build_hashed_graph(data.data(), args.n, dim);
    auto t7 = clk::now();

    std::vector<uint32_t> qhash((size_t)args.nq * HASH_WORDS);
    for (int i = 0; i < args.nq; i++)
        hkg.bh.project(query[i].data(), qhash.data() + (size_t)i * HASH_WORDS);

    auto t8 = clk::now();
    hkg.upload_to_gpu();
    auto t9 = clk::now();
    auto t10 = clk::now();
    auto res_hkg = gpu_search(hkg, qhash.data(), args.nq, args.k, N_MULTIQUERY);
    auto t11 = clk::now();

    write_flat_ids(args.raw_ids_out, res_kg, args.nq, args.k);
    write_flat_ids(args.hash_ids_out, res_hkg, args.nq, args.k);
    write_json(args.json_out, args.n, args.nq, dim, args.k,
               ms(t0, t1), ms(t2, t3), ms(t4, t5),
               ms(t6, t7), ms(t8, t9), ms(t10, t11));

    std::cout << "[repro_gpu_raw] build_ms=" << ms(t0, t1)
              << " upload_ms=" << ms(t2, t3)
              << " search_ms=" << ms(t4, t5)
              << " qps=" << (args.nq / (ms(t4, t5) / 1000.0)) << "\n";
    std::cout << "[repro_gpu_hash] build_ms=" << ms(t6, t7)
              << " upload_ms=" << ms(t8, t9)
              << " search_ms=" << ms(t10, t11)
              << " qps=" << (args.nq / (ms(t10, t11) / 1000.0)) << "\n";
    return 0;
}
