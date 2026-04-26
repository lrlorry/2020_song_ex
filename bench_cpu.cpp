#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "parser.h"
#include "graph.h"

using clk = std::chrono::high_resolution_clock;
namespace fs = std::filesystem;

static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Args {
    std::string base = "data/sift/sift_base.fvecs";
    std::string query = "data/sift/sift_query.fvecs";
    std::string gt_npy;
    std::string ids_out = "results/repro_cpu_raw.ids";
    std::string json_out = "results/repro_cpu_raw.json";
    int n = 10000;
    int nq = 1000;
    int k = TOPK;
    int pq = pq_size;
};

static Args parse_args(int argc, char** argv) {
    Args args;
    std::unordered_map<std::string, std::string> kv;
    for (int i = 1; i + 1 < argc; i += 2) kv[argv[i]] = argv[i + 1];
    if (kv.count("--base")) args.base = kv["--base"];
    if (kv.count("--query")) args.query = kv["--query"];
    if (kv.count("--gt-npy")) args.gt_npy = kv["--gt-npy"];
    if (kv.count("--ids-out")) args.ids_out = kv["--ids-out"];
    if (kv.count("--json-out")) args.json_out = kv["--json-out"];
    if (kv.count("--n")) args.n = std::stoi(kv["--n"]);
    if (kv.count("--nq")) args.nq = std::stoi(kv["--nq"]);
    if (kv.count("--k")) args.k = std::stoi(kv["--k"]);
    if (kv.count("--pq-size")) args.pq = std::stoi(kv["--pq-size"]);
    return args;
}

static std::vector<idx_t> brute_force(
    const float* query,
    const std::vector<dist_t>& base,
    int n, int dim, int k)
{
    std::priority_queue<std::pair<dist_t, idx_t>> pq;
    for (int i = 0; i < n; i++) {
        dist_t d = l2(query, base.data() + (size_t)i * dim, dim);
        pq.push({d, i});
        if ((int)pq.size() > k) pq.pop();
    }
    std::vector<idx_t> result(pq.size());
    for (int i = (int)pq.size() - 1; i >= 0; i--) {
        result[i] = pq.top().second;
        pq.pop();
    }
    return result;
}

static std::vector<std::vector<idx_t>> load_gt_npy(const std::string& path, int nq, int k) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open gt npy: " + path);

    char magic[6];
    in.read(magic, 6);
    if (!in || std::string(magic, 6) != std::string("\x93NUMPY", 6)) {
        throw std::runtime_error("invalid npy magic: " + path);
    }

    unsigned char version[2] = {0, 0};
    in.read(reinterpret_cast<char*>(version), 2);
    if (!in) throw std::runtime_error("failed to read npy version: " + path);

    std::size_t header_len = 0;
    if (version[0] == 1) {
        std::uint16_t x = 0;
        in.read(reinterpret_cast<char*>(&x), sizeof(x));
        header_len = x;
    } else {
        std::uint32_t x = 0;
        in.read(reinterpret_cast<char*>(&x), sizeof(x));
        header_len = x;
    }
    if (!in) throw std::runtime_error("failed to read npy header length: " + path);

    std::string header(header_len, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!in) throw std::runtime_error("failed to read npy header: " + path);

    if (header.find("False") == std::string::npos || header.find("fortran_order") == std::string::npos) {
        throw std::runtime_error("expected C-order npy array: " + path);
    }
    if (header.find("<i4") == std::string::npos && header.find("|i4") == std::string::npos) {
        throw std::runtime_error("expected int32 npy array: " + path);
    }

    auto shape_pos = header.find("shape");
    auto lparen = header.find('(', shape_pos);
    auto rparen = header.find(')', lparen);
    if (shape_pos == std::string::npos || lparen == std::string::npos || rparen == std::string::npos) {
        throw std::runtime_error("failed to parse npy shape: " + path);
    }
    std::string shape = header.substr(lparen + 1, rparen - lparen - 1);
    for (char& ch : shape) {
        if (!(ch >= '0' && ch <= '9')) ch = ' ';
    }
    std::istringstream iss(shape);
    int rows = 0, cols = 0;
    iss >> rows >> cols;
    if (rows < nq || cols < k) {
        throw std::runtime_error("gt npy smaller than requested nq/k: " + path);
    }

    std::vector<std::int32_t> raw((size_t)rows * cols);
    in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size() * sizeof(std::int32_t)));
    if (!in) throw std::runtime_error("failed to read npy payload: " + path);

    std::vector<std::vector<idx_t>> gt(nq, std::vector<idx_t>(k));
    for (int i = 0; i < nq; ++i)
        for (int j = 0; j < k; ++j)
            gt[i][j] = static_cast<idx_t>(raw[(size_t)i * cols + j]);
    return gt;
}

static void write_ids(const std::string& path, const std::vector<std::vector<idx_t>>& results) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream out(path);
    for (const auto& row : results) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i) out << ' ';
            out << row[i];
        }
        out << '\n';
    }
}

static void write_json(const std::string& path, int n, int nq, int dim, int k, int pq,
                       double build_ms, double search_ms, double gt_ms,
                       double recall_at_k) {
    double qps = nq / (search_ms / 1000.0);
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream out(path);
    out << "{\n";
    out << "  \"method\": \"repro_cpu_raw\",\n";
    out << "  \"n\": " << n << ",\n";
    out << "  \"nq\": " << nq << ",\n";
    out << "  \"dim\": " << dim << ",\n";
    out << "  \"k\": " << k << ",\n";
    out << "  \"pq_size\": " << pq << ",\n";
    out << "  \"build_ms\": " << build_ms << ",\n";
    out << "  \"search_ms\": " << search_ms << ",\n";
    out << "  \"gt_ms\": " << gt_ms << ",\n";
    out << "  \"qps\": " << qps << ",\n";
    out << "  \"recall_at_k\": " << recall_at_k << "\n";
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

    auto t_gt0 = clk::now();
    std::vector<std::vector<idx_t>> gt;
    if (!args.gt_npy.empty()) {
        gt = load_gt_npy(args.gt_npy, args.nq, args.k);
    } else {
        gt.resize(args.nq);
        for (int i = 0; i < args.nq; i++)
            gt[i] = brute_force(query[i].data(), data, args.n, dim, args.k);
    }
    auto t_gt1 = clk::now();

    auto t0 = clk::now();
    Graph g = build_graph(std::vector<dist_t>(data), args.n, dim);
    auto t1 = clk::now();

    std::vector<std::vector<idx_t>> results(args.nq);
    auto t2 = clk::now();
    for (int i = 0; i < args.nq; i++)
        results[i] = g.search(query[i].data(), args.k, args.pq);
    auto t3 = clk::now();

    float total = 0.0f;
    for (int i = 0; i < args.nq; i++) {
        int hit = 0;
        for (int j = 0; j < args.k && j < (int)results[i].size(); j++) {
            idx_t id = results[i][j];
            for (int r = 0; r < args.k && r < (int)gt[i].size(); r++)
                if (gt[i][r] == id) { hit++; break; }
        }
        total += (float)hit / args.k;
    }
    float rec = total / args.nq;

    write_ids(args.ids_out, results);
    write_json(args.json_out, args.n, args.nq, dim, args.k, args.pq,
               ms(t0, t1), ms(t2, t3), ms(t_gt0, t_gt1), rec);

    std::cout << "[repro_cpu_raw] build_ms=" << ms(t0, t1)
              << " search_ms=" << ms(t2, t3)
              << " qps=" << (args.nq / (ms(t2, t3) / 1000.0))
              << " recall@" << args.k << "=" << rec << "\n";
    return 0;
}
