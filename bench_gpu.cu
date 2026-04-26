#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
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
    std::string gt_npy;
    std::string out_dir = "results/repro_gpu";
    std::string raw_ids_out;
    std::string hash_ids_out;
    std::string json_out = "results/repro_gpu.json";
    std::string pq_list = std::to_string(pq_size);
    int n = 10000;
    int nq = 1000;
    int k = TOPK;
    int repeats = 3;
    int batch_size = N_MULTIQUERY;
};

struct Summary {
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
};

struct SweepRow {
    std::string method;
    int pq = 0;
    double build_ms = 0.0;
    double upload_ms = 0.0;
    double query_projection_ms = 0.0;
    double search_ms_telemetry = 0.0;
    double search_ms_mean = 0.0;
    double search_ms_std = 0.0;
    double search_ms_min = 0.0;
    double search_ms_max = 0.0;
    double qps_mean = 0.0;
    double qps_std = 0.0;
    double recall = 0.0;
    double alloc_ms = 0.0;
    double h2d_ms = 0.0;
    double memset_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    Summary approx_latency;
    Summary recall_per_query;
    Summary visited_nodes;
    Summary expanded_nodes;
    Summary neighbor_slots_scanned;
    Summary candidate_nodes;
    Summary distance_evals;
    Summary frontier_pushes;
    Summary result_updates;
    Summary peak_frontier;
    Summary peak_results;
    double early_stop_rate = 0.0;
    std::string ids_path;
    std::string query_stats_path;
    std::string batch_stats_path;
};

static Args parse_args(int argc, char** argv) {
    Args args;
    std::unordered_map<std::string, std::string> kv;
    for (int i = 1; i + 1 < argc; i += 2) kv[argv[i]] = argv[i + 1];
    if (kv.count("--base")) args.base = kv["--base"];
    if (kv.count("--query")) args.query = kv["--query"];
    if (kv.count("--gt-npy")) args.gt_npy = kv["--gt-npy"];
    if (kv.count("--out-dir")) args.out_dir = kv["--out-dir"];
    if (kv.count("--raw-ids-out")) args.raw_ids_out = kv["--raw-ids-out"];
    if (kv.count("--hash-ids-out")) args.hash_ids_out = kv["--hash-ids-out"];
    if (kv.count("--json-out")) args.json_out = kv["--json-out"];
    if (kv.count("--pq-list")) args.pq_list = kv["--pq-list"];
    if (kv.count("--n")) args.n = std::stoi(kv["--n"]);
    if (kv.count("--nq")) args.nq = std::stoi(kv["--nq"]);
    if (kv.count("--k")) args.k = std::stoi(kv["--k"]);
    if (kv.count("--repeats")) args.repeats = std::stoi(kv["--repeats"]);
    if (kv.count("--batch-size")) args.batch_size = std::stoi(kv["--batch-size"]);
    if (kv.count("--pq-size")) args.pq_list = kv["--pq-size"];
    if (args.repeats < 1) args.repeats = 1;
    if (args.batch_size < 1) args.batch_size = 1;
    return args;
}

static std::vector<int> parse_int_list(const std::string& text) {
    std::vector<int> values;
    std::stringstream ss(text);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        values.push_back(std::stoi(tok));
    }
    if (values.empty()) values.push_back(pq_size);
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

static double quantile_sorted(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    if (sorted.size() == 1) return sorted[0];
    double pos = q * (sorted.size() - 1);
    size_t lo = (size_t)std::floor(pos);
    size_t hi = (size_t)std::ceil(pos);
    double frac = pos - lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

static Summary summarize(const std::vector<double>& values) {
    Summary s;
    if (values.empty()) return s;
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    s.min = sorted.front();
    s.max = sorted.back();
    s.mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    double var = 0.0;
    for (double v : sorted) {
        double d = v - s.mean;
        var += d * d;
    }
    s.stddev = std::sqrt(var / sorted.size());
    s.p50 = quantile_sorted(sorted, 0.50);
    s.p90 = quantile_sorted(sorted, 0.90);
    s.p95 = quantile_sorted(sorted, 0.95);
    s.p99 = quantile_sorted(sorted, 0.99);
    return s;
}

static Summary summarize_ints(const std::vector<int>& values) {
    std::vector<double> as_double(values.begin(), values.end());
    return summarize(as_double);
}

static std::vector<std::vector<int>> load_gt_npy(const std::string& path, int nq, int k) {
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

    std::vector<std::vector<int>> gt(nq, std::vector<int>(k));
    for (int i = 0; i < nq; ++i)
        for (int j = 0; j < k; ++j)
            gt[i][j] = static_cast<int>(raw[(size_t)i * cols + j]);
    return gt;
}

static double recall_for_query_flat(const std::vector<int>& ids, size_t offset, const std::vector<int>& gt, int k) {
    int hit = 0;
    for (int j = 0; j < k; ++j) {
        int id = ids[offset + j];
        for (int r = 0; r < k && r < (int)gt.size(); ++r) {
            if (gt[r] == id) {
                hit++;
                break;
            }
        }
    }
    return (double)hit / k;
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

static std::string pq_tag(int pq) {
    std::ostringstream oss;
    oss << "pq" << std::setw(4) << std::setfill('0') << pq;
    return oss.str();
}

static void write_batch_stats_csv(const fs::path& path, int pq, const GpuSearchTrace& trace) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    out << "pq,batch_index,batch_offset,batch_size,h2d_ms,memset_ms,kernel_ms,d2h_ms,total_ms,per_query_total_ms\n";
    for (const auto& batch : trace.batches) {
        double total = batch.h2d_ms + batch.memset_ms + batch.kernel_ms + batch.d2h_ms;
        out << pq << "," << batch.batch_index << "," << batch.batch_offset << "," << batch.batch_size << ","
            << batch.h2d_ms << "," << batch.memset_ms << "," << batch.kernel_ms << "," << batch.d2h_ms << ","
            << total << "," << (total / batch.batch_size) << "\n";
    }
}

static void write_query_stats_csv(
    const fs::path& path,
    int pq,
    const GpuSearchTrace& trace,
    const std::vector<double>& recalls)
{
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    out << "query_id,pq,batch_index,approx_h2d_ms,approx_memset_ms,approx_kernel_ms,approx_d2h_ms,approx_total_ms,"
           "recall,seed_count,visited_nodes,expanded_nodes,neighbor_slots_scanned,candidate_nodes,distance_evals,"
           "frontier_pushes,result_updates,peak_frontier,peak_results,result_count,early_stopped\n";
    for (size_t i = 0; i < trace.queries.size(); ++i) {
        const auto& q = trace.queries[i];
        const auto& batch = trace.batches[i / trace.batch_limit];
        double h2d = batch.h2d_ms / batch.batch_size;
        double memset = batch.memset_ms / batch.batch_size;
        double kernel = batch.kernel_ms / batch.batch_size;
        double d2h = batch.d2h_ms / batch.batch_size;
        out << i << "," << pq << "," << batch.batch_index << "," << h2d << "," << memset << "," << kernel << ","
            << d2h << "," << (h2d + memset + kernel + d2h) << "," << recalls[i] << "," << q.seed_count << ","
            << q.visited_nodes << "," << q.expanded_nodes << "," << q.neighbor_slots_scanned << ","
            << q.candidate_nodes << "," << q.distance_evals << "," << q.frontier_pushes << "," << q.result_updates
            << "," << q.peak_frontier << "," << q.peak_results << "," << q.result_count << ","
            << q.early_stopped << "\n";
    }
}

static void write_summary_csv(const fs::path& path, const std::vector<SweepRow>& rows) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    out << "method,pq,build_ms,upload_ms,query_projection_ms,search_ms_telemetry,search_ms_mean,search_ms_std,"
           "search_ms_min,search_ms_max,qps_mean,qps_std,recall,alloc_ms,h2d_ms,memset_ms,kernel_ms,d2h_ms,"
           "approx_latency_mean_ms,approx_latency_p95_ms,recall_mean,recall_p95,visited_mean,visited_p95,"
           "expanded_mean,expanded_p95,neighbor_slots_mean,distance_evals_mean,distance_evals_p95,"
           "frontier_pushes_mean,result_updates_mean,peak_frontier_mean,peak_results_mean,early_stop_rate,"
           "ids_path,query_stats_path,batch_stats_path\n";
    for (const auto& row : rows) {
        out << row.method << "," << row.pq << "," << row.build_ms << "," << row.upload_ms << ","
            << row.query_projection_ms << "," << row.search_ms_telemetry << "," << row.search_ms_mean << ","
            << row.search_ms_std << "," << row.search_ms_min << "," << row.search_ms_max << "," << row.qps_mean
            << "," << row.qps_std << "," << row.recall << "," << row.alloc_ms << "," << row.h2d_ms << ","
            << row.memset_ms << "," << row.kernel_ms << "," << row.d2h_ms << "," << row.approx_latency.mean << ","
            << row.approx_latency.p95 << "," << row.recall_per_query.mean << "," << row.recall_per_query.p95 << ","
            << row.visited_nodes.mean << "," << row.visited_nodes.p95 << "," << row.expanded_nodes.mean << ","
            << row.expanded_nodes.p95 << "," << row.neighbor_slots_scanned.mean << "," << row.distance_evals.mean
            << "," << row.distance_evals.p95 << "," << row.frontier_pushes.mean << "," << row.result_updates.mean
            << "," << row.peak_frontier.mean << "," << row.peak_results.mean << "," << row.early_stop_rate << ","
            << row.ids_path << "," << row.query_stats_path << "," << row.batch_stats_path << "\n";
    }
}

static void write_json(const fs::path& path, const Args& args, int dim,
                       double base_load_ms, double query_load_ms, double flatten_ms,
                       double raw_build_ms, double raw_upload_ms,
                       double hash_build_ms, double hash_projection_ms, double hash_upload_ms,
                       const std::vector<SweepRow>& rows) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    const SweepRow* raw_primary = nullptr;
    const SweepRow* hash_primary = nullptr;
    for (const auto& row : rows) {
        if (row.method == "repro_gpu_raw") raw_primary = &row;
        if (row.method == "repro_gpu_hash") hash_primary = &row;
    }
    out << "{\n";
    out << "  \"n\": " << args.n << ",\n";
    out << "  \"nq\": " << args.nq << ",\n";
    out << "  \"dim\": " << dim << ",\n";
    out << "  \"k\": " << args.k << ",\n";
    out << "  \"repeats\": " << args.repeats << ",\n";
    out << "  \"batch_size\": " << args.batch_size << ",\n";
    out << "  \"base_load_ms\": " << base_load_ms << ",\n";
    out << "  \"query_load_ms\": " << query_load_ms << ",\n";
    out << "  \"flatten_ms\": " << flatten_ms << ",\n";
    out << "  \"raw_build_ms\": " << raw_build_ms << ",\n";
    out << "  \"raw_upload_ms\": " << raw_upload_ms << ",\n";
    out << "  \"hash_build_ms\": " << hash_build_ms << ",\n";
    out << "  \"hash_projection_ms\": " << hash_projection_ms << ",\n";
    out << "  \"hash_upload_ms\": " << hash_upload_ms << ",\n";
    if (raw_primary) {
        out << "  \"raw_search_ms\": " << raw_primary->search_ms_mean << ",\n";
        out << "  \"raw_qps\": " << raw_primary->qps_mean << ",\n";
    }
    if (hash_primary) {
        out << "  \"hash_search_ms\": " << hash_primary->search_ms_mean << ",\n";
        out << "  \"hash_qps\": " << hash_primary->qps_mean << ",\n";
    }
    out << "  \"summary_csv\": \"" << (fs::path(args.out_dir) / "sweep_summary.csv").string() << "\"\n";
    out << "}\n";
}

template <typename SearchFn>
static SweepRow run_variant(
    const std::string& method,
    const Args& args,
    int pq,
    int nq,
    int k,
    SearchFn&& search_fn,
    const std::vector<std::vector<int>>& gt,
    const fs::path& out_dir)
{
    GpuSearchTrace first_trace;
    std::vector<int> ids = search_fn(pq, &first_trace);

    std::vector<double> repeat_times = {first_trace.total_ms};
    std::vector<double> repeat_qps = {(nq / (first_trace.total_ms / 1000.0))};
    for (int rep = 1; rep < args.repeats; ++rep) {
        GpuSearchTrace rep_trace;
        (void)search_fn(pq, &rep_trace);
        repeat_times.push_back(rep_trace.total_ms);
        repeat_qps.push_back(nq / (rep_trace.total_ms / 1000.0));
    }

    std::vector<double> recalls(nq, 0.0);
    if (!gt.empty()) {
        for (int i = 0; i < nq; ++i) {
            recalls[i] = recall_for_query_flat(ids, (size_t)i * k, gt[i], k);
        }
    }

    const fs::path ids_path = out_dir / (method + "_" + pq_tag(pq) + ".ids");
    const fs::path query_stats_path = out_dir / (method + "_" + pq_tag(pq) + "_query_stats.csv");
    const fs::path batch_stats_path = out_dir / (method + "_" + pq_tag(pq) + "_batch_stats.csv");
    write_flat_ids(ids_path.string(), ids, nq, k);
    write_query_stats_csv(query_stats_path, pq, first_trace, recalls);
    write_batch_stats_csv(batch_stats_path, pq, first_trace);

    std::vector<double> approx_latency;
    approx_latency.reserve(nq);
    std::vector<int> visited, expanded, neighbor_slots, candidate_nodes, distance_evals;
    std::vector<int> frontier_pushes, result_updates, peak_frontier, peak_results, early_stop;
    visited.reserve(nq);
    expanded.reserve(nq);
    neighbor_slots.reserve(nq);
    candidate_nodes.reserve(nq);
    distance_evals.reserve(nq);
    frontier_pushes.reserve(nq);
    result_updates.reserve(nq);
    peak_frontier.reserve(nq);
    peak_results.reserve(nq);
    early_stop.reserve(nq);
    for (size_t i = 0; i < first_trace.queries.size(); ++i) {
        const auto& q = first_trace.queries[i];
        const auto& batch = first_trace.batches[i / first_trace.batch_limit];
        approx_latency.push_back((batch.h2d_ms + batch.memset_ms + batch.kernel_ms + batch.d2h_ms) / batch.batch_size);
        visited.push_back(q.visited_nodes);
        expanded.push_back(q.expanded_nodes);
        neighbor_slots.push_back(q.neighbor_slots_scanned);
        candidate_nodes.push_back(q.candidate_nodes);
        distance_evals.push_back(q.distance_evals);
        frontier_pushes.push_back(q.frontier_pushes);
        result_updates.push_back(q.result_updates);
        peak_frontier.push_back(q.peak_frontier);
        peak_results.push_back(q.peak_results);
        early_stop.push_back(q.early_stopped);
    }

    SweepRow row;
    row.method = method;
    row.pq = pq;
    row.search_ms_telemetry = first_trace.total_ms;
    Summary repeat_time_stats = summarize(repeat_times);
    Summary qps_stats = summarize(repeat_qps);
    row.search_ms_mean = repeat_time_stats.mean;
    row.search_ms_std = repeat_time_stats.stddev;
    row.search_ms_min = repeat_time_stats.min;
    row.search_ms_max = repeat_time_stats.max;
    row.qps_mean = qps_stats.mean;
    row.qps_std = qps_stats.stddev;
    row.recall = gt.empty() ? 0.0 : summarize(recalls).mean;
    row.alloc_ms = first_trace.alloc_ms;
    row.h2d_ms = first_trace.h2d_ms;
    row.memset_ms = first_trace.memset_ms;
    row.kernel_ms = first_trace.kernel_ms;
    row.d2h_ms = first_trace.d2h_ms;
    row.approx_latency = summarize(approx_latency);
    row.recall_per_query = summarize(recalls);
    row.visited_nodes = summarize_ints(visited);
    row.expanded_nodes = summarize_ints(expanded);
    row.neighbor_slots_scanned = summarize_ints(neighbor_slots);
    row.candidate_nodes = summarize_ints(candidate_nodes);
    row.distance_evals = summarize_ints(distance_evals);
    row.frontier_pushes = summarize_ints(frontier_pushes);
    row.result_updates = summarize_ints(result_updates);
    row.peak_frontier = summarize_ints(peak_frontier);
    row.peak_results = summarize_ints(peak_results);
    row.early_stop_rate = summarize_ints(early_stop).mean;
    row.ids_path = ids_path.string();
    row.query_stats_path = query_stats_path.string();
    row.batch_stats_path = batch_stats_path.string();
    return row;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    const std::vector<int> pq_values = parse_int_list(args.pq_list);

    auto t_load0 = clk::now();
    auto base = read_fvecs(args.base);
    auto t_load1 = clk::now();
    auto query = read_fvecs(args.query);
    auto t_load2 = clk::now();
    if (base.empty() || query.empty()) {
        std::cerr << "empty base/query\n";
        return 1;
    }
    if (args.n > (int)base.size()) args.n = (int)base.size();
    if (args.nq > (int)query.size()) args.nq = (int)query.size();
    const int dim = (int)base[0].size();

    auto t_flat0 = clk::now();
    std::vector<dist_t> data((size_t)args.n * dim);
    for (int i = 0; i < args.n; i++)
        for (int j = 0; j < dim; j++)
            data[(size_t)i * dim + j] = base[i][j];

    std::vector<float> qdata((size_t)args.nq * dim);
    for (int i = 0; i < args.nq; i++)
        for (int j = 0; j < dim; j++)
            qdata[(size_t)i * dim + j] = query[i][j];
    auto t_flat1 = clk::now();

    std::vector<std::vector<int>> gt;
    if (!args.gt_npy.empty()) gt = load_gt_npy(args.gt_npy, args.nq, args.k);

    auto t_raw0 = clk::now();
    KernelGraph kg = build_kernel_graph(std::vector<dist_t>(data), args.n, dim);
    auto t_raw1 = clk::now();
    auto t_raw_up0 = clk::now();
    kg.upload_to_gpu();
    auto t_raw_up1 = clk::now();

    auto t_hash0 = clk::now();
    HashKernelGraph hkg = build_hashed_graph(data.data(), args.n, dim);
    auto t_hash1 = clk::now();

    auto t_proj0 = clk::now();
    std::vector<uint32_t> qhash((size_t)args.nq * HASH_WORDS);
    for (int i = 0; i < args.nq; i++)
        hkg.bh.project(query[i].data(), qhash.data() + (size_t)i * HASH_WORDS);
    auto t_proj1 = clk::now();

    auto t_hash_up0 = clk::now();
    hkg.upload_to_gpu();
    auto t_hash_up1 = clk::now();

    const double base_load_ms = ms(t_load0, t_load1);
    const double query_load_ms = ms(t_load1, t_load2);
    const double flatten_ms = ms(t_flat0, t_flat1);
    const double raw_build_ms = ms(t_raw0, t_raw1);
    const double raw_upload_ms = ms(t_raw_up0, t_raw_up1);
    const double hash_build_ms = ms(t_hash0, t_hash1);
    const double hash_projection_ms = ms(t_proj0, t_proj1);
    const double hash_upload_ms = ms(t_hash_up0, t_hash_up1);

    fs::create_directories(args.out_dir);
    std::vector<SweepRow> rows;

    for (int pq : pq_values) {
        SweepRow raw_row = run_variant(
            "repro_gpu_raw",
            args,
            pq,
            args.nq,
            args.k,
            [&](int ef_search, GpuSearchTrace* trace) {
                return gpu_search(kg, qdata.data(), args.nq, args.k, args.batch_size, ef_search, trace);
            },
            gt,
            args.out_dir);
        raw_row.build_ms = raw_build_ms;
        raw_row.upload_ms = raw_upload_ms;
        rows.push_back(raw_row);
        if (!args.raw_ids_out.empty() && pq == pq_values.back()) {
            fs::copy_file(raw_row.ids_path, args.raw_ids_out, fs::copy_options::overwrite_existing);
        }

        SweepRow hash_row = run_variant(
            "repro_gpu_hash",
            args,
            pq,
            args.nq,
            args.k,
            [&](int ef_search, GpuSearchTrace* trace) {
                return gpu_search(hkg, qhash.data(), args.nq, args.k, args.batch_size, ef_search, trace);
            },
            gt,
            args.out_dir);
        hash_row.build_ms = hash_build_ms;
        hash_row.upload_ms = hash_upload_ms;
        hash_row.query_projection_ms = hash_projection_ms;
        rows.push_back(hash_row);
        if (!args.hash_ids_out.empty() && pq == pq_values.back()) {
            fs::copy_file(hash_row.ids_path, args.hash_ids_out, fs::copy_options::overwrite_existing);
        }

        std::cout << "[repro_gpu_raw] pq=" << pq
                  << " build_ms=" << raw_row.build_ms
                  << " upload_ms=" << raw_row.upload_ms
                  << " search_mean_ms=" << raw_row.search_ms_mean
                  << " qps_mean=" << raw_row.qps_mean
                  << " recall@" << args.k << "=" << raw_row.recall << "\n";
        std::cout << "[repro_gpu_hash] pq=" << pq
                  << " build_ms=" << hash_row.build_ms
                  << " upload_ms=" << hash_row.upload_ms
                  << " search_mean_ms=" << hash_row.search_ms_mean
                  << " qps_mean=" << hash_row.qps_mean
                  << " recall@" << args.k << "=" << hash_row.recall << "\n";
    }

    write_summary_csv(fs::path(args.out_dir) / "sweep_summary.csv", rows);
    write_json(args.json_out, args, dim, base_load_ms, query_load_ms, flatten_ms,
               raw_build_ms, raw_upload_ms, hash_build_ms, hash_projection_ms, hash_upload_ms, rows);
    return 0;
}
