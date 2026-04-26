#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
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
    std::string out_dir = "results/repro_cpu_raw";
    std::string ids_out;
    std::string json_out = "results/repro_cpu_raw.json";
    std::string pq_list = std::to_string(pq_size);
    int n = 10000;
    int nq = 1000;
    int k = TOPK;
    int repeats = 3;
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

struct DegreeStats {
    int min_degree = 0;
    int max_degree = 0;
    int zero_degree_nodes = 0;
    double mean_degree = 0.0;
    double std_degree = 0.0;
};

struct SweepRow {
    int pq = 0;
    double build_ms = 0.0;
    double gt_ms = 0.0;
    double search_ms_telemetry = 0.0;
    double search_ms_mean = 0.0;
    double search_ms_std = 0.0;
    double search_ms_min = 0.0;
    double search_ms_max = 0.0;
    double qps_mean = 0.0;
    double qps_std = 0.0;
    double recall = 0.0;
    Summary latency;
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
};

static Args parse_args(int argc, char** argv) {
    Args args;
    std::unordered_map<std::string, std::string> kv;
    for (int i = 1; i + 1 < argc; i += 2) kv[argv[i]] = argv[i + 1];
    if (kv.count("--base")) args.base = kv["--base"];
    if (kv.count("--query")) args.query = kv["--query"];
    if (kv.count("--gt-npy")) args.gt_npy = kv["--gt-npy"];
    if (kv.count("--out-dir")) args.out_dir = kv["--out-dir"];
    if (kv.count("--ids-out")) args.ids_out = kv["--ids-out"];
    if (kv.count("--json-out")) args.json_out = kv["--json-out"];
    if (kv.count("--pq-list")) args.pq_list = kv["--pq-list"];
    if (kv.count("--n")) args.n = std::stoi(kv["--n"]);
    if (kv.count("--nq")) args.nq = std::stoi(kv["--nq"]);
    if (kv.count("--k")) args.k = std::stoi(kv["--k"]);
    if (kv.count("--repeats")) args.repeats = std::stoi(kv["--repeats"]);
    if (kv.count("--pq-size")) args.pq_list = kv["--pq-size"];
    if (args.repeats < 1) args.repeats = 1;
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

static double recall_for_query(const std::vector<idx_t>& ids, const std::vector<idx_t>& gt, int k) {
    int hit = 0;
    for (int j = 0; j < k && j < (int)ids.size(); ++j) {
        idx_t id = ids[j];
        for (int r = 0; r < k && r < (int)gt.size(); ++r) {
            if (gt[r] == id) {
                hit++;
                break;
            }
        }
    }
    return (double)hit / k;
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

static DegreeStats compute_degree_stats(const Graph& g) {
    std::vector<int> degrees;
    degrees.reserve(g.num_nodes);
    for (int u = 0; u < g.num_nodes; ++u) {
        int cnt = 0;
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            if (g.adj[u * BLOCK_SIZE + j] >= 0) cnt++;
        }
        degrees.push_back(cnt);
    }
    DegreeStats stats;
    stats.min_degree = *std::min_element(degrees.begin(), degrees.end());
    stats.max_degree = *std::max_element(degrees.begin(), degrees.end());
    stats.zero_degree_nodes = (int)std::count(degrees.begin(), degrees.end(), 0);
    Summary s = summarize_ints(degrees);
    stats.mean_degree = s.mean;
    stats.std_degree = s.stddev;
    return stats;
}

static std::string pq_tag(int pq) {
    std::ostringstream oss;
    oss << "pq" << std::setw(4) << std::setfill('0') << pq;
    return oss.str();
}

static void write_query_stats_csv(
    const fs::path& path,
    int pq,
    const std::vector<double>& latencies_ms,
    const std::vector<double>& recalls,
    const std::vector<QueryTrace>& traces)
{
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    out << "query_id,pq,latency_ms,recall,seed_count,visited_nodes,expanded_nodes,neighbor_slots_scanned,"
           "candidate_nodes,distance_evals,frontier_pushes,result_updates,peak_frontier,peak_results,"
           "result_count,early_stopped\n";
    for (size_t i = 0; i < traces.size(); ++i) {
        const auto& t = traces[i];
        out << i << "," << pq << "," << latencies_ms[i] << "," << recalls[i] << ","
            << t.seed_count << "," << t.visited_nodes << "," << t.expanded_nodes << ","
            << t.neighbor_slots_scanned << "," << t.candidate_nodes << "," << t.distance_evals << ","
            << t.frontier_pushes << "," << t.result_updates << "," << t.peak_frontier << ","
            << t.peak_results << "," << t.result_count << "," << t.early_stopped << "\n";
    }
}

static void write_summary_csv(const fs::path& path, const std::vector<SweepRow>& rows) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    out << "method,pq,build_ms,gt_ms,search_ms_telemetry,search_ms_mean,search_ms_std,search_ms_min,search_ms_max,"
           "qps_mean,qps_std,recall,latency_mean_ms,latency_std_ms,latency_min_ms,latency_p50_ms,latency_p90_ms,"
           "latency_p95_ms,latency_p99_ms,latency_max_ms,recall_mean,recall_p50,recall_p95,visited_mean,visited_p95,"
           "expanded_mean,expanded_p95,neighbor_slots_mean,distance_evals_mean,distance_evals_p95,"
           "frontier_pushes_mean,result_updates_mean,peak_frontier_mean,peak_results_mean,early_stop_rate,"
           "ids_path,query_stats_path\n";
    for (const auto& row : rows) {
        out << "repro_cpu_raw" << "," << row.pq << "," << row.build_ms << "," << row.gt_ms << ","
            << row.search_ms_telemetry << "," << row.search_ms_mean << "," << row.search_ms_std << ","
            << row.search_ms_min << "," << row.search_ms_max << "," << row.qps_mean << "," << row.qps_std << ","
            << row.recall << "," << row.latency.mean << "," << row.latency.stddev << "," << row.latency.min << ","
            << row.latency.p50 << "," << row.latency.p90 << "," << row.latency.p95 << "," << row.latency.p99 << ","
            << row.latency.max << "," << row.recall_per_query.mean << "," << row.recall_per_query.p50 << ","
            << row.recall_per_query.p95 << "," << row.visited_nodes.mean << "," << row.visited_nodes.p95 << ","
            << row.expanded_nodes.mean << "," << row.expanded_nodes.p95 << "," << row.neighbor_slots_scanned.mean << ","
            << row.distance_evals.mean << "," << row.distance_evals.p95 << "," << row.frontier_pushes.mean << ","
            << row.result_updates.mean << "," << row.peak_frontier.mean << "," << row.peak_results.mean << ","
            << row.early_stop_rate << "," << row.ids_path << "," << row.query_stats_path << "\n";
    }
}

static void write_json(
    const fs::path& path,
    const Args& args,
    int dim,
    double base_load_ms,
    double query_load_ms,
    double flatten_ms,
    double gt_ms,
    double build_ms,
    const DegreeStats& degree_stats,
    const std::vector<SweepRow>& rows)
{
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    const SweepRow& primary = rows.back();
    out << "{\n";
    out << "  \"method\": \"repro_cpu_raw\",\n";
    out << "  \"n\": " << args.n << ",\n";
    out << "  \"nq\": " << args.nq << ",\n";
    out << "  \"dim\": " << dim << ",\n";
    out << "  \"k\": " << args.k << ",\n";
    out << "  \"repeats\": " << args.repeats << ",\n";
    out << "  \"base_load_ms\": " << base_load_ms << ",\n";
    out << "  \"query_load_ms\": " << query_load_ms << ",\n";
    out << "  \"flatten_ms\": " << flatten_ms << ",\n";
    out << "  \"gt_ms\": " << gt_ms << ",\n";
    out << "  \"build_ms\": " << build_ms << ",\n";
    out << "  \"primary_pq\": " << primary.pq << ",\n";
    out << "  \"search_ms\": " << primary.search_ms_mean << ",\n";
    out << "  \"qps\": " << primary.qps_mean << ",\n";
    out << "  \"recall_at_k\": " << primary.recall << ",\n";
    out << "  \"degree_stats\": {\n";
    out << "    \"min_degree\": " << degree_stats.min_degree << ",\n";
    out << "    \"max_degree\": " << degree_stats.max_degree << ",\n";
    out << "    \"zero_degree_nodes\": " << degree_stats.zero_degree_nodes << ",\n";
    out << "    \"mean_degree\": " << degree_stats.mean_degree << ",\n";
    out << "    \"std_degree\": " << degree_stats.std_degree << "\n";
    out << "  },\n";
    out << "  \"summary_csv\": \"" << (fs::path(args.out_dir) / "sweep_summary.csv").string() << "\",\n";
    out << "  \"pq_runs\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        out << "    {\n";
        out << "      \"pq\": " << row.pq << ",\n";
        out << "      \"search_ms_mean\": " << row.search_ms_mean << ",\n";
        out << "      \"search_ms_std\": " << row.search_ms_std << ",\n";
        out << "      \"qps_mean\": " << row.qps_mean << ",\n";
        out << "      \"recall_at_k\": " << row.recall << ",\n";
        out << "      \"ids_path\": \"" << row.ids_path << "\",\n";
        out << "      \"query_stats_path\": \"" << row.query_stats_path << "\"\n";
        out << "    }" << (i + 1 == rows.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
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
    auto t_flat1 = clk::now();

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

    auto t_build0 = clk::now();
    Graph g = build_graph(std::vector<dist_t>(data), args.n, dim);
    auto t_build1 = clk::now();

    const double base_load_ms = ms(t_load0, t_load1);
    const double query_load_ms = ms(t_load1, t_load2);
    const double flatten_ms = ms(t_flat0, t_flat1);
    const double gt_ms = ms(t_gt0, t_gt1);
    const double build_ms = ms(t_build0, t_build1);
    const DegreeStats degree_stats = compute_degree_stats(g);

    fs::create_directories(args.out_dir);
    std::vector<SweepRow> rows;

    for (int pq : pq_values) {
        std::vector<std::vector<idx_t>> results(args.nq);
        std::vector<QueryTrace> traces(args.nq);
        std::vector<double> latencies(args.nq);
        std::vector<double> recalls(args.nq);

        auto t_search0 = clk::now();
        for (int i = 0; i < args.nq; i++) {
            auto q0 = clk::now();
            results[i] = g.search(query[i].data(), args.k, pq, &traces[i]);
            auto q1 = clk::now();
            latencies[i] = ms(q0, q1);
            recalls[i] = recall_for_query(results[i], gt[i], args.k);
        }
        auto t_search1 = clk::now();
        const double telemetry_ms = ms(t_search0, t_search1);

        std::vector<double> repeat_times = {telemetry_ms};
        std::vector<double> repeat_qps = {(args.nq / (telemetry_ms / 1000.0))};
        for (int rep = 1; rep < args.repeats; ++rep) {
            auto t0 = clk::now();
            for (int i = 0; i < args.nq; ++i) {
                (void)g.search(query[i].data(), args.k, pq);
            }
            auto t1 = clk::now();
            double rep_ms = ms(t0, t1);
            repeat_times.push_back(rep_ms);
            repeat_qps.push_back(args.nq / (rep_ms / 1000.0));
        }

        const fs::path ids_path = fs::path(args.out_dir) / (pq_tag(pq) + ".ids");
        const fs::path query_csv_path = fs::path(args.out_dir) / (pq_tag(pq) + "_query_stats.csv");
        write_ids(ids_path.string(), results);
        write_query_stats_csv(query_csv_path, pq, latencies, recalls, traces);

        if (!args.ids_out.empty() && pq == pq_values.back()) {
            write_ids(args.ids_out, results);
        }

        std::vector<int> visited, expanded, neighbor_slots, candidate_nodes, distance_evals;
        std::vector<int> frontier_pushes, result_updates, peak_frontier, peak_results, early_stop;
        visited.reserve(args.nq);
        expanded.reserve(args.nq);
        neighbor_slots.reserve(args.nq);
        candidate_nodes.reserve(args.nq);
        distance_evals.reserve(args.nq);
        frontier_pushes.reserve(args.nq);
        result_updates.reserve(args.nq);
        peak_frontier.reserve(args.nq);
        peak_results.reserve(args.nq);
        early_stop.reserve(args.nq);
        for (const auto& trace : traces) {
            visited.push_back(trace.visited_nodes);
            expanded.push_back(trace.expanded_nodes);
            neighbor_slots.push_back(trace.neighbor_slots_scanned);
            candidate_nodes.push_back(trace.candidate_nodes);
            distance_evals.push_back(trace.distance_evals);
            frontier_pushes.push_back(trace.frontier_pushes);
            result_updates.push_back(trace.result_updates);
            peak_frontier.push_back(trace.peak_frontier);
            peak_results.push_back(trace.peak_results);
            early_stop.push_back(trace.early_stopped);
        }

        Summary repeat_time_stats = summarize(repeat_times);
        Summary qps_stats = summarize(repeat_qps);
        SweepRow row;
        row.pq = pq;
        row.build_ms = build_ms;
        row.gt_ms = gt_ms;
        row.search_ms_telemetry = telemetry_ms;
        row.search_ms_mean = repeat_time_stats.mean;
        row.search_ms_std = repeat_time_stats.stddev;
        row.search_ms_min = repeat_time_stats.min;
        row.search_ms_max = repeat_time_stats.max;
        row.qps_mean = qps_stats.mean;
        row.qps_std = qps_stats.stddev;
        row.recall = summarize(recalls).mean;
        row.latency = summarize(latencies);
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
        row.query_stats_path = query_csv_path.string();
        rows.push_back(row);

        std::cout << "[repro_cpu_raw] pq=" << pq
                  << " build_ms=" << build_ms
                  << " search_mean_ms=" << row.search_ms_mean
                  << " qps_mean=" << row.qps_mean
                  << " recall@" << args.k << "=" << row.recall << "\n";
    }

    write_summary_csv(fs::path(args.out_dir) / "sweep_summary.csv", rows);
    write_json(args.json_out, args, dim, base_load_ms, query_load_ms, flatten_ms, gt_ms, build_ms, degree_stats, rows);
    return 0;
}
