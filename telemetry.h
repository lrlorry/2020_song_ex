#pragma once

#include <vector>

struct QueryTrace {
    int seed_count = 0;
    int visited_nodes = 0;
    int expanded_nodes = 0;
    int neighbor_slots_scanned = 0;
    int candidate_nodes = 0;
    int distance_evals = 0;
    int frontier_pushes = 0;
    int result_updates = 0;
    int peak_frontier = 0;
    int peak_results = 0;
    int result_count = 0;
    int early_stopped = 0;
};

struct BatchTiming {
    int batch_index = 0;
    int batch_offset = 0;
    int batch_size = 0;
    double h2d_ms = 0.0;
    double memset_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
};

struct GpuSearchTrace {
    int ef_search = 0;
    int batch_limit = 0;
    double alloc_ms = 0.0;
    double h2d_ms = 0.0;
    double memset_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double total_ms = 0.0;
    std::vector<BatchTiming> batches;
    std::vector<QueryTrace> queries;
};
