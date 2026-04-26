#pragma once
#include <cuda_runtime.h>

// Candidates: fixed-size min-heap, BFS exploration queue.
// Holds up to MAX_SIZE (dist, id) pairs; pop_min always returns the closest node.
// When full, a new entry evicts the farthest one (bounds the frontier size).
// Lives in shared memory; managed by lane 0 only.
template<int MAX_SIZE>
struct Candidates {
    float dist[MAX_SIZE];  // distances of queued candidates
    int   id[MAX_SIZE];    // node ids of queued candidates
    int   sz;              // current number of entries

    __device__ void init()  { sz = 0; }
    __device__ bool empty() const { return sz == 0; }
    __device__ int  size()  const { return sz; }

    // Push (d, node). If full, evict the farthest entry so the frontier stays bounded.
    __device__ void push(float d, int node) {
        if (sz < MAX_SIZE) {
            dist[sz] = d; id[sz] = node; sz++;
            return;
        }
        int fi = 0;  // fi: index of the farthest current entry
        for (int i = 1; i < MAX_SIZE; i++) if (dist[i] > dist[fi]) fi = i;
        if (d < dist[fi]) { dist[fi] = d; id[fi] = node; }
    }

    // Pop and return the closest node; write its distance to *out_dist.
    __device__ int pop_min(float* out_dist) {
        int mi = 0;  // mi: index of the closest entry
        for (int i = 1; i < sz; i++) if (dist[i] < dist[mi]) mi = i;
        *out_dist = dist[mi];
        int node  = id[mi];
        // fill the gap by swapping with the last entry
        dist[mi]  = dist[sz - 1];
        id[mi]    = id[sz - 1];
        sz--;
        return node;
    }
};

// Results: fixed-size max-heap, tracks the ef_search best nodes found so far.
// top = worst (farthest) entry = pruning threshold:
//   a new node enters only if its distance beats this threshold.
// At the end: trim to k via pop_worst(), then read ids with drain_best().
template<int MAX_SIZE>
struct Results {
    float dist[MAX_SIZE];  // distances of the best nodes found so far
    int   id[MAX_SIZE];    // node ids of the best nodes found so far
    int   sz;              // current number of entries

    __device__ void init()  { sz = 0; }
    __device__ int  size()  const { return sz; }
    __device__ bool full()  const { return sz == MAX_SIZE; }

    // Worst (largest) distance — incoming nodes must beat this to be admitted.
    __device__ float worst() const {
        float m = dist[0];
        for (int i = 1; i < sz; i++) if (dist[i] > m) m = dist[i];
        return m;
    }

    // Insert (d, node) if it improves the result set (better than worst or not yet full).
    __device__ bool try_push(float d, int node) {
        if (sz < MAX_SIZE) {
            dist[sz] = d; id[sz] = node; sz++;
            return true;
        }
        int wi = 0;  // wi: index of the worst current entry
        for (int i = 1; i < MAX_SIZE; i++) if (dist[i] > dist[wi]) wi = i;
        if (d < dist[wi]) { dist[wi] = d; id[wi] = node; return true; }
        return false;
    }

    // Evict the worst entry (used to trim ef_search results down to k).
    __device__ void pop_worst() {
        int wi = 0;  // wi: index of the worst (farthest) entry
        for (int i = 1; i < sz; i++) if (dist[i] > dist[wi]) wi = i;
        dist[wi] = dist[sz - 1];
        id[wi]   = id[sz - 1];
        sz--;
    }

    // Write the k best ids into out[] in best-first order; returns actual count written.
    __device__ int drain_best(int* out, int k) {
        int count = sz < k ? sz : k;  // count: number of entries to extract
        for (int i = 0; i < count; i++) {
            int mi = 0;  // mi: index of best remaining entry
            for (int j = 1; j < sz; j++) if (dist[j] < dist[mi]) mi = j;
            out[i] = id[mi];
            dist[mi] = dist[sz - 1]; id[mi] = id[sz - 1]; sz--;
        }
        return count;
    }
};
