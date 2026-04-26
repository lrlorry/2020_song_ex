#pragma once
#include <cstdint>

using idx_t  = int32_t;  // node index type
using dist_t = float;    // distance value type

// BLOCK_SIZE: slots allocated per node in the flat adj array (= FIXED_DEGREE + 1;
//             the extra slot ensures a -1 sentinel is always present)
constexpr int BLOCK_SIZE      = 32;

// FIXED_DEGREE: max outgoing edges per node written during build (< BLOCK_SIZE)
constexpr int FIXED_DEGREE    = 31;

// SEARCH_DEGREE: number of neighbors a newly inserted node keeps initially;
//                matches official SONG SEARCH_DEGREE = 15. Reverse edges may
//                later increase a node up to FIXED_DEGREE.
constexpr int SEARCH_DEGREE   = 15;

// EF_CONSTRUCTION: BFS beam width when inserting node u — how many nearest candidates
//                  to keep exploring during build-time graph search.
//                  Controls graph quality: larger → more edges explored per insertion →
//                  better-connected graph → higher recall at query time, but slower build.
//                  Matches official SONG CONSTRUCT_SEARCH_BUDGET = 150.
constexpr int EF_CONSTRUCTION = 150;

// pq_size: exploration width at query time; Candidates and Results heaps are both
//          capped at this size. Larger → better recall, more work per query.
constexpr int pq_size         = 50;

// TOPK: number of nearest neighbors to return per query
constexpr int TOPK            = 10;

// N_MULTIPROBE: number of entry points used to seed each search
constexpr int N_MULTIPROBE    = 4;

// N_MULTIQUERY: queries per GPU kernel launch; controls peak memory for visited flags
//               (visited = N_MULTIQUERY * n bytes — tune down if OOM)
constexpr int N_MULTIQUERY    = 512;

// hash_bits: number of output bits from the random projection (BitHash)
constexpr int hash_bits       = 256;

// HASH_WORDS: hash_bits packed into 32-bit words for __popc-based Hamming distance
constexpr int HASH_WORDS      = hash_bits / 32;
