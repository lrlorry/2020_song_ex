#pragma once
#include <cuda_runtime.h>
#include "config.h"

// GPU Bloom filter for visited-node tracking during graph search.
// Matches official SONG bloomfilter.h in structure:
//   - Hash function  : XOR-shift-multiply (same style as official)
//   - Bit layout     : word = pos % BF_WORDS, bit = pos / BF_WORDS
//   - Storage        : uint64_t data[BF_WORDS] in shared memory
//
// Only lane 0 calls these functions; no atomics needed.

// Random 64-bit constants for hash mixing (same style as official SONG)
__device__ __constant__ uint64_t BF_RAND[BF_NHASH * 2] = {
    0x4bcb391f924ed183ULL, 0xa0ab69ccd854fc0aULL,
    0x91086b9cecf5e3b7ULL, 0xc68e01641bead407ULL,
    0x3a7b976128a30449ULL, 0x6d122efabfc4d99fULL,
    0xe6700ef8715030e2ULL, 0x80dd0c3bffcfb45bULL,
};

// XOR-shift-multiply hash: maps node id x with hash index h to [0, BF_WORDS*64).
// Matches official SONG's hash() in bloomfilter.h.
__device__ inline int bf_hash(int h, uint64_t x) {
    x ^= x >> 33;
    x *= BF_RAND[h * 2];
    x ^= x >> 33;
    x *= BF_RAND[h * 2 + 1];
    x ^= x >> 33;
    return (int)(x % (uint64_t)(BF_WORDS * 64));
}

// Mark node x as visited. Bit layout: word = pos % BF_WORDS, bit = pos / BF_WORDS.
// Matches official SONG's set_bit() / add().
__device__ inline void bf_set(uint64_t* bloom, int x) {
    for (int h = 0; h < BF_NHASH; h++) {
        int pos = bf_hash(h, (uint64_t)x);
        bloom[pos % BF_WORDS] |= (1ULL << (pos / BF_WORDS));
    }
}

// Return true if node x is possibly marked (may have false positives).
// Matches official SONG's test_bit() / test().
__device__ inline bool bf_test(const uint64_t* __restrict__ bloom, int x) {
    for (int h = 0; h < BF_NHASH; h++) {
        int pos = bf_hash(h, (uint64_t)x);
        if (!((bloom[pos % BF_WORDS] >> (pos / BF_WORDS)) & 1ULL)) return false;
    }
    return true;
}
