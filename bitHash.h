#pragma once
#include <vector>
#include <random>
#include <cstdint>
#include "config.h"

// Hamming distance between two HASH_WORDS-word bit vectors
inline int hamming(const uint32_t* a, const uint32_t* b) {
    int dist = 0;
    for (int i = 0; i < HASH_WORDS; i++)
        dist += __builtin_popcount(a[i] ^ b[i]);
    return dist;
}

// 1-bit Gaussian random projection: projects a float vector to hash_bits binary bits.
// Each bit = sign(R[row] · vec), where R is a random Gaussian matrix.
struct BitHash {
    int dim;                 // input vector dimensionality
    std::vector<float> R;   // random projection matrix, row-major [hash_bits x dim]

    // dim  : input vector dimensionality
    // seed : RNG seed (same seed → same projection matrix, required for build/query consistency)
    explicit BitHash(int dim, unsigned seed = 42) : dim(dim), R(hash_bits * dim) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        for (auto& v : R) v = gauss(rng);
    }

    // Project one float vector to HASH_WORDS packed uint32_t words.
    // vec : input vector, length = dim
    // out : output bit vector, length = HASH_WORDS
    void project(const float* vec, uint32_t* out) const {
        for (int w = 0; w < HASH_WORDS; w++) {       // w  : word index (0..HASH_WORDS-1)
            uint32_t word = 0;
            for (int b = 0; b < 32; b++) {            // b  : bit index within word (0..31)
                int   row = w * 32 + b;               // row: index of this hyperplane in R
                float dot = 0;
                for (int d = 0; d < dim; d++)
                    dot += R[row * dim + d] * vec[d]; // dot: projection onto hyperplane row
                if (dot > 0) word |= (1u << b);       // bit = sign of dot product
            }
            out[w] = word;
        }
    }

    // Project entire dataset to bit vectors.
    // data : input row-major dataset [n x dim]
    // n    : number of vectors
    // returns: bit vectors row-major [n x HASH_WORDS]
    std::vector<uint32_t> project_all(const float* data, int n) const {
        std::vector<uint32_t> out(n * HASH_WORDS);
        for (int i = 0; i < n; i++)
            project(data + (size_t)i * dim, out.data() + (size_t)i * HASH_WORDS);
        return out;
    }
};
