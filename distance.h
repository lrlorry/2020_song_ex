#pragma once
#include "config.h"

// squared L2 distance between two vectors of length dim
inline dist_t l2(const dist_t* a, const dist_t* b, int dim) {
    dist_t sum = 0;
    for (int i = 0; i < dim; i++) {
        dist_t d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}
