#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

// reads .fvecs file, returns all vectors
// path : file path to .fvecs file
//        binary layout per vector: [dim(int32), v0, v1, ..., v_{dim-1}(float32)]
std::vector<std::vector<float>> read_fvecs(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);

    std::vector<std::vector<float>> vecs;
    int dim;
    while (f.read(reinterpret_cast<char*>(&dim), 4)) {
        std::vector<float> v(dim);
        f.read(reinterpret_cast<char*>(v.data()), dim * 4);
        vecs.push_back(std::move(v));
    }
    return vecs;
}

// reads .ivecs file, returns all vectors
// path : file path to .ivecs file
//        binary layout per vector: [dim(int32), v0, v1, ..., v_{dim-1}(int32)]
std::vector<std::vector<int>> read_ivecs(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);

    std::vector<std::vector<int>> vecs;
    int dim;
    while (f.read(reinterpret_cast<char*>(&dim), 4)) {
        std::vector<int> v(dim);
        f.read(reinterpret_cast<char*>(v.data()), dim * 4);
        vecs.push_back(std::move(v));
    }
    return vecs;
}
