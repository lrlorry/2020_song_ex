#!/bin/bash
set -e

mkdir -p data/nytimes

# NYTimes 256-dim angular (cosine) from ANN benchmarks
wget -c http://ann-benchmarks.com/nytimes-256-angular.hdf5 \
    -O data/nytimes/nytimes-256-angular.hdf5

ls -lh data/nytimes/
