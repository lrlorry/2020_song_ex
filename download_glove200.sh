#!/bin/bash
set -e

mkdir -p data/glove200

# GloVe 200-dim angular (cosine) from ANN benchmarks
wget -c http://ann-benchmarks.com/glove-200-angular.hdf5 \
    -O data/glove200/glove-200-angular.hdf5

ls -lh data/glove200/
