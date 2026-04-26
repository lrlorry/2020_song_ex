#!/bin/bash
set -e

mkdir -p data/sift

wget -c ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz -O data/sift.tar.gz
tar -xzf data/sift.tar.gz -C data/sift --strip-components=1

ls -lh data/sift/
