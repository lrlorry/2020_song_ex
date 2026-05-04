#!/bin/bash
set -e

mkdir -p data/gist

wget -c ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz -O data/gist.tar.gz
tar -xzf data/gist.tar.gz -C data/gist --strip-components=1

ls -lh data/gist/
