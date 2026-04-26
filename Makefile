CXX    = g++
NVCC   = nvcc
PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip
CXXFLAGS  = -std=c++17 -O3
NVCCFLAGS = -std=c++17 -O3

ARCH ?= $(shell nvcc --list-gpu-code 2>/dev/null | grep -o 'sm_[0-9]*' | sort -t_ -k2 -n | tail -1)
ifeq ($(ARCH),)
  ARCH = sm_75
endif

.PHONY: all gpu cpu bench bench_cpu bench_gpu deps compare clean

all: gpu cpu

gpu: main.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -o song_gpu main.cu

cpu: main.cpp
	$(CXX) $(CXXFLAGS) -o song_cpu main.cpp

bench: bench_cpu bench_gpu

bench_cpu: bench_cpu.cpp
	$(CXX) $(CXXFLAGS) -o bench_cpu bench_cpu.cpp

bench_gpu: bench_gpu.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -o bench_gpu bench_gpu.cu

deps:
	$(PIP) install -q numpy matplotlib

compare: deps bench
	$(PYTHON) run_compare.py

clean:
	rm -f song_gpu song_cpu bench_cpu bench_gpu
