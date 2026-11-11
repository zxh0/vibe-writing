# 第6章：在装有CUDA的Linux机器（比如Colab）上用make编译。ARCH默认是T4的sm_75。
NVCC ?= nvcc
ARCH ?= sm_75
FLAGS = -O3 -arch=$(ARCH) -std=c++17
PROGS = blocks latency warps divergence occupancy

all: $(PROGS)

$(PROGS): %: %.cu common.cuh
	$(NVCC) $(FLAGS) -o $@ $<

clean:
	rm -f $(PROGS)
