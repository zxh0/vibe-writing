# 第5章：在装有CUDA的Linux机器（比如Colab）上用make编译。ARCH默认是T4的sm_75。
NVCC ?= nvcc
ARCH ?= sm_75
FLAGS = -O3 -arch=$(ARCH) -std=c++17

all: vector_add peak bandwidth matmul

vector_add peak bandwidth: %: %.cu common.cuh
	$(NVCC) $(FLAGS) -o $@ $<

matmul: matmul.cu common.cuh
	$(NVCC) $(FLAGS) -o $@ $< -lcublas

clean:
	rm -f vector_add peak bandwidth matmul
