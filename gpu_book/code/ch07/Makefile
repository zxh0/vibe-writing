# 第7章：在装有CUDA的Linux机器（比如Colab）上用make编译。ARCH默认是T4的sm_75。
NVCC ?= nvcc
ARCH ?= sm_75
FLAGS = -O3 -arch=$(ARCH) -std=c++17 -lineinfo

all: coalesce matmul

coalesce: coalesce.cu common.cuh
	$(NVCC) $(FLAGS) -o $@ $<

matmul: matmul.cu common.cuh
	$(NVCC) $(FLAGS) -o $@ $< -lcublas

clean:
	rm -f coalesce matmul
