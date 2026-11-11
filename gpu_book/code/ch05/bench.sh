#!/bin/bash
# 第5章：在T4上跑本章的全部实验，结果写到results.txt。书中引用的数字都来自这个文件。
# 每个实验跑三遍，取最快的一次（每一遍内部已经预热并多次测量取平均）。
set -e
cd "$(dirname "$0")"
make -s

best() {  # best <命令...>：运行三次，输出GFLOPS或GB/s那一列最大的一行
  for _ in 1 2 3; do "$@"; done |
    awk '{for(i=1;i<=NF;i++) if($i=="GFLOPS"||$i=="GB/s"){print $(i-1) "\t" $0; break}}' |
    sort -g -r | head -1 | cut -f2-
}

{
echo "# GPU：$(nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv,noheader)"
echo "# CUDA：$(nvcc --version | grep release)"
echo "# CPU：$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)，$(nproc)个逻辑CPU"
echo; echo "## 5.4 向量加法"
./vector_add
echo; echo "## 5.6 正确性"
./matmul check 1000
./matmul check 1024
echo; echo "## 5.7 朴素kernel，不同规模"
for n in 256 512 1024 2048 4096; do best ./matmul naive $n; done
echo; echo "## 5.7 朴素kernel，不同block大小（n=4096）"
for bs in 8 16 32; do best ./matmul naive 4096 $bs; done
echo; echo "## 5.8 cuBLAS"
for n in 256 512 1024 2048 4096; do best ./matmul cublas $n; done
echo; echo "## 实测T4的两块天花板"
best ./peak
for m in read copy; do best ./bandwidth $m; done
echo; echo "## 测完时的GPU状态"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active --format=csv
} 2>&1 | tee results.txt
