#!/bin/bash
# 第6章：在T4上跑本章的全部实验，结果写到results.txt。书中引用的数字都来自这个文件。
set -e
cd "$(dirname "$0")"
make -s

best() {  # best <命令...>：运行三次，输出GFLOPS或GB/s那一列最大的一行
  for _ in 1 2 3; do "$@"; done |
    awk '{for(i=1;i<=NF;i++) if($i=="GFLOPS"||$i=="GB/s"||$i=="GFLOPS（按每个线程做的FMA计）"){print $(i-1) "\t" $0; break}}' |
    sort -g -r | head -1 | cut -f2-
}

{
echo "# GPU：$(nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit --format=csv,noheader)"
echo "# CUDA：$(nvcc --version | grep release)"
echo; echo "## 6.2 block如何分配到SM"
./blocks 40 1024
./blocks 64 1024
./blocks 160 256
./blocks 320 256
echo; echo "## 6.3 访存延迟"
./latency shared
./latency global
echo; echo "## 6.3 每个SM的warp数与访存带宽"
for t in 32 64 128 256 512 1024; do best ./warps read $t; done
echo; echo "## 6.3 每个SM的warp数与算力（每个线程一条FMA链）"
for t in 32 64 128 256 512 1024; do best ./warps fma $t; done
echo; echo "## 6.4 分支分歧"
for m in none warp lane; do best ./divergence $m; done
echo; echo "## 6.5 SM的上限"
./occupancy limits
echo; echo "## 6.5 朴素kernel的occupancy"
./occupancy naive
echo; echo "## 6.5 用共享内存压低occupancy"
./occupancy smem
echo; echo "## 测完时的GPU状态"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active --format=csv
} 2>&1 | tee results.txt
