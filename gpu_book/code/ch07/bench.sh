#!/bin/bash
# 第7章：在T4上跑本章的全部实验，结果写到results.txt。书中引用的数字都来自这个文件。
set -e
cd "$(dirname "$0")"
make -s

best() {  # best <命令...>：运行三次，输出GFLOPS或GB/s那一列最大的一行
  for _ in 1 2 3; do "$@"; done |
    awk '{for(i=1;i<=NF;i++) if($i=="GFLOPS"||$i=="GB/s"){print $(i-1) "\t" $0; break}}' |
    sort -g -r | head -1 | cut -f2-
}

# Nsight Compute的命令行工具ncu：不在PATH里时，到CUDA的安装目录里找
NCU=$(command -v ncu || ls /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1 || true)
METRICS=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed

profile() {  # profile <标题> <matmul的参数...>：用ncu采集一次kernel的指标
  echo "### $1"
  if [ -z "$NCU" ]; then echo "没有找到ncu"; return; fi
  "$NCU" --metrics "$METRICS" ./matmul "${@:2}" once 2>&1 | grep -v '^==PROF==' || true
}

{
echo "# GPU：$(nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit --format=csv,noheader)"
echo "# CUDA：$(nvcc --version | grep release)"
echo "# ncu：${NCU:-没有找到}  $([ -n "$NCU" ] && "$NCU" --version | tail -1)"
echo; echo "## 7.2 访存合并的微基准：步长"
for s in 1 2 4 8 16 32; do best ./coalesce stride $s; done
echo; echo "## 7.2 访存合并的微基准：起始偏移"
for o in 0 1 4 8 16 32; do best ./coalesce offset $o; done
echo; echo "## 7.4 正确性"
./matmul check 1000
./matmul check 1024 16
echo; echo "## 7.4 两种映射，不同规模（block 32×32）"
for n in 1024 2048 4096; do for v in naive coalesced; do best ./matmul $v $n 32; done; done
echo; echo "## 7.4 两种映射，不同block大小（n=4096）"
for bs in 8 16 32; do for v in naive coalesced; do best ./matmul $v 4096 $bs; done; done
echo; echo "## 7.1 / 7.4 Nsight Compute（n=2048）"
profile "naive，block 32×32" naive 2048 32
profile "naive，block 8×8" naive 2048 8
profile "coalesced，block 32×32" coalesced 2048 32
echo; echo "## 7.1 Nsight Compute的Speed Of Light小节（naive，n=2048，block 32×32）"
if [ -n "$NCU" ]; then "$NCU" --section SpeedOfLight ./matmul naive 2048 32 once 2>&1 | grep -v '^==PROF==' || true; fi
echo; echo "## 测完时的GPU状态"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active --format=csv
} 2>&1 | tee results.txt
