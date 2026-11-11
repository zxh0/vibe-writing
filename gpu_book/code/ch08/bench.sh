#!/bin/bash
# 第8章：在T4上跑本章的全部实验，结果写到results.txt。书中引用的数字都来自这个文件。
set -e
cd "$(dirname "$0")"
make -s

best() {  # best <命令...>：运行三次，输出GFLOPS那一列最大的一行
  for _ in 1 2 3; do "$@"; done |
    awk '{for(i=1;i<=NF;i++) if($i=="GFLOPS"){print $(i-1) "\t" $0; break}}' |
    sort -g -r | head -1 | cut -f2-
}

# Nsight Compute的命令行工具ncu：不在PATH里时，到CUDA的安装目录里找
NCU=$(command -v ncu || ls /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1 || true)
# 第一组与第7章相同：用时、显存读写量、全局内存的请求数和sector数、L2的读取量、两个吞吐百分比
METRICS1=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed
# 第二组是本章新看的：执行的指令总数、全局内存和共享内存的读指令数、共享内存的bank冲突、L1吞吐、实际occupancy
METRICS2=smsp__inst_executed.sum,smsp__inst_executed_op_global_ld.sum,smsp__inst_executed_op_shared_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active

profile() {  # profile <标题> <matmul的参数...>：用ncu采集一次kernel的两组指标
  echo "### $1"
  if [ -z "$NCU" ]; then echo "没有找到ncu"; return; fi
  for m in "$METRICS1" "$METRICS2"; do
    "$NCU" --metrics "$m" ./matmul "${@:2}" once 2>&1 | grep -v '^==PROF==' || true
  done
}

# 一边运行命令，一边每200毫秒记录一次GPU的主频、功耗、温度和降频原因，最后输出统计
watch_gpu() {  # watch_gpu <标题> <命令...>
  echo "### $1"
  nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_slowdown \
    --format=csv,noheader,nounits -lms 200 > gpu.log &
  local pid=$!
  sleep 1
  "${@:2}"
  kill $pid 2>/dev/null || true
  wait $pid 2>/dev/null || true
  awk -F', *' '
    { n++; c=$1+0; p=$2+0; csum+=c; psum+=p; if(n==1||c<cmin)cmin=c; if(c>cmax)cmax=c; if(p>pmax)pmax=p; if($3+0>tmax)tmax=$3+0
      if($4=="Active")pc++; if($5=="Active")th++; if($6=="Active")hw++ }
    END { if(n==0){print "    没有采到数据"; exit}
      printf "    采样%d次  SM主频 平均%.0f MHz（最低%d，最高%d）  功耗 平均%.1f W（最高%.1f）  最高温度%d℃\n", n, csum/n, cmin, cmax, psum/n, pmax, tmax
      printf "    降频原因出现的比例：功耗上限 %.0f%%  温度 %.0f%%  硬件 %.0f%%\n", 100*pc/n, 100*th/n, 100*hw/n }' gpu.log
}

{
echo "# GPU：$(nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit --format=csv,noheader)"
echo "# CUDA：$(nvcc --version | grep release)"
echo "# ncu：${NCU:-没有找到}  $([ -n "$NCU" ] && "$NCU" --version | tail -1)"
echo; echo "## 8.3 / 8.5 正确性（n=1024是32的整数倍，n=1000和n=1023不是）"
for n in 1024 1000 1023; do ./matmul check $n; done
echo; echo "## 8.4 共享内存分块与访存合并，不同规模（block 32×32）"
for n in 1024 2048 4096; do for v in coalesced smem; do best ./matmul $v $n 32; done; done
echo; echo "## 8.6 不同分块大小（n=4096）"
for bs in 8 16 32; do best ./matmul smem 4096 $bs; done
echo; echo "## 8.5 尺寸不是分块大小的整数倍（分块32）"
for n in 4064 4095 4096 4097; do best ./matmul smem $n 32; done
echo; echo "## 每个kernel用到的寄存器和共享内存"
cuobjdump --dump-resource-usage matmul 2>/dev/null | grep -E 'Function|REG' | sed -n '1,40p' || echo "没有cuobjdump"
echo; echo "## 8.1 / 8.6 Nsight Compute（n=2048）"
profile "coalesced，block 32×32" coalesced 2048 32
profile "smem，分块8" smem 2048 8
profile "smem，分块16" smem 2048 16
profile "smem，分块32" smem 2048 32
echo; echo "## 8.1 Nsight Compute的Speed Of Light小节（coalesced，n=2048，block 32×32）"
if [ -n "$NCU" ]; then "$NCU" --section SpeedOfLight ./matmul coalesced 2048 32 once 2>&1 | grep -v '^==PROF==' || true; fi
echo; echo "## 8.6 Nsight Compute的Speed Of Light小节（smem，n=2048，分块32）"
if [ -n "$NCU" ]; then "$NCU" --section SpeedOfLight ./matmul smem 2048 32 once 2>&1 | grep -v '^==PROF==' || true; fi
echo; echo "## 第11章的准备：cuBLAS在不同规模下的速度"
for n in 1024 2048 3072 4096; do best ./matmul cublas $n; done
echo; echo "## 第11章的准备：连续运行10秒，看主频和功耗"
watch_gpu "cuBLAS，n=2048" ./matmul cublas 2048 32 hold 10
watch_gpu "cuBLAS，n=4096" ./matmul cublas 4096 32 hold 10
watch_gpu "smem，n=4096，分块32" ./matmul smem 4096 32 hold 10
echo; echo "## 测完时的GPU状态"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active --format=csv
} 2>&1 | tee results.txt
