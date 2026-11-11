#!/bin/bash
# 第2章：一次性跑完本章所有实验，结果写到results.txt。书中引用的数字都来自这个文件。
# 笔记本电脑的测量值会随温度、后台程序波动，所以较快的实验各跑三遍，取最快的一次。
set -e
cd "$(dirname "$0")"
make -s
UV="env -u UV_CACHE_DIR -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy uv run -q"

best() {  # best <重复次数> <命令...>：多次运行，输出GFLOPS最高的一行
  local reps=$1; shift
  for _ in $(seq "$reps"); do "$@"; done | sort -k7 -n -r | head -1
}

{
echo "# 机器：$(sysctl -n machdep.cpu.brand_string)，$(sysctl -n hw.physicalcpu)核$(sysctl -n hw.logicalcpu)线程"
echo "# L1d $(($(sysctl -n hw.l1dcachesize)/1024))KB，L2 $(($(sysctl -n hw.l2cachesize)/1024))KB，L3 $(($(sysctl -n hw.l3cachesize)/1024/1024))MB"
echo "# $(clang --version | head -1)"
echo; echo "## 2.1 Python三重循环"
$UV python_loop.py
echo; echo "## 2.2 C语言：-O0与-O2（ijk）"
./matmul_O0 ijk 2048
./matmul ijk 2048
echo; echo "## 2.3 循环顺序（-O2，不向量化）"
best 3 ./matmul ikj 2048
echo; echo "## 2.4 SIMD（-O3 -march=native）"
./matmul_simd ijk 2048
best 3 ./matmul_simd ikj 2048
for n in 256 512; do best 3 ./matmul ikj $n; best 3 ./matmul_simd ikj $n; done
echo; echo "## 2.5 Cache与分块（-O3 -march=native）"
for n in 256 512 1024 2048 4096; do
  if [ $n -le 2048 ]; then best 3 ./matmul_simd ikj $n; else ./matmul_simd ikj $n; fi
  best 3 ./matmul_simd blocked $n 64
done
for bs in 16 32 64 128 256; do best 3 ./matmul_simd blocked 2048 $bs; done
echo; echo "## 2.6 多线程（-O3 -march=native，分块64）"
for t in 1 2 3 6 12; do best 3 ./matmul_simd threads 2048 64 $t; done
echo; echo "## 2.7 NumPy"
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $UV numpy_bench.py
$UV numpy_bench.py
} 2>&1 | tee results.txt
