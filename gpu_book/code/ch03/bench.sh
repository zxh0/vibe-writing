#!/bin/bash
# 第3章：实测本机的峰值算力和内存带宽，结果写到results.txt。书中引用的数字都来自这个文件。
# 每个实验跑三遍，取最快的一次。
set -e
cd "$(dirname "$0")"
make -s

best() {  # best <命令...>：运行三次，输出数值（倒数第二列）最大的一行
  for _ in 1 2 3; do "$@"; done | awk '{print $(NF-1) "\t" $0}' | sort -g -r | head -1 | cut -f2-
}
best_bw() {  # 带宽实验：按GB/s那一列排序
  for _ in 1 2 3; do "$@"; done | awk '{for(i=1;i<=NF;i++) if($i=="GB/s") print $(i-1) "\t" $0}' | sort -g -r | head -1 | cut -f2-
}

{
echo "# 机器：$(sysctl -n machdep.cpu.brand_string)，$(sysctl -n hw.physicalcpu)核$(sysctl -n hw.logicalcpu)线程"
echo "# 内存：$(system_profiler SPMemoryDataType 2>/dev/null | awk -F': ' '/Type:/{t=$2} /Speed:/{s=$2} /Size:/{n++} END{print n "条 " t " " s}')"
echo; echo "## 峰值算力"
best ./peak scalar 1
for t in 1 2 3 6 12; do best ./peak avx2 $t; done
echo; echo "## 内存带宽"
for m in read copy add; do for t in 1 6; do best_bw ./bandwidth $m $t; done; done
} 2>&1 | tee results.txt
