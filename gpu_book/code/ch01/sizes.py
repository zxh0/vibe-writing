# /// script
# requires-python = ">=3.10"
# ///
"""计算方阵乘法（FP32）的计算量、数据量和二者之比。"""

print(f"{'n':>6} {'FLOP':>22} {'GFLOP':>10} {'数据量(字节)':>16} {'MiB':>8} {'FLOP/字节':>10}")
for n in [256, 512, 1024, 2048, 4096, 8192]:
    flop = 2 * n**3
    nbytes = 3 * n * n * 4
    print(f"{n:>6} {flop:>22,} {flop/1e9:>10.1f} {nbytes:>16,} {nbytes/2**20:>8.0f} {flop/nbytes:>10.1f}")

print()
for d in [1024, 4096, 8192]:
    print(f"d={d}: 线性层每个词元 2d²={2*d*d:,} 次运算，逐元素操作约 d={d} 次，比值 {2*d:,}")
