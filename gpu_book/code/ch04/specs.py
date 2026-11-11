"""第4章：由规格参数推算T4的峰值算力、显存带宽和拐点，并与第3章实测的CPU对比。

T4的参数来自NVIDIA的T4数据手册和Turing架构白皮书；CPU的数字是第3章的实测值。
只用标准库，直接运行：python3 specs.py
"""

# ---- NVIDIA T4（Turing架构，TU104核心的裁剪版）----
SM = 40                 # 流式多处理器个数
FP32_PER_SM = 64        # 每个SM的FP32单元（即“CUDA Core”）个数
TC_PER_SM = 8           # 每个SM的Tensor Core个数
TC_FMA_PER_CLK = 64     # 每个Tensor Core每周期完成的FP16乘加次数
BOOST_GHZ = 1.59        # 最高主频（GHz）
BUS_BITS = 256          # 显存位宽
GDDR6_GTPS = 10         # 每根数据线每秒传输次数（10 GT/s）
PCIE_GBPS = 16 * 8 * 128 / 130 / 8   # PCIe 3.0 x16：16条通道，每条8 GT/s，128b/130b编码

# ---- 第3章实测的i7-9750H（6核）----
CPU_PEAK = 605.3        # GFLOPS
CPU_BW = 22.0           # GB/s

fp32_per_clk = SM * FP32_PER_SM * 2
fp32_peak = fp32_per_clk * BOOST_GHZ               # GFLOPS
tc_peak = SM * TC_PER_SM * TC_FMA_PER_CLK * 2 * BOOST_GHZ
bw = BUS_BITS / 8 * GDDR6_GTPS                     # GB/s

print(f"T4 FP32：{SM}×{FP32_PER_SM}×2 = 每周期{fp32_per_clk}次，峰值 {fp32_peak:.0f} GFLOPS")
print(f"T4 Tensor Core（FP16）：峰值 {tc_peak:.0f} GFLOPS")
print(f"T4 显存带宽：{BUS_BITS // 8}字节 × {GDDR6_GTPS} GT/s = {bw:.0f} GB/s")
print(f"T4 拐点：FP32 {fp32_peak / bw:.1f}，Tensor Core {tc_peak / bw:.0f} FLOP/字节")
print(f"CPU 拐点：{CPU_PEAK / CPU_BW:.1f} FLOP/字节；每周期 6×32 = 192 次")
print(f"倍数：算力 {fp32_peak / CPU_PEAK:.1f}×，带宽 {bw / CPU_BW:.1f}×，每周期运算次数 {fp32_per_clk / 192:.1f}×")
print(f"寄存器：每SM 256 KiB，共 {SM * 256 / 1024:.0f} MiB；常驻线程 {SM}×1024 = {SM * 1024}")
print(f"PCIe 3.0 x16 理论带宽：{PCIE_GBPS:.2f} GB/s")
for n in (1024, 4096):
    nbytes = 3 * n * n * 4
    flops = 2 * n ** 3
    print(f"n={n}：三个矩阵共 {nbytes / 2**20:.0f} MiB，传输 {nbytes / PCIE_GBPS / 1e6:.2f} ms；"
          f"T4 FP32峰值计算 {flops / fp32_peak / 1e6:.2f} ms；CPU 6核峰值 {flops / CPU_PEAK / 1e6:.1f} ms")
