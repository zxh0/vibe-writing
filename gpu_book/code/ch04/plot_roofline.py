# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""第4章：把i7-9750H和T4的roofline画在同一张图上。

CPU的屋顶来自../ch03/results.txt的实测值；T4的屋顶是specs.py按规格算出的理论值，第5章再实测。
输出：../../images/ch04/roofline_cpu_gpu.png
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
r3 = (HERE.parent / "ch03" / "results.txt").read_text()


def num(pattern):
    for line in r3.splitlines():
        if re.search(pattern, line):
            return float(re.findall(r"([\d.]+)\s*(?:GFLOPS|GB/s)", line)[-1])
    raise ValueError(pattern)


cpu_peak, cpu_bw = num(r"^avx2\s+threads=6 "), num(r"^read\s+threads=6 ")
t4_fp32, t4_tc, t4_bw = 40 * 64 * 2 * 1.59, 40 * 8 * 64 * 2 * 1.59, 320.0

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(8, 5.2), dpi=100)
xs = [v / 100 for v in range(1, 400000)]
roofs = [
    (t4_tc, t4_bw, ":", "#059669", f"T4 Tensor Core（FP16）：{t4_tc / 1000:.0f} TFLOPS，{t4_bw:.0f} GB/s"),
    (t4_fp32, t4_bw, "-", "#059669", f"T4 FP32：{t4_fp32 / 1000:.1f} TFLOPS，{t4_bw:.0f} GB/s"),
    (cpu_peak, cpu_bw, "-", "#374151", f"i7-9750H 6核（实测）：{cpu_peak:.0f} GFLOPS，{cpu_bw:.0f} GB/s"),
]
for peak, bw, style, color, label in roofs:
    ax.plot(xs, [min(peak, bw * v) for v in xs], style, color=color, lw=1.8, label=label)
    ax.plot([peak / bw], [peak], "o", color=color, ms=4)
    ax.annotate(f"拐点 {peak / bw:.1f}" if peak / bw < 100 else f"拐点 {peak / bw:.0f}",
                (peak / bw, peak), xytext=(peak / bw * 1.15, peak * 0.62), fontsize=9, color=color)

ai = 4096 / 6
ax.axvline(ai, color="#dc2626", lw=1, ls="--")
ax.annotate(f"基准问题 n=4096\n理想算术强度 {ai:.0f}", (ai, 30), xytext=(ai * 1.1, 30), fontsize=9, color="#dc2626")

x = [2 ** e for e in range(-2, 13)]
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=10)
ax.set_xlim(1 / 4, 4096)
ax.set_ylim(5, 200000)
ax.set_xticks(x)
ax.set_xticklabels([f"1/{int(1 / v)}" if v < 1 else f"{int(v)}" for v in x], fontsize=8)
ax.set_xlabel("算术强度（FLOP/字节）")
ax.set_ylabel("性能（GFLOPS）")
ax.set_title("CPU与T4的roofline对比", fontsize=11)
ax.grid(True, which="major", color="#e5e7eb", lw=0.8)
ax.legend(loc="upper left", fontsize=9, frameon=False)
fig.tight_layout()

out = HERE.parent.parent / "images" / "ch04" / "roofline_cpu_gpu.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200)
print(f"已写入 {out}，{fig.get_size_inches()[0] * 200:.0f}×{fig.get_size_inches()[1] * 200:.0f} 像素")
