# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""第3章：画出本机CPU的roofline，并把第2章的各个实现放上去。

屋顶（峰值算力、内存带宽）来自本目录的results.txt，各实现的速度来自../ch02/results.txt，
所以图和正文引用的数字出自同一次测量。各实现的算术强度是正文中估算的值。
输出：../../images/ch03/roofline.png
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
r3 = (HERE / "results.txt").read_text()
r2 = (HERE.parent / "ch02" / "results.txt").read_text()


def num(text, pattern):
    """在text中找到第一行匹配pattern的内容，返回其中最后一个“数字 + 单位”前的数字。"""
    for line in text.splitlines():
        if re.search(pattern, line):
            return float(re.findall(r"([\d.]+)\s*(?:GFLOPS|GB/s)", line)[-1])
    raise ValueError(pattern)


peak1 = num(r3, r"^avx2\s+threads=1 ")
peak6 = num(r3, r"^avx2\s+threads=6 ")
bw1 = num(r3, r"^read\s+threads=1 ")
bw6 = num(r3, r"^read\s+threads=6 ")
vadd = float(re.search(r"^add\s+threads=6 .*?([\d.]+) GFLOPS", r3, re.M).group(1))

numpy_lines = r2.split("## 2.7 NumPy")[1]
single, multi = numpy_lines.split("线程数：默认")
points = [
    # 名称, 算术强度, GFLOPS, 是否多线程
    ("向量加法", 1 / 12, vadd, True),
    # 第2章中ijk出现了三次（-O0、-O2、SIMD），这里取-O2那一次
    ("ijk", 2 / 64, float(re.findall(r"^ijk\s+n=2048 .*?([\d.]+) GFLOPS", r2, re.M)[1]), False),
    ("ikj（SIMD）", 0.5, float(re.findall(r"^ikj\s+n=2048 .*?([\d.]+) GFLOPS", r2, re.M)[-1]), False),
    ("分块", 64 / 6, float(re.search(r"^blocked\s+n=2048\s+bs=64 .*?([\d.]+) GFLOPS", r2, re.M).group(1)), False),
    ("分块，6线程", 64 / 6, float(re.search(r"^threads\s+n=2048\s+bs=64\s+threads=6 .*?([\d.]+) GFLOPS", r2, re.M).group(1)), True),
    ("NumPy，单线程", 2048 / 6, float(re.search(r"n=2048\s+[\d.]+ 秒\s+([\d.]+) GFLOPS", single).group(1)), False),
    ("NumPy，多线程", 2048 / 6, float(re.search(r"n=2048\s+[\d.]+ 秒\s+([\d.]+) GFLOPS", multi).group(1)), True),
]

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(8, 5.2), dpi=100)
x = [2 ** e for e in range(-6, 10)]
xs = [v / 100 for v in range(1, 100000)]
for peak, bw, style, label in [(peak6, bw6, "-", f"6核：{peak6:.0f} GFLOPS，{bw6:.0f} GB/s"),
                               (peak1, bw1, "--", f"单核：{peak1:.0f} GFLOPS，{bw1:.1f} GB/s")]:
    ax.plot(xs, [min(peak, bw * v) for v in xs], style, color="#374151", lw=1.6, label=label)
    ax.plot([peak / bw], [peak], "o", color="#374151", ms=4)

for name, ai, gf, mt in points:
    color = "#2563eb" if mt else "#dc2626"
    ax.plot(ai, gf, "s" if mt else "o", color=color, ms=7, zorder=3)
    dy = 1.25 if name not in ("分块", "ikj（SIMD）") else 0.72
    ax.annotate(f"{name}\n{gf:g}", (ai, gf), xytext=(ai * 1.12, gf * dy), fontsize=9, color=color,
                va="center")

ax.set_xscale("log", base=2)
ax.set_yscale("log", base=10)
ax.set_xlim(1 / 64, 1024)
ax.set_ylim(0.1, 2000)
ax.set_xticks(x)
ax.set_xticklabels([f"1/{int(1 / v)}" if v < 1 else f"{int(v)}" for v in x], fontsize=8)
ax.set_xlabel("算术强度（FLOP/字节）")
ax.set_ylabel("性能（GFLOPS）")
ax.set_title("i7-9750H的roofline（红色圆点为单线程，蓝色方块为多线程）", fontsize=11)
ax.grid(True, which="major", color="#e5e7eb", lw=0.8)
ax.legend(loc="lower right", fontsize=9, frameon=False)
fig.tight_layout()

out = HERE.parent.parent / "images" / "ch03" / "roofline.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200)
print(f"已写入 {out}，{fig.get_size_inches()[0] * 200:.0f}×{fig.get_size_inches()[1] * 200:.0f} 像素")
for name, ai, gf, mt in points:
    peak, bw = (peak6, bw6) if mt else (peak1, bw1)
    roof = min(peak, bw * ai)
    print(f"{name:<12} 算术强度 {ai:8.3f}  实测 {gf:7.2f}  屋顶 {roof:7.1f}  达到屋顶的 {gf / roof:6.1%}")
