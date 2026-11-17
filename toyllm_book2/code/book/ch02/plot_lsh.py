# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""第二章 · 画LSH的S形曲线

两篇Jaccard相似度为J的文档，签名分成b组、每组r个，成为候选对的概率是

    P = 1 - (1 - J^r)^b

签名长度k = b × r固定为112，换三种分法各画一条，存到书的images/ch02/lsh.png。
中间那条b=14、r=8是FineWeb的参数，也是dedup.py用的。

运行：
    cd code
    uv run book/ch02/plot_lsh.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent.parent.parent
IMAGES = CODE.parent / "images" / "ch02"

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC",
                                   "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def p_candidate(j, b, r):
    return 1 - (1 - j ** r) ** b


def main():
    js = [i / 500 for i in range(501)]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for b, r, style in [(28, 4, ":"), (14, 8, "-"), (7, 16, "--")]:
        threshold = (1 / b) ** (1 / r)
        ax.plot(js, [p_candidate(j, b, r) for j in js], style,
                lw=2 if (b, r) == (14, 8) else 1.2,
                label=f"b={b}，r={r}，阈值≈{threshold:.2f}")
        print(f"b={b:>2} r={r:>2}  阈值 {threshold:.3f}")
    # FineWeb的那条线上，标出相似度0.65的那几对。
    p = p_candidate(0.65, 14, 8)
    print(f"b=14 r= 8  J=0.65时 P={p:.3f}")
    ax.plot([0.65], [p], "o", color="tab:orange", ms=5)
    ax.annotate(f"J=0.65, P≈{p:.2f}", (0.65, p), xytext=(0.1, 0.6),
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9)
    ax.set_xlabel("两篇文档的Jaccard相似度 J")
    ax.set_ylabel("成为候选对的概率 P")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMAGES / "lsh.png", dpi=150)
    print(f"写出 {IMAGES / 'lsh.png'}")


if __name__ == "__main__":
    main()
