# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""第七章 · 画SFT的loss曲线

读runs/sft/log.jsonl（从Colab上下载下来），画训练loss和验证loss，
存到书的images/ch07/loss.png。左边是全程，右边是第200步以后，纵轴放大。

运行：
    cd code
    uv run book/ch07/plot_log.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent.parent.parent
LOG = CODE / "runs" / "sft" / "log.jsonl"
IMAGES = CODE.parent / "images" / "ch07"

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC",
                                   "Heiti SC", "Noto Sans CJK SC",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def series(recs, key):
    pts = [(r["step"], r[key]) for r in recs if key in r]
    return [p[0] for p in pts], [p[1] for p in pts]


def smooth(ys, k=20):
    """前k步的滑动平均。"""
    out, acc = [], []
    for y in ys:
        acc.append(y)
        acc = acc[-k:]
        out.append(sum(acc) / len(acc))
    return out


def main():
    recs = [json.loads(line) for line in LOG.open()]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, start in zip(axes, [0, 200]):
        part = [r for r in recs if r["step"] >= start]
        xs, ys = series(part, "loss")
        ax.plot(xs, ys, alpha=0.25, color="C0", label="训练loss（每一步）")
        ax.plot(xs, smooth(ys), color="C0", label="训练loss（20步平均）")
        ax.plot(*series(part, "val_loss"), "o-", color="C1", ms=3,
                label="验证loss")
        ax.set_xlabel("步数")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("loss（只算回答）")
    axes[0].set_title("全程")
    axes[1].set_title("第200步以后")
    axes[0].legend()
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMAGES / "loss.png", dpi=150)
    print(f"写出 {IMAGES / 'loss.png'}")


if __name__ == "__main__":
    main()
