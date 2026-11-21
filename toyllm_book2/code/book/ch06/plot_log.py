# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""第六章 · 画训练曲线

读runs/pretrain/log.jsonl（从Colab上下载下来），画本章的三张图，
存到书的images/ch06/下：loss、梯度范数和学习率、吞吐量。

运行：
    cd code
    uv run book/ch06/plot_log.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent.parent.parent
LOG = CODE / "runs" / "pretrain" / "log.jsonl"
IMAGES = CODE.parent / "images" / "ch06"

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


def save(fig, name):
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMAGES / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"写出 {IMAGES / name}.png")


def plot_loss(recs):
    """左边是全程，右边是第500步以后，纵轴放大。"""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, start in zip(axes, [0, 500]):
        part = [r for r in recs if r["step"] >= start]
        xs, ys = series(part, "loss")
        ax.plot(xs, ys, alpha=0.25, color="C0", label="训练loss（每一步）")
        ax.plot(xs, smooth(ys), color="C0", label="训练loss（20步平均）")
        ax.plot(*series(part, "val_loss"), "o-", color="C1", ms=4,
                label="验证loss")
        ax.set_xlabel("步数")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("loss")
    axes[0].set_title("全程")
    axes[1].set_title("第500步以后")
    axes[0].legend()
    save(fig, "loss")


def plot_grad(recs):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(*series(recs, "grad_norm"), linewidth=0.5)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax1.set_yscale("log")
    ax1.set_ylabel("梯度范数（裁剪前）")
    ax1.grid(alpha=0.3)
    ax2.plot(*series(recs, "lr"), color="C2")
    ax2.set_ylabel("学习率")
    ax2.set_xlabel("步数")
    ax2.grid(alpha=0.3)
    save(fig, "grad")


def plot_speed(recs):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(*series(recs, "tok_per_s"))
    ax.set_xlabel("步数")
    ax.set_ylabel("每秒词元数（10步平均）")
    ax.grid(alpha=0.3)
    save(fig, "speed")


def main():
    recs = [json.loads(line) for line in LOG.open()]
    plot_loss(recs)
    plot_grad(recs)
    plot_speed(recs)


if __name__ == "__main__":
    main()
