# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""第六章 · 画训练曲线

读runs/pretrain/log.jsonl（从Colab上下载下来），画本章的三张图，
存到书的images/ch06/下：loss、梯度范数和学习率、吞吐量。
对照实验的日志（runs/exp-*/log.jsonl）也在的话，再画两张：
fp16溢出时放大倍数和loss怎么变，以及权重共享和不共享的loss曲线。

运行：
    cd code
    uv run book/ch06/plot_log.py
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent.parent.parent
RUNS = CODE / "runs"
LOG = RUNS / "pretrain" / "log.jsonl"
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


def plot_overflow(recs):
    """放大倍数从2的40次方开始：上面是放大倍数（以2为底的指数），
    跳过的步标成红点；下面是loss。"""
    recs = [r for r in recs if "loss" in r]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    xs, ys = series(recs, "scale")
    ax1.plot(xs, [math.log2(y) for y in ys], ".-", color="C0")
    skip = [r for r in recs if r.get("skipped")]
    ax1.plot([r["step"] for r in skip], [math.log2(r["scale"]) for r in skip],
             "o", color="C3", label="梯度溢出，跳过")
    ax1.axhline(16, color="gray", linestyle="--", linewidth=1)
    ax1.set_ylabel("放大倍数（2的几次方）")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(*series(recs, "loss"), color="C0")
    ax2.set_ylabel("训练loss")
    ax2.set_xlabel("步数")
    ax2.grid(alpha=0.3)
    save(fig, "overflow")


def plot_untie(base, untie):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for recs, label, color in [(base, "共享", "C0"), (untie, "不共享", "C1")]:
        xs, ys = series(recs, "loss")
        ax.plot(xs, smooth(ys), color=color, label=f"{label}，训练loss（20步平均）")
        ax.plot(*series(recs, "val_loss"), "o", color=color, ms=5,
                label=f"{label}，验证loss")
    ax.set_ylim(4, 6.5)
    ax.set_xlabel("步数")
    ax.set_ylabel("loss")
    ax.set_title("输出层和词嵌入：共享和不共享")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "untie")


def read(path):
    return [json.loads(line) for line in path.open()]


def main():
    recs = read(LOG)
    plot_loss(recs)
    plot_grad(recs)
    plot_speed(recs)
    if (RUNS / "exp-overflow" / "log.jsonl").exists():
        plot_overflow(read(RUNS / "exp-overflow" / "log.jsonl"))
    if (RUNS / "exp-untie" / "log.jsonl").exists():
        plot_untie(read(RUNS / "exp-base" / "log.jsonl"),
                   read(RUNS / "exp-untie" / "log.jsonl"))


if __name__ == "__main__":
    main()
