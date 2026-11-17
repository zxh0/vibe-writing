# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""第五章 · 画loss曲线

读runs/<名字>/log.jsonl，画本章的几张图，存到书的images/ch05/下。
matplotlib只有画图用得到，不进项目的依赖，用PEP 723写在文件开头，
uv run会临时给它建一个环境。

运行：
    cd code
    uv run book/ch05/plot_loss.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent.parent.parent
RUNS = CODE / "runs"
IMAGES = CODE.parent / "images" / "ch05"

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC",
                                   "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load(name):
    return [json.loads(line) for line in (RUNS / name / "log.jsonl").open()]


def series(recs, key):
    pts = [(r["step"], r[key]) for r in recs if key in r]
    return [p[0] for p in pts], [p[1] for p in pts]


def smooth(ys, k=10):
    """前k步的滑动平均。每一步只有8段文本，单步的loss抖得厉害。"""
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


def plot_overfit():
    xs, ys = series(load("overfit"), "loss")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(xs, ys)
    ax.set_yscale("log")
    ax.set_xlabel("步数")
    ax.set_ylabel("loss（对数坐标）")
    ax.set_title("反复训同一个batch")
    ax.grid(alpha=0.3)
    save(fig, "overfit")


def plot_base():
    recs = load("base")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    xs, ys = series(recs, "loss")
    ax1.plot(xs, ys, alpha=0.3, color="C0", label="训练loss（每一步）")
    ax1.plot(xs, smooth(ys), color="C0", label="训练loss（10步平均）")
    xs, ys = series(recs, "val_loss")
    ax1.plot(xs, ys, "o-", color="C1", label="验证loss")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    xs, ys = series(recs, "lr")
    ax2.plot(xs, ys, color="C2")
    ax2.set_ylabel("学习率")
    ax2.set_xlabel("步数")
    ax2.grid(alpha=0.3)
    save(fig, "base")


def plot_compare(name, runs, title, key="loss", ylim=None, xlim=None, logy=False, k=10):
    """几次训练的同一个量画在一张图上。runs是[(记录名, 图例), ...]。

    k是loss的滑动平均取几步，k=1就是不平滑。"""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for run, label in runs:
        xs, ys = series(load(run), key)
        if xlim:
            xs, ys = zip(*[(x, y) for x, y in zip(xs, ys) if x <= xlim[1]])
        ax.plot(xs, smooth(ys, k) if key == "loss" else ys, label=label)
    if logy:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("步数")
    if key == "grad_norm":
        ax.set_ylabel("梯度范数（裁剪前）")
    else:
        ax.set_ylabel("训练loss" if k == 1 else f"训练loss（{k}步平均）")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, name)


def plot_clip():
    """学习率1e-3，裁剪和不裁剪的头100步：上面是梯度范数，下面是loss。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True)
    for run, label in [("base", "裁剪到1.0"), ("noclip", "不裁剪")]:
        recs = [r for r in load(run) if r["step"] <= 100]
        ax1.plot(*series(recs, "grad_norm"), label=label)
        ax2.plot(*series(recs, "loss"), label=label)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax1.set_yscale("log")
    ax1.set_ylabel("梯度范数（裁剪前）")
    ax1.set_title("学习率1e-3，要不要裁剪梯度（头100步）")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("训练loss")
    ax2.set_xlabel("步数")
    ax2.grid(alpha=0.3)
    save(fig, "clip")


def main():
    plot_overfit()
    plot_base()
    plot_compare("optimizer", [("base", "AdamW，学习率1e-3"),
                               ("sgd-1e-3", "SGD，学习率1e-3"),
                               ("sgd-1e-1", "SGD，学习率0.1")],
                 "SGD和AdamW", ylim=(5.5, 9.5))
    plot_compare("lr", [("lr-1e-4", "1e-4"), ("lr-3e-4", "3e-4"), ("base", "1e-3"),
                        ("lr-3e-3", "3e-3"), ("lr-1e-2", "1e-2"), ("lr-3e-2", "3e-2")],
                 "不同的最大学习率", ylim=(5.5, 9.5))
    plot_compare("warmup", [("lr-1e-2", "warmup 30步"), ("lr-1e-2-nowarmup", "不warmup")],
                 "学习率1e-2，要不要warmup（头100步）", xlim=(0, 100), k=1)
    plot_clip()


if __name__ == "__main__":
    main()
