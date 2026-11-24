# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""第九章 · 画GRPO的训练曲线

读runs/grpo/log.jsonl和对照实验的runs/grpo-*/log.jsonl（从Colab上下载
下来），画两张图，存到书的images/ch09/：

    grpo.png  左  贪婪解码的正确率，每50步量一次，1到3位数
              中  训练时采样的奖励，2位数的步和3位数的步分开，20步平均
              右  和参考模型的KL，20步平均
    exp.png   四次200步的对照实验，3位数的正确率

运行：
    cd code
    uv run book/ch09/plot_log.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent.parent.parent
RUNS = CODE / "runs"
IMAGES = CODE.parent / "images" / "ch09"

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC",
                                   "Heiti SC", "Noto Sans CJK SC",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

EXPERIMENTS = [("grpo-base", "默认：8个一组，32道题"),
               ("grpo-g4", "4个一组，64道题"),
               ("grpo-g16", "16个一组，16道题"),
               ("grpo-nokl", "不要KL")]


def read(name):
    return [json.loads(line) for line in (RUNS / name / "log.jsonl").open()]


def smooth(ys, k=20):
    """前k个点的滑动平均。"""
    out, acc = [], []
    for y in ys:
        acc.append(y)
        acc = acc[-k:]
        out.append(sum(acc) / len(acc))
    return out


def accuracy(recs, d):
    pts = [(r["step"], r["acc"][str(d)]) for r in recs if "acc" in r]
    return [p[0] for p in pts], [100 * p[1] for p in pts]


def plot_grpo(recs):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    ax = axes[0]
    for d in (1, 2, 3):
        ax.plot(*accuracy(recs, d), "o-", ms=3, label=f"{d}位")
    ax.set_title("贪婪解码的正确率（%）")
    ax.set_ylim(84, 101)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    for d in (2, 3):
        pts = [(r["step"], r["reward"]) for r in recs
               if r.get("digits") == d]
        ax.plot([p[0] for p in pts], smooth([p[1] for p in pts]),
                label=f"{d}位的步")
    ax.set_title("采样的奖励（20步平均）")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[2]
    pts = [(r["step"], r["kl"]) for r in recs if "kl" in r]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], alpha=0.2, color="C0")
    ax.plot([p[0] for p in pts], smooth([p[1] for p in pts]), color="C0")
    ax.set_title("和参考模型的KL")

    for ax in axes:
        ax.set_xlabel("步数")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGES / "grpo.png", dpi=150)
    print(f"写出 {IMAGES / 'grpo.png'}")


def plot_experiments():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for name, label in EXPERIMENTS:
        ax.plot(*accuracy(read(name), 3), "o-", ms=3, label=label)
    ax.set_title("3位数的贪婪正确率（%）")
    ax.set_xlabel("步数")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(IMAGES / "exp.png", dpi=150)
    print(f"写出 {IMAGES / 'exp.png'}")


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    plot_grpo(read("grpo"))
    plot_experiments()


if __name__ == "__main__":
    main()
