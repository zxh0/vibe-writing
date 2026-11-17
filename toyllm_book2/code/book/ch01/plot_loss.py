# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""第一章 · 画字符级小模型的loss曲线

读char_gpt.py写的runs/ch01/log.jsonl，画到书的images/ch01/loss.png。
matplotlib只有画图用得到，不进项目的依赖，用PEP 723写在文件开头，
uv run会临时给它建一个环境。

运行（先跑一遍char_gpt.py）：
    cd code
    uv run book/ch01/plot_loss.py
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent.parent.parent
LOG = CODE / "runs" / "ch01" / "log.jsonl"
IMAGES = CODE.parent / "images" / "ch01"

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC",
                                   "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def smooth(ys, k=20):
    """前k步的滑动平均。每一步只有32段文本，单步的loss抖得厉害。"""
    out, acc = [], []
    for y in ys:
        acc.append(y)
        acc = acc[-k:]
        out.append(sum(acc) / len(acc))
    return out


def main():
    recs = [json.loads(line) for line in LOG.open()]
    steps = [r["step"] for r in recs if "loss" in r]
    losses = [r["loss"] for r in recs if "loss" in r]
    evals = [r for r in recs if "val_loss" in r]

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(steps, losses, color="tab:blue", alpha=0.25, lw=0.6)
    ax.plot(steps, smooth(losses), color="tab:blue", lw=1.5,
            label="训练loss（每一步，前20步平均）")
    ax.plot([r["step"] for r in evals], [r["val_loss"] for r in evals],
            "o", color="tab:orange", ms=5, label="验证loss（每300步）")
    ax.axhline(math.log(65), color="gray", ls="--", lw=0.8)
    ax.text(1500, math.log(65) + 0.05, "乱猜：ln 65", ha="right", va="bottom",
            color="gray", fontsize=9)
    ax.set_xlabel("步数")
    ax.set_ylabel("loss")
    ax.set_xlim(0, 1500)
    ax.set_ylim(1.2, 4.6)
    ax.grid(alpha=0.3)
    ax.legend()

    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMAGES / "loss.png", dpi=150)
    print(f"写出 {IMAGES / 'loss.png'}")


if __name__ == "__main__":
    main()
