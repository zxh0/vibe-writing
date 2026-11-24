# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""第八章 · 画DPO的训练曲线

读runs/dpo/log.jsonl（从Colab上下载下来），画三张小图，存到书的
images/ch08/dpo.png：

    左    训练loss和验证loss，横线是起点的log 2
    中    准确率：隐式奖励好回答高于坏回答的比例
    右    验证集上好回答、坏回答的隐式奖励，以及两者的差距

运行：
    cd code
    uv run book/ch08/plot_log.py
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent.parent.parent
LOG = CODE / "runs" / "dpo" / "log.jsonl"
IMAGES = CODE.parent / "images" / "ch08"

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
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))

    ax = axes[0]
    xs, ys = series(recs, "loss")
    ax.plot(xs, ys, alpha=0.2, color="C0")
    ax.plot(xs, smooth(ys), color="C0", label="训练（20步平均）")
    ax.plot(*series(recs, "val_loss"), "o-", color="C1", ms=3, label="验证")
    ax.axhline(math.log(2), color="gray", ls="--", lw=1)
    ax.set_title("loss")

    ax = axes[1]
    xs, ys = series(recs, "acc")
    ax.plot(xs, smooth(ys), color="C0", label="训练（20步平均）")
    ax.plot(*series(recs, "val_acc"), "o-", color="C1", ms=3, label="验证")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_title("准确率")

    ax = axes[2]
    ax.plot(*series(recs, "val_chosen"), "o-", color="C2", ms=3,
            label="好回答")
    ax.plot(*series(recs, "val_rejected"), "o-", color="C3", ms=3,
            label="坏回答")
    ax.plot(*series(recs, "val_margin"), "o-", color="C4", ms=3,
            label="差距")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_title("验证集上的隐式奖励")
    ax.set_ylim(-0.72, 0.68)
    ax.legend(fontsize=8, loc="upper center", ncol=3)

    for ax in axes:
        ax.set_xlabel("步数")
        ax.grid(alpha=0.3)
    for ax in axes[:2]:
        ax.legend(fontsize=8)
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMAGES / "dpo.png", dpi=150)
    print(f"写出 {IMAGES / 'dpo.png'}")


if __name__ == "__main__":
    main()
