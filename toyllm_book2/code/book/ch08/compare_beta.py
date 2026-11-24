"""第八章 · 对照实验：不同的beta

读几次短跑的日志（runs/exp-beta*/log.jsonl，在Colab上跑完下载下来），
列出最后一次验证的结果：准确率、差距，以及好坏回答的对数概率比起点
变了多少。差距是beta乘出来的，除以beta，才能看出模型离参考模型走了多远。
验证loss也和beta有关，不同的beta之间不好直接比，所以不列。

运行：
    cd code
    uv run python book/ch08/compare_beta.py
"""

import json

from config import CODE

BETAS = [0.05, 0.1, 0.5]


def main():
    print("beta    准确率    差距   差距/beta   好回答变化   坏回答变化")
    for beta in BETAS:
        log = CODE / "runs" / f"exp-beta{beta}" / "log.jsonl"
        recs = [json.loads(line) for line in log.open()]
        evals = [r for r in recs if "val_acc" in r]
        first, last = evals[0], evals[-1]
        dc = last["val_logp_chosen"] - first["val_logp_chosen"]
        dr = last["val_logp_rejected"] - first["val_logp_rejected"]
        print(f"{beta:<6}{last['val_acc']:>8.1%}{last['val_margin']:>8.3f}"
              f"{last['val_margin'] / beta:>12.1f}{dc:>13.1f}{dr:>13.1f}")


if __name__ == "__main__":
    main()
