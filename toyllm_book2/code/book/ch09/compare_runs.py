"""第九章 · 对照实验：组的大小、有没有KL

读四次200步短跑的日志（runs/grpo-*/log.jsonl，在Colab上跑完下载下来），
每次一行：

    2位、3位    最后一次量的贪婪正确率
    有信号      一组回答里有对有错的组，在所有组里占多少（全程平均）
    KL          和参考模型的KL，最后50步的平均，以及全程最大的一步

运行：
    cd code
    uv run python book/ch09/compare_runs.py
"""

import json

from config import CODE

RUNS = [("grpo-base", "8 × 32"), ("grpo-g4", "4 × 64"),
        ("grpo-g16", "16 × 16"), ("grpo-nokl", "8 × 32，不要KL")]


def width(s):
    return sum(2 if ord(c) > 0x2e80 else 1 for c in s)


def main():
    print("一组几个回答 × 几道题，每步都是256个回答，各训200步\n")
    head = ["实验", "2位", "3位", "有信号", "KL平均", "KL最大"]
    cols = [16, 8, 8, 8, 9, 9]
    print("".join(h + " " * (c - width(h)) if i == 0
                  else " " * (c - width(h)) + h
                  for i, (h, c) in enumerate(zip(head, cols))))
    for name, label in RUNS:
        recs = [json.loads(line)
                for line in (CODE / "runs" / name / "log.jsonl").open()]
        acc = [r["acc"] for r in recs if "acc" in r][-1]
        steps = [r for r in recs if "kl" in r]
        mixed = sum(r["mixed"] for r in steps) / len(steps)
        last = steps[-50:]
        kl = sum(r["kl"] for r in last) / len(last)
        top = max(r["kl"] for r in steps)
        print(label + " " * (cols[0] - width(label))
              + f"{acc['2']:>8.1%}{acc['3']:>8.1%}{mixed:>8.0%}"
              f"{kl:>9.4f}{top:>9.4f}")


if __name__ == "__main__":
    main()
