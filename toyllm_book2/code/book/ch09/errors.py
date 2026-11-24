"""第九章 · 错在哪：3位数做错的题，和4位数的题它写了什么

    1. 3位数：用正确率那200道固定的题，贪婪解码，把做错的题按进位的
       次数分开数。三位加三位，个位、十位、百位各可能进一次位。
    2. 4位数：挑几道题，看它写出了什么。

运行（在CPU上，一两分钟）：
    cd code
    uv run python book/ch09/errors.py runs/cold/model runs/grpo/model
"""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from chat_template import ChatTemplate
from export import load
from rollout import fixed_problems, rollout


def carries(a, b):
    """a + b一共进了几次位。"""
    n, c = 0, 0
    while a or b:
        c = (a % 10 + b % 10 + c) // 10
        n += c
        a, b = a // 10, b // 10
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", type=Path, nargs="+", help="导出的模型目录")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    tests = fixed_problems([1, 2, 3, 4, 5], 200)   # 和正确率用的题一样
    three, four = tests[3], tests[4][:4]
    total = [0] * 4
    for a, b in three:
        total[carries(a, b)] += 1
    print("3位数200道题，按进位次数分：", "  ".join(
        f"{c}次 {n}道" for c, n in enumerate(total)))
    for path in args.models:
        model = load(path).eval()
        chat = ChatTemplate(Tokenizer.from_file(str(path / "tokenizer.json")))
        name = path.parent.name if path.name == "model" else path.name
        oks, texts = [], []
        for i in range(0, len(three), 64):          # 和accuracy一样
            r = rollout(model, chat, three[i:i + 64], 1, 24, temperature=0)
            oks += r["rewards"].tolist()
            texts += r["texts"]
        wrong = [0] * 4
        for (a, b), ok in zip(three, oks):
            if not ok:
                wrong[carries(a, b)] += 1
        print(f"\n{name}  3位数做错{sum(wrong)}道，按进位次数：", "  ".join(
            f"{c}次 {w}/{n}" for c, (w, n) in enumerate(zip(wrong, total))))
        for (a, b), ok, text in zip(three, oks, texts):
            if not ok:
                print(f"  {text:<22}正确是{a + b}")
                break
        r = rollout(model, chat, four, 1, 24, temperature=0)
        print("  4位数：")
        for (a, b), text in zip(four, r["texts"]):
            print(f"  {text:<22}正确是{a + b}")


if __name__ == "__main__":
    main()
