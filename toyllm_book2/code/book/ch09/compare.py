"""第九章 · 比一比：冷启动之后和GRPO之后，每种位数做对多少

两个数：

    贪婪    每题写一个回答，每步取概率最大的词元，做对的比例
    pass@k  每题采样k个回答（温度1.0），至少有一个做对的题的比例

强化学习的奖励只看对错，它能做的，是把“采样k次偶尔做对”的题，
变成“一次就做对”。所以贪婪正确率涨得多，pass@k涨得少，甚至可能跌：
模型越来越确定，采样出的回答就越来越像，试k次也只是同一个答案。

运行（在CPU上，每个模型几分钟）：
    cd code
    uv run python book/ch09/compare.py runs/cold/model runs/grpo/model
"""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from chat_template import ChatTemplate
from export import load
from rollout import accuracy, fixed_problems, rollout


def pass_at_k(model, chat, problems, k, max_new=24, batch=16):
    """每题采样k个回答，至少一个对的题占多少。"""
    out = {}
    for d, probs in problems.items():
        hit = 0
        for i in range(0, len(probs), batch):
            r = rollout(model, chat, probs[i:i + batch], k, max_new, 1.0)
            hit += (r["rewards"].view(-1, k).max(-1).values > 0).sum().item()
        out[d] = hit / len(probs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", type=Path, nargs="+", help="导出的模型目录")
    ap.add_argument("--digits", default="1,2,3,4,5")
    ap.add_argument("--n", type=int, default=200, help="每种位数几道题")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    digits = [int(d) for d in args.digits.split(",")]
    tests = fixed_problems(digits, args.n)
    print(f"每种位数{args.n}道题，两个加数位数相同；pass@{args.k}采样温度1.0\n")
    head = "".join(" " * 5 + f"{d}位" for d in digits)
    print(f"模型{' ' * 18}{head}")
    for path in args.models:
        model = load(path).to(args.device).eval()
        chat = ChatTemplate(Tokenizer.from_file(str(path / "tokenizer.json")))
        name = path.parent.name if path.name == "model" else path.name
        torch.manual_seed(args.seed)
        greedy = accuracy(model, chat, tests)
        sampled = pass_at_k(model, chat, tests, args.k)
        for label, acc in [("贪婪", greedy), (f"pass@{args.k}", sampled)]:
            cells = "".join(f"{acc[d]:>8.1%}" for d in digits)
            pad = 8 - sum(2 if ord(c) > 0x2e80 else 1 for c in label)
            print(f"{name:<14}{label}{' ' * pad}{cells}", flush=True)
            name = ""


if __name__ == "__main__":
    main()
