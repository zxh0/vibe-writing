"""第十章 · 比一比：学老师的回答和学原来的回答，训出的学生有什么不同

两件事：

    1. 交叉的验证loss：每个学生在两份验证集上各算一次loss（只算回答）。
       自己那一份低是应该的；看的是每份数据有多“好学”，以及学了一份，
       对另一份的回答有多意外。
    2. 学生自己写：验证集的前--n个问题，每个学生各答一遍，比三样，
       和第八章的answer_len.py一样：
           平均长度    回答有多少个词元，没停下的就是上限
           自己停下    在上限之前写出<|im_end|>的比例
           重复        回答里的四元组有多少是前面出现过的

运行（在CPU上，每个学生几分钟）：
    cd code
    uv run python book/ch10/compare.py runs/distill-orig/model \\
        runs/distill/model runs/distill-5k/model
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from chat_template import EOT, IM_END, ChatTemplate
from config import DATA
from export import load

ROW = 1025


def width(s):
    return sum(2 if ord(c) > 0x2e80 else 1 for c in s)


def pad(s, n):
    return s + " " * (n - width(s))


@torch.no_grad()
def val_loss(model, data_dir, B=16):
    """验证集上的loss，只在回答的词元上平均。"""
    device = next(model.parameters()).device
    rows = np.fromfile(data_dir / "val.bin", dtype=np.uint16).reshape(-1, ROW)
    mask = np.fromfile(data_dir / "val_mask.bin", dtype=np.uint8).reshape(-1, ROW)
    total, count = 0.0, 0
    for i in range(0, len(rows), B):
        x = torch.from_numpy(rows[i:i + B, :-1].astype(np.int64)).to(device)
        y = torch.from_numpy(rows[i:i + B, 1:].astype(np.int64)).to(device)
        m = torch.from_numpy(mask[i:i + B, 1:].astype(bool)).to(device)
        logits = model(x)[0].float()
        nll = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), y.flatten(), reduction="none")
        total += nll[m.flatten()].sum().item()
        count += m.sum().item()
    return total / count


def repeated(ids, n=4):
    seen, rep = set(), 0
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        rep += g in seen
        seen.add(g)
    return rep / max(1, len(ids) - n + 1)


@torch.no_grad()
def answer_stats(model, chat, prompts, max_new, temperature, top_k, seed):
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    lengths, stopped, reps = [], [], []
    for p in prompts:
        ids = torch.tensor([chat.prompt([{"role": "user", "content": p}])],
                           device=device)
        out = model.generate(ids, max_new, temperature=temperature,
                             top_k=top_k, stop=(IM_END, EOT))[0, ids.shape[1]:]
        out = out.tolist()
        stop = out[-1] in (IM_END, EOT)
        body = out[:-1] if stop else out
        lengths.append(len(body))
        stopped.append(stop)
        reps.append(repeated(body))
    n = len(prompts)
    return sum(lengths) / n, sum(stopped) / n, sum(reps) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", type=Path, nargs="+", help="导出的模型目录")
    ap.add_argument("--data", type=Path, default=DATA / "distill")
    ap.add_argument("--n", type=int, default=100, help="自己写回答的问题数")
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    prompts = [json.loads(line) for line in (args.data / "prompts.jsonl").open()]
    val = [p["prompt"] for p in prompts if p["split"] == "val"][:args.n]
    print(f"验证集的前{len(val)}个问题，温度{args.temperature}，"
          f"top-k {args.top_k}，每个最多{args.max_new}个词元\n")
    cols = [("老师的loss", 12), ("原来的loss", 12), ("平均长度", 10),
            ("自己停下", 10), ("重复", 8)]
    print(pad("学生", 16) + "".join(" " * (w - width(s)) + s for s, w in cols))
    for path in args.models:
        model = load(path).to(args.device).eval()
        chat = ChatTemplate(Tokenizer.from_file(str(path / "tokenizer.json")))
        name = path.parent.name if path.name == "model" else path.name
        lt = val_loss(model, args.data / "teacher")
        lo = val_loss(model, args.data / "orig")
        length, stop, rep = answer_stats(model, chat, val, args.max_new,
                                         args.temperature, args.top_k,
                                         args.seed)
        print(f"{pad(name, 16)}{lt:>12.3f}{lo:>12.3f}{length:>10.0f}"
              f"{stop:>10.0%}{rep:>8.0%}", flush=True)


if __name__ == "__main__":
    main()
