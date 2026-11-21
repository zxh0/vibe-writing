"""第六章 · 评估：训完的模型，和SmolLM2-135M比一比

三件事：

    1. 和transformers核对：用transformers加载导出的目录，同一句话的logits
       和我们自己的model.py比，确认导出的是同一个模型。
    2. 每字节比特数：验证集里取前几十篇文档，两个模型各算一遍loss，
       按字节换算。两个分词器不一样，按词元算的loss不能直接比。
    3. 探针问题：写一句话的开头，答案就是下一个词，两个模型都用贪婪解码。

transformers只在这个检查脚本里用，训练的代码不用它。

运行（先把Colab上的runs/pretrain/model/下载到code/runs/pretrain/model/）：
    cd code
    uv run python book/ch06/eval.py
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

from config import CODE, DATA, MODELS
from export import load

PROBES = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "The largest planet in the solar system is",
    "The chemical symbol for gold is",
    "The opposite of hot is",
    "Monday, Tuesday, Wednesday,",
    "1, 2, 3, 4, 5,",
    "2 + 3 =",
    "The first president of the United States was",
    "A cat is an animal that",
]


def width(text):
    """显示宽度：汉字占两列。"""
    return sum(2 if ord(c) > 0x2e80 else 1 for c in text)


def ljust(text, n):
    return text + " " * (n - width(text))


def rjust(text, n):
    return " " * (n - width(text)) + text


def val_docs(tok, n, max_len):
    """验证集按<|endoftext|>切成文档，取前n篇不超过max_len个词元的，
    还原成文字。两个模型拿同样的文字，各用各的分词器。"""
    val = np.memmap(DATA / "tokens" / "val.bin", dtype=np.uint16, mode="r")
    docs, start = [], 0
    for end in np.flatnonzero(val == 0):
        if 0 < end - start <= max_len:
            docs.append(tok.decode(val[start:end].tolist()))
            if len(docs) == n:
                break
        start = end + 1
    return docs


@torch.no_grad()
def total_loss(logits_fn, encode, docs, bos):
    """每篇文档前面加一个起始符，从第一个词元起就要猜。返回loss的总和
    （自然对数）和词元数。"""
    total, count = 0.0, 0
    for text in docs:
        ids = torch.tensor([[bos] + encode(text)])
        logp = torch.log_softmax(logits_fn(ids[:, :-1])[0].float(), dim=-1)
        target = ids[0, 1:]
        total -= logp[torch.arange(len(target)), target].sum().item()
        count += len(target)
    return total, count


@torch.no_grad()
def greedy(logits_fn, encode, decode, prompt, n):
    """贪婪解码n个词元，只留第一行。"""
    ids = torch.tensor([encode(prompt)])
    start = ids.shape[1]
    for _ in range(n):
        nxt = logits_fn(ids)[0, -1].argmax()
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return decode(ids[0, start:].tolist()).strip().split("\n")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=CODE / "runs/pretrain/model")
    ap.add_argument("--ref", type=Path, default=MODELS / "SmolLM2-135M")
    ap.add_argument("--docs", type=int, default=100, help="取验证集的几篇文档")
    args = ap.parse_args()
    torch.manual_seed(0)

    ours = load(args.model).eval()
    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    ref = AutoModelForCausalLM.from_pretrained(args.ref, dtype=torch.float32).eval()
    ref_tok = Tokenizer.from_file(str(args.ref / "tokenizer.json"))
    models = {
        "我们的模型": (lambda x: ours(x)[0], tok),
        "SmolLM2-135M": (lambda x: ref(x).logits, ref_tok),
    }

    # 1. 用transformers加载我们导出的目录，算同一句话
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    ids = torch.tensor([tok.encode("The capital of France is").ids])
    with torch.no_grad():
        a, b = ours(ids)[0], hf(ids).logits
    err = ((a - b).abs().max() / b.abs().max()).item()
    print(f"和transformers核对：logits最大相对误差 {err:.1e}\n")

    # 2. 每字节比特数。文档长度以我们的分词器为准，不超过训练时的1024
    docs = val_docs(tok, args.docs, 1023)
    n_bytes = sum(len(d.encode("utf-8")) for d in docs)
    print(f"验证集前{len(docs)}篇文档，共{n_bytes:,}个字节\n")
    cols = [("模型", 14), ("词元数", 8), ("每词元loss", 12), ("困惑度", 8),
            ("每词元字节", 12), ("每字节比特", 12)]
    print(ljust(cols[0][0], cols[0][1])
          + "".join(rjust(name, w) for name, w in cols[1:]))
    for name, (fn, t) in models.items():
        loss, n = total_loss(fn, lambda s: t.encode(s).ids, docs, bos=0)
        print(ljust(name, 14) + f"{n:>8,}{loss / n:>12.3f}{math.exp(loss / n):>8.1f}"
              f"{n_bytes / n:>12.2f}{loss / n_bytes / math.log(2):>12.3f}")

    # 3. 探针
    print("\n探针（贪婪解码12个词元，只留第一行）")
    for prompt in PROBES:
        print(f"\n{prompt}")
        for name, (fn, t) in models.items():
            out = greedy(fn, lambda s: t.encode(s).ids, t.decode, prompt, 12)
            print(f"  {ljust(name, 14)}{out[:60]}")


if __name__ == "__main__":
    main()
