"""第七章 · Paris是从哪学来的

SFT之后，模型答得出法国的首都是Paris；基座模型在第六章的探针问题里
却没答对过。这件事是预训练学的，还是SFT学的？两个办法看一看：

    1. 看概率：给两个模型同一个开头“The capital of France is”，
       下一个词元是“ Paris”的概率有多大，在8192个词元里排第几。
       开头分两种：直接续写，和按对话模板问完以后助手这样开口。
    2. 数一数：SFT的训练集里，有多少个回答里写着这句话。

运行：
    cd code
    uv run python book/ch07/probe_fact.py
"""

from pathlib import Path

import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer

from chat_template import ChatTemplate
from config import CODE, DATA
from export import load

RUNS = CODE / "runs"
QUESTION = "What is the capital of France?"
START = "The capital of France is"
FACT = "capital of France is Paris"


@torch.no_grad()
def next_token(model, ids, target):
    """下一个词元是target的概率，以及它在所有词元里排第几。"""
    p = model(torch.tensor([ids]))[0][0, -1].softmax(-1)
    return p[target].item(), (p > p[target]).sum().item() + 1


def main():
    tok = Tokenizer.from_file(str(RUNS / "sft" / "model" / "tokenizer.json"))
    chat = ChatTemplate(tok)
    (paris,) = tok.encode(" Paris").ids
    prompts = {
        "直接续写": tok.encode(START).ids,
        "对话里回答": chat.prompt([{"role": "user", "content": QUESTION}])
                     + tok.encode(START).ids,
    }
    print(f"“{START}”之后，下一个词元是“ Paris”的概率和排名\n")
    print("  模型       开头            概率   排名")
    for name in ("pretrain", "sft"):
        model = load(RUNS / name / "model").eval()
        for how, ids in prompts.items():
            p, rank = next_token(model, ids, paris)
            pad = " " * (12 - 2 * len(how))     # 汉字占两格
            print(f"  {name:<10} {how}{pad}{p:8.2%}  {rank:>5}")

    n = total = 0
    for path in sorted((DATA / "smol-smoltalk").glob("train-*.parquet")):
        for messages in pq.read_table(path, columns=["messages"])[0].to_pylist():
            for m in messages:
                if m["role"] == "assistant":
                    total += 1
                    n += FACT in m["content"]
    print(f"\nSFT训练集的{total:,}个回答里，有{n}个写着“{FACT}”")


if __name__ == "__main__":
    main()
