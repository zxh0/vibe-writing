"""第十章 · 挑问题：从smol-smoltalk里挑单轮的问题，连同原来的回答

每段对话只看开头两轮：用户提问，助手回答。第一轮是系统提示的跳过。
挑出来的问题要满足：

    问题不超过256个词元（按我们的分词器）；
    原来的回答拼进对话以后，整段放得进一行（1025个词元）；
    问题不重复，一字不差的只留一个。

训练集从train的4个文件里随机挑--n个，验证集从test里挑--n-val个。
原来的回答也存下来：同一个问题，一份数据用它，一份用老师写的，
训出两个学生来比。

输出data/distill/prompts.jsonl，每行一个问题：

    {"id": "train-0", "split": "train", "source": 出自smol-smoltalk的
     哪一部分, "prompt": 问题, "orig": 原来的回答}

运行（数据第七章已经下载过；没有的话先下载，约1GB）：
    cd code
    uv run python book/ch10/pick_prompts.py
"""

import argparse
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from chat_template import TOKENIZER, ChatTemplate
from config import DATA

BASE = "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/resolve/main/data"
RAW = DATA / "smol-smoltalk"
OUT = DATA / "distill"
ROW = 1025
MAX_PROMPT = 256


def download(url, path):
    """和第七章的一样：先写到.part，下完再改名，断了重跑会接着下。"""
    if path.exists():
        return
    part = path.with_name(path.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
    with urllib.request.urlopen(req) as resp, open(part, "ab") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    part.rename(path)


def width(s):
    return sum(2 if ord(c) > 0x2e80 else 1 for c in s)


def pick(chat, records, n, seed, split, seen):
    """按随机顺序看对话，挑出n个合格的问题。返回(挑中的, 各种情况的计数)。
    seen是已经挑过的问题，训练集不要和验证集重复。"""
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    out, status = [], Counter()
    for i in range(0, len(order), 1000):
        batch = []
        for j in order[i:i + 1000]:
            ms = records[j]["messages"]
            if len(ms) < 2 or ms[0]["role"] != "user" \
                    or ms[1]["role"] != "assistant":
                status["开头不是一问一答"] += 1
            elif ms[0]["content"] in seen:
                status["重复的问题"] += 1
            else:
                seen.add(ms[0]["content"])
                batch.append(j)
        enc = chat.tok.encode_batch(
            [m["content"] for j in batch for m in records[j]["messages"][:2]],
            add_special_tokens=False)
        for k, j in enumerate(batch):
            ms = records[j]["messages"][:2]
            bodies = [enc[2 * k].ids, enc[2 * k + 1].ids]
            if len(bodies[0]) > MAX_PROMPT:
                status["问题太长"] += 1
            elif len(chat.render(ms, bodies)[0]) > ROW:
                status["回答太长"] += 1
            else:
                status["挑中"] += 1
                out.append({"id": f"{split}-{len(out)}", "split": split,
                            "source": records[j]["source"],
                            "prompt": ms[0]["content"],
                            "orig": ms[1]["content"]})
                if len(out) == n:
                    return out, status
    return out, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="训练集挑几个问题")
    ap.add_argument("--n-val", type=int, default=500, help="验证集挑几个问题")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw", type=Path, default=RAW, help="下载的parquet放在哪")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tokenizer", type=Path, default=TOKENIZER,
                    help="在Colab上用基座模型目录里的那一份")
    args = ap.parse_args()

    args.raw.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    names = {"val": ["test-00000-of-00001.parquet"],
             "train": [f"train-{i:05d}-of-00004.parquet" for i in range(4)]}
    chat = ChatTemplate(Tokenizer.from_file(str(args.tokenizer)))
    picked, seen = [], set()
    for split, files in names.items():
        records = []
        for name in files:
            print(f"下载 {name} ...", flush=True)
            download(f"{BASE}/{name}", args.raw / name)
            records += pq.read_table(args.raw / name).to_pylist()
        n = args.n_val if split == "val" else args.n
        out, status = pick(chat, records, n, args.seed, split, seen)
        picked += out
        n_seen = sum(status.values())
        print(f"{split}：{len(records):,}段对话，看了{n_seen:,}段，"
              f"挑中{len(out):,}个")
        for s, c in status.most_common():
            print(f"  {s}{' ' * (18 - width(s))}{c:>7,}  {c / n_seen:6.1%}")
        items = [f"{s} {c:,}" for s, c in
                 Counter(p["source"] for p in out).most_common()]
        for k in range(0, len(items), 3):       # 一行三个，免得太宽
            print(("  来源：" if k == 0 else " " * 8) + "，".join(items[k:k + 3]))

    path = args.out / "prompts.jsonl"
    with path.open("w") as f:
        for p in picked:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"写到 {path}")


if __name__ == "__main__":
    main()
