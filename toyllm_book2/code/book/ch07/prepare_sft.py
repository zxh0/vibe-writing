"""第七章 · SFT数据：下载smol-smoltalk，按对话模板分词，装进定长的行

smol-smoltalk是SmolLM2-135M和360M的对话版用的SFT数据，约46万段英文对话，
许可证Apache 2.0。训练集4个parquet文件，共约0.9GB；测试集1个，用作验证集。

三步：

    1. 每段对话按chat_template.py拼成词元，同时得到掩码。
    2. 太长的对话，只留前面放得下的几轮（截在某个回答的末尾）；
       连第一个回答都放不下的，整段扔掉。
    3. 把对话装进长1025的行（1024个输入，错开一位就是答案），一行可以
       装几段，空着的地方用<|endoftext|>补齐，掩码是0。

一行可能装着几段对话。后面的对话能看到前面的，和预训练时一篇文档能看到
前一篇一样，影响不大，换来的是几乎不浪费算力。

输出到data/sft/：

    train.bin、val.bin             词元编号，uint16，每行1025个
    train_mask.bin、val_mask.bin   掩码，uint8，和上面一一对应

运行（下载约1GB，分词在Mac上几分钟）：
    cd code
    uv run python book/ch07/prepare_sft.py
    uv run python book/ch07/prepare_sft.py --shards 1   # 只用训练集的四分之一
"""

import argparse
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from chat_template import EOT, TOKENIZER, ChatTemplate
from config import DATA

BASE = "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/resolve/main/data"
RAW = DATA / "smol-smoltalk"
OUT = DATA / "sft"
ROW = 1025


def download(url, path):
    """和第二章的一样：先写到.part，下完再改名，断了重跑会接着下。"""
    if path.exists():
        return
    part = path.with_name(path.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
    with urllib.request.urlopen(req) as resp, open(part, "ab") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    part.rename(path)


def fit(chat, messages, bodies, limit):
    """拼好一段对话；超过limit个词元，就截到放得下的最后一个回答为止。
    返回(ids, mask, 状态)，状态是“完整”“截短”“扔掉”之一。"""
    ids, mask = chat.render(messages, bodies)
    if len(ids) <= limit:
        return ids, mask, "完整"
    # 每个回答末尾的换行符就是一个可以截断的位置。
    ends, n = [], 0
    for m, body in zip(messages, bodies):
        n += len(chat.header[m["role"]]) + len(body) + 1 + len(chat.newline)
        if m["role"] == "assistant" and n <= limit:
            ends.append(n)
    if not ends:
        return None, None, "扔掉"
    return ids[:ends[-1]], mask[:ends[-1]], "截短"


def pack(convs, row, seed):
    """把对话装进长row的行，行尾空着的地方补<|endoftext|>。

    按顺序装、装不下就换行，四分之一的位置会空着：截短的对话都接近一整行，
    后面跟来的放不进去。所以先装长的，每段放进“剩下的空间最少、但还放得下”
    的那一行（装箱问题里的best fit）。先只看长度定下位置，再一次填好，
    最后把行的顺序打乱。"""
    free = [[] for _ in range(row + 1)]     # free[f]：还剩f个空位的那些行
    places, n = [], 0
    for c in sorted(range(len(convs)), key=lambda c: -len(convs[c][0])):
        size = len(convs[c][0])
        for f in range(size, row + 1):
            if free[f]:
                r = free[f].pop()
                break
        else:
            r, f, n = n, row, n + 1
        places.append((c, r, row - f))
        free[f - size].append(r)
    rows = np.full((n, row), EOT, dtype=np.uint16)
    masks = np.zeros((n, row), dtype=np.uint8)
    for c, r, at in places:
        ids, mask = convs[c]
        rows[r, at:at + len(ids)] = ids
        masks[r, at:at + len(ids)] = mask
    order = np.random.default_rng(seed).permutation(n)
    return rows[order], masks[order]


def prepare(split, files, chat, seed, out):
    t0 = time.perf_counter()
    status, convs = Counter(), []
    for path in files:
        table = pq.read_table(path, columns=["messages"]).to_pylist()
        for i in range(0, len(table), 1000):
            batch = [r["messages"] for r in table[i:i + 1000]]
            # 一批对话里所有轮次的内容放在一起分词，比一条一条地分快得多。
            flat = chat.tok.encode_batch([m["content"] for ms in batch for m in ms],
                                         add_special_tokens=False)
            k = 0
            for messages in batch:
                bodies = [e.ids for e in flat[k:k + len(messages)]]
                k += len(messages)
                ids, mask, s = fit(chat, messages, bodies, ROW)
                status[s] += 1
                if ids:
                    # 存成numpy数组：几亿个词元，存成Python的整数列表，
                    # 每个数要占几十个字节，内存放不下。
                    convs.append((np.array(ids, dtype=np.uint16),
                                  np.array(mask, dtype=np.uint8)))
        print(f"  {path.name}：累计{sum(status.values()):,}段", flush=True)
    random.Random(seed).shuffle(convs)
    rows, masks = pack(convs, ROW, seed)
    rows.tofile(out / f"{split}.bin")
    masks.tofile(out / f"{split}_mask.bin")

    n_conv = sum(status.values())
    n_tok = sum(len(c[0]) for c in convs)
    print(f"{split}：{n_conv:,}段对话，用时{time.perf_counter() - t0:.0f}秒")
    for s in ("完整", "截短", "扔掉"):
        print(f"  {s}  {status[s]:>9,}段  {status[s] / n_conv:6.1%}")
    print(f"  装成{len(rows):,}行，共{rows.size:,}个位置：对话占{n_tok / rows.size:.1%}，"
          f"算loss的占{masks.sum() / rows.size:.1%}，补齐的占{1 - n_tok / rows.size:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=4, help="训练集用前几个文件，共4个")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw", type=Path, default=RAW, help="下载的parquet放在哪")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tokenizer", type=Path, default=TOKENIZER,
                    help="在Colab上用基座模型目录里的那一份")
    args = ap.parse_args()

    args.raw.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    names = {"val": ["test-00000-of-00001.parquet"],
             "train": [f"train-{i:05d}-of-00004.parquet" for i in range(args.shards)]}
    for files in names.values():
        for name in files:
            print(f"下载 {name} ...", flush=True)
            download(f"{BASE}/{name}", args.raw / name)

    chat = ChatTemplate(Tokenizer.from_file(str(args.tokenizer)))
    for split, files in names.items():
        prepare(split, [args.raw / f for f in files], chat, args.seed, args.out)


if __name__ == "__main__":
    main()
