"""第十章 · 蒸馏数据：过滤老师的回答，和原来的回答各装一份

老师的回答先过一遍筛子，有问题的扔掉：

    没写完    写到上限还没停下
    空回答    什么也没写
    中文      问题是英文，回答里却有中日韩文字（Qwen3是中英双语的老师）
    重复      回答里的四元组（连续四个词元）一半以上是前面出现过的，
              原地打转
    太长      拼进对话以后，一行（1025个词元）放不下

留下的问题，装两份数据，问题一模一样，只有回答不同：

    teacher/       老师的回答
    orig/          smol-smoltalk原来的回答
    teacher-5k/    老师的回答，训练集只取前5000个，对照实验用

每份和第七章一样：train.bin、val.bin是词元，每行1025个；
train_mask.bin、val_mask.bin是掩码。一行装几段对话，best fit。

运行（在Mac上，一两分钟）：
    cd code
    uv run python book/ch10/prepare_distill.py
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from chat_template import EOT, TOKENIZER, ChatTemplate
from config import DATA

ROW = 1025
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def repeated(ids, n=4):
    """回答里的n元组，有多少比例是前面出现过的。"""
    seen, rep, total = set(), 0, 0
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        rep += g in seen
        total += 1
        seen.add(g)
    return rep / total if total else 0.0


def check(chat, p, t):
    """老师的回答能不能用。返回(状态, 拼好的ids, mask)。"""
    if not t["finished"]:
        return "没写完", None, None
    if not t["answer"]:
        return "空回答", None, None
    if CJK.search(t["answer"]) and not CJK.search(p["prompt"]):
        return "中文", None, None
    body = chat.tok.encode(t["answer"]).ids
    if repeated(body) > 0.5:
        return "重复", None, None
    ids, mask = chat.render([{"role": "user", "content": p["prompt"]},
                             {"role": "assistant", "content": t["answer"]}])
    if len(ids) > ROW:
        return "太长", None, None
    return "留下", ids, mask


def pack(convs, row, seed):
    """和第七章的一样：先装长的，每段放进“剩下的空间最少、但还放得下”
    的那一行，空着的地方补<|endoftext|>，最后把行的顺序打乱。"""
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


def write(out, split, convs, seed):
    out.mkdir(parents=True, exist_ok=True)
    convs = list(convs)
    random.Random(seed).shuffle(convs)
    rows, masks = pack(convs, ROW, seed)
    rows.tofile(out / f"{split}.bin")
    masks.tofile(out / f"{split}_mask.bin")
    return len(rows), int(masks.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA / "distill",
                    help="prompts.jsonl和teacher.jsonl所在的目录")
    ap.add_argument("--out", type=Path, default=None, help="默认和--data一样")
    ap.add_argument("--small", type=int, default=5000,
                    help="teacher-5k的训练集取几个问题")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", type=Path, default=TOKENIZER,
                    help="在Colab上用基座模型目录里的那一份")
    args = ap.parse_args()
    out = args.out or args.data

    chat = ChatTemplate(Tokenizer.from_file(str(args.tokenizer)))
    prompts = [json.loads(line) for line in (args.data / "prompts.jsonl").open()]
    answers = {}
    for line in (args.data / "teacher.jsonl").open():
        t = json.loads(line)
        answers[t["id"]] = t
    print(f"{len(prompts):,}个问题，老师写了{len(answers):,}个回答")

    kept = {"train": [], "val": []}
    for split in ("val", "train"):
        status = Counter()
        for p in prompts:
            if p["split"] != split or p["id"] not in answers:
                continue
            s, ids, mask = check(chat, p, answers[p["id"]])
            status[s] += 1
            if ids:
                orig = chat.render([{"role": "user", "content": p["prompt"]},
                                    {"role": "assistant", "content": p["orig"]}])
                kept[split].append((ids, mask, *orig))
        n = sum(status.values())
        print(f"{split}：{n:,}个回答")
        for s in ("留下", "没写完", "空回答", "中文", "重复", "太长"):
            print(f"  {s}{' ' * (8 - 2 * len(s))}{status[s]:>8,}  "
                  f"{status[s] / n:6.1%}")

    print("\n回答的长度（我们的分词器，词元数）：")
    for split in ("train", "val"):
        tl = [sum(c[1]) for c in kept[split]]
        ol = [sum(c[3]) for c in kept[split]]
        longer = sum(a > b for a, b in zip(tl, ol)) / len(tl)
        print(f"  {split:<6}老师的中位数{int(np.median(tl)):>4}  "
              f"原来的{int(np.median(ol)):>4}  老师写得更长的占{longer:.1%}")

    print()
    for name, part in [("teacher", lambda c: (c[0], c[1])),
                       ("orig", lambda c: (c[2], c[3]))]:
        for split in ("train", "val"):
            convs = [tuple(np.array(x, dtype=np.uint16 if k == 0 else np.uint8)
                           for k, x in enumerate(part(c))) for c in kept[split]]
            rows, learn = write(out / name, split, convs, args.seed)
            print(f"  {name}/{split}：{len(convs):,}段，装成{rows:,}行，"
                  f"算loss的词元{learn:,}个")
            if name == "teacher":
                if split == "train":
                    convs = convs[:args.small]
                rows, learn = write(out / f"teacher-{args.small // 1000}k",
                                    split, convs, args.seed)
                print(f"  teacher-{args.small // 1000}k/{split}：{len(convs):,}段，"
                      f"装成{rows:,}行，算loss的词元{learn:,}个")


if __name__ == "__main__":
    main()
