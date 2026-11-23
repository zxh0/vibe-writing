"""第八章 · 偏好数据：下载UltraFeedback，把每一对回答按对话模板分好词

UltraFeedback：每个问题由几个大模型各答一遍，再由GPT-4打分。我们用的是
HuggingFaceH4整理的binarized版：分最高的回答当chosen（好的），剩下的
随机取一个当rejected（差的）。训练集61,135对，测试集2,000对，用作验证集。
SmolLM2的对话版做DPO用的就是它，许可证MIT。

每一对是同一个问题的两段对话，一段以好回答结尾，一段以坏回答结尾。
两段都按第七章的对话模板拼好，掩码只在回答上是1。扔掉两种：

    1. 两个回答分数相同，分不出好坏；
    2. 任何一段超过1025个词元，放不进模型（截短会截掉回答的结尾，
       好坏可能就变了）。

和SFT不同，这里不把几段装进一行：DPO要分别算出每个回答的概率，
一段对话占一行，长短不一，训练时一批一批地补齐。

输出到data/dpo/，每个split三个文件：

    train.bin、val.bin               词元编号，uint16，所有对话首尾相连
    train_mask.bin、val_mask.bin     掩码，uint8，和上面一一对应
    train_offsets.npy、val_...       每段对话的起点，第2i段是第i对的好回答，
                                     第2i+1段是坏回答，最后一个数是总长度

运行（下载约230MB，分词在Mac上一两分钟）：
    cd code
    uv run python book/ch08/prepare_dpo.py
"""

import argparse
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from chat_template import TOKENIZER, ChatTemplate
from config import DATA

BASE = ("https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized"
        "/resolve/main/data")
FILES = {"train": "train_prefs-00000-of-00001.parquet",
         "val": "test_prefs-00000-of-00001.parquet"}
RAW = DATA / "ultrafeedback"
OUT = DATA / "dpo"
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


def prepare(split, path, chat, out):
    t0 = time.perf_counter()
    cols = ["chosen", "rejected", "score_chosen", "score_rejected"]
    table = pq.read_table(path, columns=cols).to_pylist()
    status, seqs = Counter(), []
    for r in table:
        if r["score_chosen"] == r["score_rejected"]:
            status["分数相同"] += 1
            continue
        pair = [chat.render(r["chosen"]), chat.render(r["rejected"])]
        if max(len(ids) for ids, _ in pair) > ROW:
            status["太长"] += 1
            continue
        status["留下"] += 1
        seqs += pair

    lens = np.array([len(ids) for ids, _ in seqs])
    offsets = np.concatenate([[0], np.cumsum(lens)])
    np.array([t for ids, _ in seqs for t in ids], dtype=np.uint16).tofile(
        out / f"{split}.bin")
    np.array([m for _, mask in seqs for m in mask], dtype=np.uint8).tofile(
        out / f"{split}_mask.bin")
    np.save(out / f"{split}_offsets.npy", offsets)

    n = len(table)
    print(f"{split}：{n:,}对，用时{time.perf_counter() - t0:.0f}秒")
    for s in ("留下", "分数相同", "太长"):
        pad = " " * (8 - 2 * len(s))        # 汉字占两格
        print(f"  {s}{pad}{status[s]:>7,}对  "
              f"{status[s] / n:6.1%}")
    good, bad = lens[0::2], lens[1::2]
    print(f"  长度中位数：好回答那一段{np.median(good):.0f}个词元，"
          f"坏回答那一段{np.median(bad):.0f}个")
    print(f"  好回答那一段更长的占{(good > bad).mean():.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW, help="下载的parquet放在哪")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tokenizer", type=Path, default=TOKENIZER,
                    help="在Colab上用模型目录里的那一份")
    args = ap.parse_args()

    args.raw.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    chat = ChatTemplate(Tokenizer.from_file(str(args.tokenizer)))
    for split, name in FILES.items():
        print(f"下载 {name} ...", flush=True)
        download(f"{BASE}/{name}", args.raw / name)
        prepare(split, args.raw / name, chat, args.out)


if __name__ == "__main__":
    main()
