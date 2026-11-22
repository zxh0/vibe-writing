"""第六章 · 对照实验的数据：同一批文档，换三种词表重新分词

词表大小的对照实验，要用4096、8192、16384三种词表各训一个模型。三个模型
应该看同样的文字，所以这里不去重读第二章的语料，而是从第三章的train.bin里
取开头的一段文档，用原来的分词器还原成文字，再用三个分词器各分一遍。
字节级的BPE能把文字原样还原，8192那一份重新分出来，和原来的编号一个不差，
脚本会顺手检查这一点。验证集整个照此处理。

输出和data/一样的结构，train.py用--data指过去：

    data/vocab-4k/tokens/train.bin、val.bin
    data/vocab-4k/tokenizer/tokenizer.json

运行（三种词表，训练集各取约1亿个词元，在Mac上几分钟）：
    cd code
    uv run python book/ch06/retokenize.py
"""

import argparse
import shutil
import time

import numpy as np
from tokenizers import Tokenizer

from config import DATA

VOCABS = {
    "vocab-4k": "tokenizer-4k.json",
    "vocab-8k": "tokenizer.json",
    "vocab-16k": "tokenizer-16k.json",
}
EOT = 0           # <|endoftext|>，三个分词器的编号都是0
BATCH = 1000


def iter_docs(path, limit):
    """按结束符切开，每次产出一批文档的词元编号，只取前limit个词元以内的。"""
    data = np.memmap(path, dtype=np.uint16, mode="r")
    if limit:
        data = data[:limit]
    batch, start = [], 0
    for end in np.flatnonzero(data == EOT):
        batch.append(data[start:end].tolist())
        start = end + 1
        if len(batch) == BATCH:
            yield batch
            batch = []
    if batch:
        yield batch


def retokenize(split, limit, src, toks):
    files = {name: open(DATA / name / "tokens" / f"{split}.bin", "wb")
             for name in toks}
    n_docs, n_bytes = 0, 0
    n_tokens = dict.fromkeys(toks, 0)
    for batch in iter_docs(DATA / "tokens" / f"{split}.bin", limit):
        texts = src.decode_batch(batch, skip_special_tokens=False)
        n_docs += len(texts)
        n_bytes += sum(len(t.encode("utf-8")) for t in texts)
        for name, tok in toks.items():
            ids = []
            for i, enc in enumerate(tok.encode_batch(texts,
                                                     add_special_tokens=False)):
                if name == "vocab-8k":
                    assert enc.ids == batch[i], "还原以后再分词，和原来不一样"
                ids.extend(enc.ids)
                ids.append(EOT)
            np.array(ids, dtype=np.uint16).tofile(files[name])
            n_tokens[name] += len(ids)
    for f in files.values():
        f.close()
    return n_docs, n_bytes, n_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100_000_000,
                    help="训练集取原来train.bin的前多少个词元")
    args = ap.parse_args()

    src = Tokenizer.from_file(str(DATA / "tokenizer" / "tokenizer.json"))
    toks = {}
    for name, file in VOCABS.items():
        (DATA / name / "tokens").mkdir(parents=True, exist_ok=True)
        (DATA / name / "tokenizer").mkdir(parents=True, exist_ok=True)
        shutil.copy(DATA / "tokenizer" / file,
                    DATA / name / "tokenizer" / "tokenizer.json")
        toks[name] = Tokenizer.from_file(str(DATA / "tokenizer" / file))
        assert toks[name].token_to_id("<|endoftext|>") == EOT

    for split, limit in [("val", None), ("train", args.limit)]:
        t0 = time.perf_counter()
        n_docs, n_bytes, n_tokens = retokenize(split, limit, src, toks)
        print(f"{split}：{n_docs:,}篇文档，{n_bytes:,}个字节，"
              f"用时{time.perf_counter() - t0:.0f}秒")
        for name, n in n_tokens.items():
            print(f"  {name:<10}{n:>14,}个词元  每词元{n_bytes / n:.2f}个字节")


if __name__ == "__main__":
    main()
