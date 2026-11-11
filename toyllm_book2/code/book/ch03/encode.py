"""第三章 · 给全部语料分词，存成训练用的二进制文件

第二章定下的格式：每篇文档分词以后，末尾接一个`<|endoftext|>`，
所有文档首尾相连，存成一个`uint16`的扁平数组。训练集和验证集各一个文件：

    data/tokens/train.bin
    data/tokens/val.bin

分词用`tokenizers`的`encode_batch`，Rust那边会用上所有CPU核。一次喂一千篇，
写完一批再取下一批，内存里不会同时放下整个分片。

运行：
    cd code
    uv run python book/ch03/encode.py
"""

import textwrap
import time

import numpy as np
from tokenizers import Tokenizer

from corpus import DATA, TOKENIZER_DIR, iter_docs, wrap

TOKENS_DIR = DATA / "tokens"
BATCH = 1000


def encode_split(tok, split, eot):
    path = TOKENS_DIR / f"{split}.bin"
    n_docs = n_tokens = n_chars = 0
    batch = []

    def flush(f):
        nonlocal n_tokens
        ids = []
        for enc in tok.encode_batch(batch, add_special_tokens=False):
            ids.extend(enc.ids)
            ids.append(eot)
        np.array(ids, dtype=np.uint16).tofile(f)
        n_tokens += len(ids)
        batch.clear()

    with open(path, "wb") as f:
        for text in iter_docs(split):
            batch.append(text)
            n_docs += 1
            n_chars += len(text)
            if len(batch) == BATCH:
                flush(f)
        if batch:
            flush(f)
    return path, n_docs, n_chars, n_tokens


def main():
    tok = Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    assert tok.get_vocab_size() <= 65536, "uint16装不下"
    TOKENS_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("val", "train"):
        t0 = time.perf_counter()
        path, n_docs, n_chars, n_tokens = encode_split(tok, split, eot)
        print(f"{split:<6}{n_docs:>8,} 篇{n_chars:>14,} 字符{n_tokens:>12,} 词元"
              f"  {time.perf_counter() - t0:>4.0f} 秒", flush=True)
        print(f"{'':<6}{n_chars / n_tokens:>8.2f} 字符/词元{path.stat().st_size:>18,} 字节", flush=True)

    # 读回来看一眼：开头的几个词元，以及解码回去的文字。
    data = np.memmap(TOKENS_DIR / "val.bin", dtype=np.uint16, mode="r")
    head = data[:40].tolist()
    print("\nval.bin的前40个词元：")
    print(wrap([f"{x}," for x in head[:-1]] + [str(head[-1])], sep=" "))
    print("解码回去（折行是为了排版加的）：")
    print(textwrap.indent(textwrap.fill(repr(tok.decode(head)), 74), "  "))


if __name__ == "__main__":
    main()
