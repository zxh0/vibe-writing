"""第六章 · 对照实验：几个模型在同一批文字上比每字节比特数

词表不一样，每个词元的loss不能直接比，要换算成每字节比特数，见eval.py。
这里取验证集的前几十篇文档，还原成文字，每个模型用自己的分词器各算一遍。
只留在每个分词器下都不超过1023个词元的文档，免得超出训练时的长度。

运行（几个训完导出的模型目录）：
    cd code
    uv run python book/ch06/compare_bpb.py runs/exp-base/model \\
        runs/exp-untie/model runs/exp-vocab4k/model runs/exp-vocab16k/model
"""

import argparse
import math
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from config import DATA
from eval import ljust, rjust, total_loss
from export import load


def val_texts(n, toks, max_len=1023):
    """验证集按结束符切开，用第三章的分词器还原成文字，取前n篇在每个
    分词器下都不超过max_len个词元的。"""
    src = Tokenizer.from_file(str(DATA / "tokenizer" / "tokenizer.json"))
    val = np.memmap(DATA / "tokens" / "val.bin", dtype=np.uint16, mode="r")
    texts, start = [], 0
    for end in np.flatnonzero(val == 0):
        text = src.decode(val[start:end].tolist())
        start = end + 1
        if all(len(t.encode(text).ids) <= max_len for t in toks):
            texts.append(text)
            if len(texts) == n:
                break
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", type=Path, nargs="+", help="导出的模型目录")
    ap.add_argument("--docs", type=int, default=100)
    args = ap.parse_args()

    toks = [Tokenizer.from_file(str(m / "tokenizer.json")) for m in args.models]
    texts = val_texts(args.docs, toks)
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    print(f"验证集前{len(texts)}篇文档，共{n_bytes:,}个字节\n")
    print(ljust("模型", 18) + rjust("词表", 8) + rjust("参数", 14)
          + rjust("词元数", 10) + rjust("每词元loss", 13) + rjust("每字节比特", 13))
    for path, tok in zip(args.models, toks):
        model = load(path).eval()
        n_params = sum(p.numel() for p in model.parameters())
        loss, n = total_loss(lambda x: model(x)[0], lambda s: tok.encode(s).ids,
                             texts, bos=0)
        name = path.parent.name if path.name == "model" else path.name
        print(f"{name:<18}{tok.get_vocab_size():>8,}{n_params:>14,}{n:>10,}"
              f"{loss / n:>13.3f}{loss / n_bytes / math.log(2):>13.3f}")


if __name__ == "__main__":
    main()
