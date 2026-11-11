"""第三章 · 读第二章准备好的语料

第二章下载了FineWeb-Edu的分片，按id的哈希切出了验证集。这里只是把“按顺序读出
训练集或验证集的文档”这件事包成一个函数，本章的几个脚本都用它。

`is_val`和第二章`split.py`里的一字不差：同一篇文档，算出来永远归同一边。
"""

import hashlib
from pathlib import Path

import pyarrow.parquet as pq

# code/book/ch03/corpus.py → 上三层就是 code/
DATA = Path(__file__).resolve().parent.parent.parent / "data"
FINEWEB_DIR = DATA / "fineweb-edu"
TOKENIZER_DIR = DATA / "tokenizer"
VAL_PER_MILLE = 2


def wrap(items, indent="  ", sep="  ", width=76):
    """把一串短字符串排成几行，每行不超过width列。

    书里的输出要印在纸上，一行太长就会被折得乱七八糟，所以长列表都用它来排。
    """
    lines, cur = [], ""
    for s in items:
        if cur and len(indent + cur + sep + s) > width:
            lines.append(indent + cur)
            cur = s
        else:
            cur = cur + sep + s if cur else s
    lines.append(indent + cur)
    return "\n".join(lines)


def is_val(doc_id):
    h = int(hashlib.md5(doc_id.encode()).hexdigest(), 16)
    return h % 1000 < VAL_PER_MILLE


def shard_paths():
    return sorted(FINEWEB_DIR.glob("*_00000.parquet"))


def iter_docs(split="train", limit=None, with_tokens=False):
    """按顺序产出训练集（或验证集）的文档正文，最多limit篇。

    with_tokens=True时产出(正文, GPT-2的token数)，比较压缩率时用。
    """
    n = 0
    for path in shard_paths():
        f = pq.ParquetFile(path)
        for group in range(f.num_row_groups):
            table = f.read_row_group(group, columns=["id", "text", "token_count"])
            ids, texts, counts = (table.column(c).to_pylist() for c in ("id", "text", "token_count"))
            for doc_id, text, count in zip(ids, texts, counts):
                if is_val(doc_id) != (split == "val"):
                    continue
                yield (text, count) if with_tokens else text
                n += 1
                if limit is not None and n >= limit:
                    return
