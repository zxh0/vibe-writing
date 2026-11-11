"""第二章 · 下载FineWeb-Edu的抽样版

FineWeb-Edu的`sample/10BT`是一份约100亿token的抽样，切成14个parquet文件，
每个2GB出头。我们的预算用不了这么多，默认只下第一个。

下载写到`.part`文件里，下完才改名，中途断了再跑一遍会接着下。

运行：
    cd code
    uv run python book/ch02/download.py              # 下第一个分片，再看看里面有什么
    uv run python book/ch02/download.py --shards 2   # 下前两个
"""

import argparse
import textwrap
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

# code/book/ch02/download.py → 上三层就是 code/
DATA = Path(__file__).resolve().parent.parent.parent / "data"
FINEWEB_DIR = DATA / "fineweb-edu"
BASE = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT"


def shard_name(i):
    return f"{i:03d}_00000.parquet"


def download(url, path):
    """断点续传：`.part`里已经有多少字节，就从第几个字节接着要。"""
    if path.exists():
        return
    part = path.with_name(path.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
    with urllib.request.urlopen(req) as resp, open(part, "ab") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    part.rename(path)


def shard_paths():
    return sorted(FINEWEB_DIR.glob("*_00000.parquet"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()

    FINEWEB_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(args.shards):
        name = shard_name(i)
        print(f"下载 {name} ...", flush=True)
        download(f"{BASE}/{name}", FINEWEB_DIR / name)

    # 看看第一个分片里有什么：多少篇文档、哪些字段、文档多长、分数怎么分布。
    path = shard_paths()[0]
    meta = pq.ParquetFile(path).metadata
    print(f"\n{path.name}")
    print(f"文件大小     {path.stat().st_size:,} 字节")
    print(f"文档数       {meta.num_rows:,}")
    print(textwrap.fill(", ".join(meta.schema.names), 78,
                        initial_indent="字段         ", subsequent_indent=" " * 13))

    table = pq.read_table(path, columns=["text", "token_count", "int_score"])
    n_chars = sum(len(t) for t in table.column("text").to_pylist())
    n_tokens = sum(table.column("token_count").to_pylist())
    print(f"字符数       {n_chars:,}")
    print(f"token数      {n_tokens:,}（GPT-2分词器数的）")
    print(f"平均每篇     {n_chars // meta.num_rows:,} 字符，{n_tokens // meta.num_rows:,} token")

    scores = table.column("int_score").to_pylist()
    print("分数分布     " + "，".join(
        f"{s}分 {scores.count(s) / len(scores):.1%}" for s in sorted(set(scores))))

    text = table.column("text")[0].as_py()
    print(f"\n第一篇的开头：\n{text[:500]}")


if __name__ == "__main__":
    main()
