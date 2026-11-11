"""第二章 · 下载并解析一个Common Crawl的WET文件

一个快照切成了十万个WET文件，我们只拿第一个。WET文件是WARC格式，
一条记录一个网页，长这样：

    WARC/1.0
    WARC-Type: conversion
    WARC-Target-URI: https://example.com/some/page
    WARC-Identified-Content-Language: eng
    Content-Length: 1234
                                      ← 一个空行
    （1234字节的纯文本）

头部一行一个字段，空行之后是正文，正文多长由Content-Length说了算。
文件开头还有一条`warcinfo`记录，说明这个文件本身，跳过它。

解析结果存成JSON Lines：一行一个网页，记网址、Common Crawl自己识别的语言、正文。

运行：
    cd code
    uv run python book/ch02/wet_sample.py
"""

import gzip
import json
import urllib.request

from download import DATA, download

CRAWL = "CC-MAIN-2026-43"
BASE = "https://data.commoncrawl.org"
WET_DIR = DATA / "wet"
PAGES = WET_DIR / "pages.jsonl"


def first_wet_path():
    """快照的文件清单本身也是一个gz文件，一行一个WET文件的路径。取第一行。"""
    with urllib.request.urlopen(f"{BASE}/crawl-data/{CRAWL}/wet.paths.gz") as resp:
        return gzip.decompress(resp.read()).decode().splitlines()[0]


def read_records(path):
    """逐条读WARC记录，产出(头部字典, 正文字节)。"""
    with gzip.open(path, "rb") as f:
        while line := f.readline():
            if not line.startswith(b"WARC/"):
                continue
            headers = {}
            while (line := f.readline().strip()):
                key, _, value = line.decode("utf-8", "replace").partition(":")
                headers[key.strip()] = value.strip()
            body = f.read(int(headers["Content-Length"]))
            yield headers, body


def main():
    WET_DIR.mkdir(parents=True, exist_ok=True)
    rel = first_wet_path()
    path = WET_DIR / rel.rsplit("/", 1)[1]
    print(f"下载 {rel} ...", flush=True)
    download(f"{BASE}/{rel}", path)
    print(f"文件大小     {path.stat().st_size:,} 字节（压缩后）")

    n, n_chars, langs = 0, 0, {}
    with open(PAGES, "w", encoding="utf-8") as out:
        for headers, body in read_records(path):
            if headers.get("WARC-Type") != "conversion":
                continue
            text = body.decode("utf-8", "replace")
            lang = headers.get("WARC-Identified-Content-Language", "")
            out.write(json.dumps({"url": headers["WARC-Target-URI"], "lang": lang,
                                  "text": text}, ensure_ascii=False) + "\n")
            n += 1
            n_chars += len(text)
            first = lang.split(",")[0]          # 可能写了好几种语言，第一种是主要的
            langs[first] = langs.get(first, 0) + 1

    print(f"网页数       {n:,}")
    print(f"字符数       {n_chars:,}，平均每页 {n_chars // n:,}")
    print("语言（Common Crawl自己识别的，前十）")
    for lang, c in sorted(langs.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {lang or '(空)':<6}{c:>8,}  {c / n:.1%}")
    print(f"\n写到 {PAGES.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
