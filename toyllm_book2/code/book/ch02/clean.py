"""第二章 · 规则过滤：挑几条有代表性的规则，在一个WET文件上亲手做一遍

规则大多来自Gopher和C4，阈值也照抄它们；“行尾标点”那一条是FineWeb自己加的。
一篇网页按顺序过每一条规则，被哪一条拦下就记在哪一条的账上，后面的规则不再看它。
每条规则单独计数，再各抽两篇被拦下的看看，才知道它是太严、太松，还是刚好。

语言识别用一个最简单的近似：ASCII字符占多少。真正的语言识别要用fastText之类的模型，
这里只求不额外下载东西。跑完会和Common Crawl自己识别的语言对一对账。

运行：
    cd code
    uv run python book/ch02/clean.py
"""

import json

from wet_sample import PAGES, WET_DIR

CLEAN = WET_DIR / "clean.jsonl"

# Gopher的停用词：正常的英文文章里，这八个词至少会出现两个。
STOPWORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}
ENDINGS = (".", "!", "?", '"', "'", "”", "’")


def ascii_ratio(text):
    return sum(c.isascii() for c in text) / max(len(text), 1)


def mean_word_len(words):
    return sum(len(w) for w in words) / max(len(words), 1)


def symbol_ratio(text, words):
    return (text.count("#") + text.count("...") + text.count("…")) / max(len(words), 1)


def ends_with_punct(lines):
    return sum(line.rstrip().endswith(ENDINGS) for line in lines) / max(len(lines), 1)


def dup_line_ratio(lines):
    return 1 - len(set(lines)) / max(len(lines), 1)


# (名字, 判断“该扔”的函数)。按顺序检查，第一条说扔就扔。
RULES = [
    ("非英文（ASCII字符不到90%）", lambda t, w, l: ascii_ratio(t) < 0.9),
    ("太短或太长（词数不在50到100,000之间）",
     lambda t, w, l: not 50 <= len(w) <= 100_000),
    ("平均词长不在3到10之间", lambda t, w, l: not 3 <= mean_word_len(w) <= 10),
    ("停用词不到两个",
     lambda t, w, l: len(STOPWORDS & {x.lower() for x in w}) < 2),
    ("#和省略号太多（超过词数的10%）",
     lambda t, w, l: symbol_ratio(t, w) > 0.1),
    ("以标点结尾的行不到12%", lambda t, w, l: ends_with_punct(l) < 0.12),
    ("重复的行超过30%", lambda t, w, l: dup_line_ratio(l) > 0.3),
    ("含有lorem ipsum", lambda t, w, l: "lorem ipsum" in t.lower()),
]


def first_failed(text):
    """返回第一条拦下这篇网页的规则编号；都过了返回None。"""
    words = text.split()
    lines = [line for line in text.splitlines() if line.strip()]
    for i, (_, drop) in enumerate(RULES):
        if drop(text, words, lines):
            return i
    return None


def preview(text, n=120):
    return " ".join(text.split())[:n]


def main():
    pages = [json.loads(line) for line in open(PAGES, encoding="utf-8")]
    dropped = [[] for _ in RULES]
    kept = []
    for page in pages:
        i = first_failed(page["text"])
        (kept if i is None else dropped[i]).append(page)

    print(f"输入         {len(pages):,} 篇")
    for (name, _), pages_i in zip(RULES, dropped):
        print(f"  {len(pages_i):>7,}  {name}")
    print(f"留下         {len(kept):,} 篇（{len(kept) / len(pages):.1%}）")

    # 和Common Crawl自己识别的语言对账。
    cc_eng = [p for p in pages if p["lang"].startswith("eng")]
    eng_kept = sum(p["lang"].startswith("eng") for p in kept)
    lost = sum(p["lang"].startswith("eng") for p in dropped[0])
    print(f"\nCommon Crawl认为是英文的 {len(cc_eng):,} 篇，留下 {eng_kept:,} 篇，"
          f"其中被我们的语言规则误杀 {lost:,} 篇")
    print(f"留下的里面，Common Crawl认为不是英文的 {len(kept) - eng_kept:,} 篇")

    print("\n每条规则拦下的例子")
    for (name, _), pages_i in zip(RULES, dropped):
        print(f"\n[{name}]")
        for p in pages_i[:2]:
            print(f"  {p['url'][:80]}")
            print(f"    {preview(p['text'])}")

    with open(CLEAN, "w", encoding="utf-8") as f:
        for p in kept:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n写到 {CLEAN.name}")


if __name__ == "__main__":
    main()
