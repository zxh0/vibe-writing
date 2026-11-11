"""第三章 · 手写一个字节级BPE训练器

训练分三步：

    1. 预切分：先把文本切成一个个片段，合并只发生在片段内部。规则和SmolLM2一样：
       每个数字单独成一段，其余的按GPT-2的正则表达式切；再把每段的UTF-8字节
       换成可打印字符（GPT-2那张字节到字符的映射表，第一本第三章讲过）。
    2. 数片段：同样的片段只存一份，记下它出现了多少次。
    3. 合并：数所有相邻两个词元的出现次数，把最多的一对合成一个新词元，
       重复，直到词表够大。

第3步要重复几千次。每次都从头数一遍太慢，所以只更新受影响的片段：
`where`记着每一对出现在哪些片段里，合并时只改这些片段，顺手把计数加减好。
找“最多的一对”用一个堆，计数变了不去堆里删旧的，弹出来时发现过时了再放回去。

两对一样多时选哪一对？我们和`tokenizers`库保持一致：选编号小的那一对。
编号的规矩也和它一样：特殊词元在最前面，然后是256个字节字符，按字符的码位排序，
之后每合并出一个新词元，编号就往后加一。

运行：
    cd code
    uv run python book/ch03/bpe_train.py                  # 默认在1万篇文档上训8192的词表
    uv run python book/ch03/bpe_train.py --docs 1000      # 先试一试
"""

import argparse
import heapq
import json
import time
import unicodedata
from collections import Counter, defaultdict

import regex

from corpus import TOKENIZER_DIR, iter_docs, wrap

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

# GPT-2的预切分规则：缩写、一串字母、一串数字、一串符号，前面都可以带一个空格；
# 剩下的是空白。\p{L}是任意语言的字母，\p{N}是任意数字，所以要用regex库。
GPT2_PATTERN = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def byte_to_char():
    """GPT-2那张字节到可打印字符的映射表，和第一本第三章的一字不差。"""
    keep = list(range(ord("!"), ord("~") + 1))
    keep += list(range(ord("¡"), ord("¬") + 1))
    keep += list(range(ord("®"), ord("ÿ") + 1))
    chars = list(keep)
    n = 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(keep, chars)}


BYTE_TO_CHAR = byte_to_char()


def split_digits(text):
    """每个数字单独成一段。判断“是不是数字”要看Unicode类别是否以N开头，
    和Rust的char::is_numeric()一致（第一本第三章踩过这个坑）。"""
    pieces, start = [], 0
    for i, ch in enumerate(text):
        if unicodedata.category(ch)[0] == "N":
            if start < i:
                pieces.append(text[start:i])
            pieces.append(ch)
            start = i + 1
    if start < len(text):
        pieces.append(text[start:])
    return pieces


def pre_tokenize(text):
    """文本 → 片段列表，每个片段已经换成了可打印字符。"""
    out = []
    for piece in split_digits(text):
        for m in GPT2_PATTERN.findall(piece):
            out.append("".join(BYTE_TO_CHAR[b] for b in m.encode("utf-8")))
    return out


def count_words(docs):
    counts = Counter()
    for text in docs:
        counts.update(pre_tokenize(text))
    return counts


def merge_word(word, a, b, new_id):
    """把一个片段里所有相邻的(a, b)换成new_id，从左往右，不重叠。"""
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def train(word_counts, vocab_size):
    """返回(词表, 合并规则)。词表是编号到字符串的列表，合并规则是(左, 右)的列表。"""
    vocab = SPECIALS + sorted(BYTE_TO_CHAR.values())
    tok2id = {t: i for i, t in enumerate(vocab)}
    words = [[tok2id[c] for c in w] for w in word_counts]
    freqs = list(word_counts.values())

    pair_count = Counter()
    where = defaultdict(set)
    for i, (w, f) in enumerate(zip(words, freqs)):
        for pair in zip(w, w[1:]):
            pair_count[pair] += f
            where[pair].add(i)

    # 堆里放(-次数, 左, 右)：次数多的先出来，一样多就是编号小的先出来。
    heap = [(-c, a, b) for (a, b), c in pair_count.items()]
    heapq.heapify(heap)
    merges = []
    while len(vocab) < vocab_size and heap:
        neg, a, b = heapq.heappop(heap)
        count = pair_count.get((a, b), 0)
        if count != -neg:                       # 过时了：按现在的次数放回去
            if count > 0:
                heapq.heappush(heap, (-count, a, b))
            continue

        new = vocab[a] + vocab[b]
        new_id = tok2id.get(new)
        if new_id is None:                      # 不同的合并顺序可能拼出同一个字符串
            new_id = len(vocab)
            vocab.append(new)
            tok2id[new] = new_id
        merges.append((vocab[a], vocab[b]))

        changed = set()
        for i in where.pop((a, b)):
            w, f = words[i], freqs[i]
            if not any(x == a and y == b for x, y in zip(w, w[1:])):
                continue                        # 这个片段早就不含这一对了
            for pair in zip(w, w[1:]):
                pair_count[pair] -= f
            w = merge_word(w, a, b, new_id)
            for pair in zip(w, w[1:]):
                pair_count[pair] += f
                where[pair].add(i)
                changed.add(pair)
            words[i] = w
        pair_count.pop((a, b), None)
        for pair in changed:
            heapq.heappush(heap, (-pair_count[pair], *pair))
    return vocab, merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=10_000)
    ap.add_argument("--vocab", type=int, default=8192)
    args = ap.parse_args()

    t0 = time.perf_counter()
    docs = list(iter_docs("train", args.docs))
    word_counts = count_words(docs)
    t1 = time.perf_counter()
    vocab, merges = train(word_counts, args.vocab)
    t2 = time.perf_counter()

    print(f"文档         {len(docs):,} 篇，{sum(len(d) for d in docs):,} 字符")
    print(f"片段         {sum(word_counts.values()):,} 个，不重复的 {len(word_counts):,} 个")
    print(f"词表         {len(vocab):,}（特殊词元{len(SPECIALS)} + 字节256 + 合并{len(merges):,}）")
    print(f"用时         预切分加计数 {t1 - t0:.0f} 秒，合并 {t2 - t1:.0f} 秒")
    print("\n最常见的片段：")
    print(wrap(f"{w!r} {c:,}" for w, c in word_counts.most_common(8)))
    print("\n前20条合并规则：")
    print(wrap(f"{a}+{b}" for a, b in merges[:20]))
    print("\n最后10个新词元：")
    print(wrap(vocab[-10:]))

    out = TOKENIZER_DIR / "hand"
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(vocab)},
                                               ensure_ascii=False, indent=0))
    (out / "merges.txt").write_text("\n".join(f"{a} {b}" for a, b in merges) + "\n")
    print(f"\n写到 {out.relative_to(TOKENIZER_DIR.parent.parent)}/")


if __name__ == "__main__":
    main()
