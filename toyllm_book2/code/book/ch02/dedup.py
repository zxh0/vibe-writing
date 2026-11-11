"""第二章 · 去重：先精确去重，再用MinHash找近似重复

精确去重：正文规范化（小写、空白合并）以后算MD5，见过的就扔。

近似去重，四步：
    1. 切片：每篇按词切开，每连续5个词算一个片段
    2. 签名：准备k=112个哈希函数，每个函数取所有片段里最小的哈希值，得到112个数
    3. 分组：112个数分成b=14组、每组r=8个；两篇只要有一组完全相同，就是候选的重复对
    4. 合并：候选对连成一簇一簇，每簇只留第一篇
b=14、r=8和FineWeb用的一样，阈值大约是 (1/14)^(1/8) ≈ 0.72：
Jaccard相似度高于它的两篇，大概率会被找出来。

哈希函数用的是 h(x) = (a·x + b) mod p，p = 2^31 - 1。x是片段的32位基础哈希，
a、b各取112个随机数，就是112个不同的哈希函数。a·x < 2^63，uint64装得下。

运行：
    cd code
    uv run python book/ch02/dedup.py
"""

import hashlib
import json
from collections import defaultdict

import numpy as np

from clean import CLEAN, preview
from wet_sample import WET_DIR

DEDUP = WET_DIR / "dedup.jsonl"

N_GRAM = 5
B, R = 14, 8               # 14组，每组8个
K = B * R                  # 签名长度112
P = (1 << 31) - 1          # 一个梅森素数

rng = np.random.default_rng(0)
A = rng.integers(1, P, K, dtype=np.uint64)
C = rng.integers(0, P, K, dtype=np.uint64)


def normalize(text):
    return " ".join(text.lower().split())


def shingles(text):
    """每连续N_GRAM个词一个片段，返回片段的集合。"""
    words = normalize(text).split()
    return {" ".join(words[i:i + N_GRAM]) for i in range(max(len(words) - N_GRAM + 1, 1))}


def base_hash(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "little")


def signature(shingle_set):
    """K个哈希函数，各取最小值。

    x是(n,)，A和C是(K,)，广播成(n, K)再按列取最小。
    """
    x = np.fromiter((base_hash(s) for s in shingle_set), dtype=np.uint64)
    return ((x[:, None] * A + C) % P).min(axis=0)


def jaccard(a, b):
    return len(a & b) / len(a | b)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        i, j = self.find(i), self.find(j)
        if i != j:
            self.parent[max(i, j)] = min(i, j)   # 簇的代表总是编号最小的那篇


def main():
    pages = [json.loads(line) for line in open(CLEAN, encoding="utf-8")]
    print(f"输入         {len(pages):,} 篇")

    # 1. 精确去重。
    seen, exact = set(), []
    for p in pages:
        h = hashlib.md5(normalize(p["text"]).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            exact.append(p)
    print(f"精确去重     扔掉 {len(pages) - len(exact):,} 篇，剩 {len(exact):,} 篇")

    # 2. 签名。
    sets = [shingles(p["text"]) for p in exact]
    sigs = np.stack([signature(s) for s in sets])                  # (n, K)

    # 3. 分组：同一组的8个数一模一样，就落进同一个桶。
    uf = UnionFind(len(exact))
    pairs = set()
    for band in range(B):
        buckets = defaultdict(list)
        for i, row in enumerate(sigs[:, band * R:(band + 1) * R]):
            buckets[row.tobytes()].append(i)
        for ids in buckets.values():
            for j in ids[1:]:
                pairs.add((ids[0], j))
                uf.union(ids[0], j)

    # 4. 每簇只留代表。
    near = [p for i, p in enumerate(exact) if uf.find(i) == i]
    clusters = defaultdict(list)
    for i in range(len(exact)):
        clusters[uf.find(i)].append(i)
    multi = [ids for ids in clusters.values() if len(ids) > 1]
    print(f"近似去重     候选对 {len(pairs):,} 个，连成 {len(multi):,} 簇，"
          f"扔掉 {len(exact) - len(near):,} 篇，剩 {len(near):,} 篇")

    # 签名估计的相似度，和真实的Jaccard相似度比一比。
    print("\n几对近似重复（签名估计 / 真实Jaccard）")
    for i, j in sorted(pairs)[:5]:
        est = (sigs[i] == sigs[j]).mean()
        print(f"  {est:.2f} / {jaccard(sets[i], sets[j]):.2f}")
        print(f"    {exact[i]['url'][:90]}")
        print(f"    {exact[j]['url'][:90]}")
        print(f"    {preview(exact[i]['text'], 100)}")

    biggest = max(multi, key=len)
    print(f"\n最大的一簇有 {len(biggest)} 篇，前三篇的网址：")
    for i in biggest[:3]:
        print(f"  {exact[i]['url'][:90]}")

    with open(DEDUP, "w", encoding="utf-8") as f:
        for p in near:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n写到 {DEDUP.name}")


if __name__ == "__main__":
    main()
