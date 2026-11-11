from collections import Counter


# 统计所有相邻符号对出现的次数（要乘上单词本身出现的次数）
def count_pairs(words: dict) -> Counter:
    pairs = Counter()
    for symbols, freq in words.items():
        for pair in zip(symbols, symbols[1:]):
            pairs[pair] += freq
    return pairs


# 把每个单词里出现的某个符号对，合并成一个新符号
def merge_pair(words: dict, pair: tuple) -> dict:
    merged = {}
    for symbols, freq in words.items():
        out, i = [], 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                out.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        merged[tuple(out)] = freq
    return merged


def train_bpe(corpus: dict, num_merges: int) -> dict:
    words = {tuple(word): freq for word, freq in corpus.items()}  # 先拆成单个字母
    for step in range(1, num_merges + 1):
        pairs = count_pairs(words)
        best = max(pairs, key=pairs.get)  # 出现次数最多的那一对
        words = merge_pair(words, best)
        print(f"第{step}次合并：{best[0]} + {best[1]}（{pairs[best]}次）")
    return words


corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
words = train_bpe(corpus, 5)
print(list(words))
