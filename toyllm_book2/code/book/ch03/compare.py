"""第三章 · 手写的训练器，和tokenizers库训出来的对答案

两边在同一批文档上、用同样的词表大小训练，然后逐条比较合并规则，再比较词表。
合并规则一条不差、顺序也一样，编码结果就一定一样：BPE编码只看这张表。

先跑一遍`bpe_train.py`，它会把手写版的结果存到`data/tokenizer/hand/`。

运行：
    cd code
    uv run python book/ch03/bpe_train.py
    uv run python book/ch03/compare.py
"""

import argparse
import json

from corpus import TOKENIZER_DIR, iter_docs
from train_tokenizer import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=10_000, help="要和bpe_train.py用的一样")
    ap.add_argument("--vocab", type=int, default=8192)
    args = ap.parse_args()

    hand = TOKENIZER_DIR / "hand"
    hand_merges = [tuple(line.split(" ")) for line in
                   (hand / "merges.txt").read_text().splitlines()]
    hand_vocab = json.loads((hand / "vocab.json").read_text())

    tok = build(iter_docs("train", args.docs), args.vocab)
    model = json.loads(tok.to_str())["model"]
    lib_merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" "))
                  for m in model["merges"]]
    lib_vocab = model["vocab"]

    same = next((i for i, (x, y) in enumerate(zip(hand_merges, lib_merges)) if x != y),
                min(len(hand_merges), len(lib_merges)))
    print(f"合并规则     手写 {len(hand_merges):,} 条，tokenizers {len(lib_merges):,} 条")
    print(f"从头开始一模一样的 {same:,} 条")
    if same < min(len(hand_merges), len(lib_merges)):
        print(f"第{same + 1}条开始不同：手写 {hand_merges[same]}，tokenizers {lib_merges[same]}")
    print(f"两边都有的合并规则 {len(set(hand_merges) & set(lib_merges)):,} 条")
    print(f"词表         完全相同：{hand_vocab == lib_vocab}")


if __name__ == "__main__":
    main()
