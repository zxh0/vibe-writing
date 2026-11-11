"""第三章 · 用tokenizers库训练正式用的分词器

和手写的`bpe_train.py`做的是同一件事，只是用HuggingFace的Rust实现，快得多，
可以在更多的文档上训。预切分、特殊词元、字节字符表都和手写版、和SmolLM2一样：

    pre_tokenizer   Digits(individual_digits) → ByteLevel(use_regex)
    decoder         ByteLevel
    特殊词元        <|endoftext|>=0、<|im_start|>=1、<|im_end|>=2

存出来的`tokenizer.json`和SmolLM2的是同一种格式，第一本的引擎能直接读。

运行：
    cd code
    uv run python book/ch03/train_tokenizer.py                       # 10万篇，8192
    uv run python book/ch03/train_tokenizer.py --vocab 16384 --out tokenizer-16k.json
"""

import argparse
import time

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from bpe_train import SPECIALS
from corpus import TOKENIZER_DIR, iter_docs, wrap


def build(docs, vocab_size):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        # 256个字节字符一个不少
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(docs, trainer)
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=100_000)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--out", default="tokenizer.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    docs = list(iter_docs("train", args.docs))
    t1 = time.perf_counter()
    tok = build(docs, args.vocab)
    t2 = time.perf_counter()

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    path = TOKENIZER_DIR / args.out
    tok.save(str(path))
    print(f"文档         {len(docs):,} 篇，{sum(len(d) for d in docs):,} 字符")
    print(f"词表         {tok.get_vocab_size():,}")
    print(f"用时         读文档 {t1 - t0:.0f} 秒，训练 {t2 - t1:.0f} 秒")
    print(f"写到         {path.relative_to(TOKENIZER_DIR.parent.parent)}")

    text = "The year 2024 wasn't bad: tokenizers split 12,345 into digits."
    enc = tok.encode(text)
    print(f"\n{text}")
    print(f"→ {len(enc.ids)} 个词元：")
    print(wrap((repr(t) for t in enc.tokens), sep=" "))


if __name__ == "__main__":
    main()
