"""第三章 · 词表多大：压缩率和嵌入参数之间的取舍

在同样的10万篇文档上，训练4096、8192、16384、32768四种大小的词表，
再拿验证集来量：

    每个词元平均几个字符     越大越好：同样的文本切出的词元越少，训练越快
    词嵌入矩阵有多少参数     越小越好：我们的模型一共才两千多万参数

作为参照，再量两个现成的分词器：SmolLM2的（49152个词元），以及GPT-2的
（50257个词元，FineWeb-Edu的`token_count`字段就是拿它数的，不用另外下载）。

SmolLM2的`tokenizer.json`第一次运行时会下载到`code/models/SmolLM2-135M/`。

运行：
    cd code
    uv run python book/ch03/vocab_sizes.py
"""

import urllib.request

from tokenizers import Tokenizer

from corpus import DATA, TOKENIZER_DIR, iter_docs
from train_tokenizer import build

D_MODEL = 512
REST = 22_553_088          # 模型里除了词嵌入以外的参数：8层 × 2,819,072 + 最后的RMSNorm 512
SIZES = [4096, 8192, 16384, 32768]
SMOLLM2 = DATA.parent / "models" / "SmolLM2-135M" / "tokenizer.json"
SMOLLM2_URL = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/tokenizer.json"


def tokenizer_path(size):
    return TOKENIZER_DIR / ("tokenizer.json" if size == 8192 else f"tokenizer-{size // 1024}k.json")


def main():
    for size in SIZES:
        path = tokenizer_path(size)
        if not path.exists():
            print(f"训练 {size} ...", flush=True)
            build(iter_docs("train", 100_000), size).save(str(path))
    if not SMOLLM2.exists():
        SMOLLM2.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SMOLLM2_URL, SMOLLM2)

    val = list(iter_docs("val", with_tokens=True))
    texts = [t for t, _ in val]
    n_chars = sum(len(t) for t in texts)
    gpt2_tokens = sum(n for _, n in val)
    print(f"验证集 {len(texts):,} 篇，{n_chars:,} 字符\n")

    print(f"{'分词器':<14}{'词表':>8}{'词元数':>13}{'字符/词元':>10}{'词嵌入参数':>14}{'占模型':>8}")
    rows = [(f"本书 {s // 1024}k", s, tokenizer_path(s)) for s in SIZES]
    rows.append(("SmolLM2", 49152, SMOLLM2))
    for name, size, path in rows:
        tok = Tokenizer.from_file(str(path))
        n = sum(len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False))
        emb = size * D_MODEL
        print(f"{name:<14}{size:>8,}{n:>13,}{n_chars / n:>10.2f}{emb:>14,}{emb / (emb + REST):>8.1%}")
    emb = 50257 * D_MODEL
    print(f"{'GPT-2':<14}{50257:>8,}{gpt2_tokens:>13,}{n_chars / gpt2_tokens:>10.2f}"
          f"{emb:>14,}{emb / (emb + REST):>8.1%}")

    # 看看同一句话，不同大小的词表怎么切。
    text = "Photosynthesis converts sunlight."
    print(f"\n{text}")
    for name, size, path in rows:
        toks = Tokenizer.from_file(str(path)).encode(text).tokens
        print(f"  {name.removeprefix('本书 '):<8}{len(toks):>2}  {' '.join(toks)}")


if __name__ == "__main__":
    main()
