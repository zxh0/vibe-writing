"""第三章 · 分词 demo

给一段文字，看它被切成哪些 token。

运行：
    cd code
    uv run python book/ch03/tokenize_demo.py
    uv run python book/ch03/tokenize_demo.py "你想试的任意一段话"
"""

import sys
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer

# code/book/ch03/tokenize_demo.py → 上三层就是 code/
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "SmolLM2-135M"

TEXTS = [
    "你好吗",
    "Once upon a time",
    "Hello world",
]


def width(s):
    """终端上占几列：汉字占2列，其余占1列。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def fold(head, items, cols=78):
    """把列表接在head后面印出来，太长就在逗号后折行，续行对齐到第一个元素。"""
    pad = " " * (width(head) + 1)
    lines, cur = [], head + "["
    for k, x in enumerate(items):
        piece = repr(x) + ("]" if k == len(items) - 1 else ",")
        if k == 0 or width(cur) + 1 + width(piece) <= cols:
            cur += ("" if k == 0 else " ") + piece
        else:
            lines.append(cur)
            cur = pad + piece
    return "\n".join(lines + [cur if items else cur + "]"])


def show(tok, text):
    enc = tok.encode(text, add_special_tokens=False)

    print(f"\n原文    {text!r}")
    print(fold("词元    ", enc.tokens))
    print(fold("ID      ", enc.ids))
    print(f"共 {len(text)} 个字符，{len(enc.ids)} 个词元")


def main():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    print(f"词表大小 {tok.get_vocab_size()}")

    for text in sys.argv[1:] or TEXTS:
        show(tok, text)


if __name__ == "__main__":
    main()
