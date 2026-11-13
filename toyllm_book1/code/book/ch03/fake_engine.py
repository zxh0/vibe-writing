"""第三章 · 只有两头的推理引擎

    文字 -> 词元 -> ID -> [ LLM + 采样器 ] -> 新ID -> 词元 -> 文字

方括号里的东西要到第十章才凑齐，这里先用随机数冒充，让整条流水线先转起来。

运行：
    cd code
    uv run python book/ch03/fake_engine.py
    uv run python book/ch03/fake_engine.py "你想试的任意一段话"
"""

import random
import re
import sys
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "SmolLM2-135M"
SEED = 20260916  # 固定住，好让书里引用的输出能复现


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


def fold_text(head, text, cols=78):
    """把repr(text)接在head后面印出来，太长就断开，一段一行。

    尽量在空格前断；一个词自己就放不下时，只好从词中间断。
    """
    pad = " " * width(head)
    pieces, cur = [], ""
    for w in re.findall(r"\s*\S+|\s+$", text):
        for c in w if width(pad + repr(w)) > cols else [w]:
            if cur and width(pad + repr(cur + c)) > cols:
                pieces.append(cur)
                cur = ""
            cur += c
    pieces.append(cur)
    return "\n".join((pad if k else head) + repr(p) for k, p in enumerate(pieces))


def fake_llm(tok, ids, n):
    """每轮吐一个随机ID，追加到序列末尾，转n圈。

    真正的引擎在这里跑30个Decoder块。
    """
    rng = random.Random(SEED)
    for _ in range(n):
        ids.append(rng.randrange(tok.get_vocab_size()))
    return ids


def main():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    text = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"

    enc = tok.encode(text, add_special_tokens=False)   # Tokenizer：文字到ID
    ids = fake_llm(tok, list(enc.ids), 10)             # 冒充LLM和采样器，转10圈
    out = tok.decode(ids)                              # Detokenizer：ID回到文字

    print(f"文字   {text!r}")
    print(fold("词元   ", enc.tokens))
    print(fold("ID     ", enc.ids))

    print("\n冒充LLM，转10圈，每圈追加一个ID：")
    for k, i in enumerate(ids[len(enc.ids):], len(enc.ids) + 1):
        print(f"  第{k:>2}个词元   ID {i:>5}   {tok.id_to_token(i)!r}")

    print("\n" + fold("ID     ", ids))
    print(fold("词元   ", [tok.id_to_token(i) for i in ids]))
    print(fold_text("文字   ", out))

    # 解码的单位是整串ID，不是单个词元
    zids = tok.encode("第三章", add_special_tokens=False).ids
    print(f"\n'第三章' 的{len(zids)}个词元 {[tok.id_to_token(i) for i in zids]}")
    print(f"逐个解码再拼   {''.join(tok.decode([i]) for i in zids)!r}")
    print(f"整串一次解码   {tok.decode(zids)!r}")


if __name__ == "__main__":
    main()
