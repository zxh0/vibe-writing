"""第十章 · 整机对拍：四级验收

和第八章那个一样。这一章没动模型，生成循环换了写法，默认的采样器是贪婪，
所以四级验收照样要全过：采样器接进来以后，贪婪那条路一个ID都不能变。
按严格程度递进，一级一级过：

    一  参数量          和第一章一个不多一个不少
    二  31层逐层比      相对误差max|diff| / max|ref|
    三  每个位置的argmax 一个都不能错
    四  贪婪解码20个词元  和第一章那句话一字不差

这个脚本不导入transformers，只读第一章`baseline.py`存下的文件。

逐层比的下标是transformers的约定，很容易搞错：

    [0]        词嵌入的输出
    [1]..[29]  第i个块的输入，也就是第i-1个块的输出
    [30]       最后一个块的输出，而且已经过了出口norm

运行：
    cd code
    uv run python book/ch01/baseline.py     # 还没有基准文件的话，先跑这个
    uv run python book/ch10/check_forward.py
"""

import re
import torch
import unicodedata

from engine import Engine
from weights import MODEL_DIR

BASELINE = MODEL_DIR.parent.parent / "reference" / "baseline.pt"


def width(s):
    """终端上占几列：汉字占2列，其余占1列。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


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


def hidden_states(model, ids):
    """走一遍30层，按transformers的约定留下31份中间结果，各(T, 576)。"""
    x = model.to_embeddings(ids)                       # (T, 576)
    states = [x]
    for layer in model.weights.layers:
        x = model.block(x, layer)
        states.append(x)
    states[-1] = model.rms_norm(x, model.weights.norm)  # 最后一份要过出口norm
    return states


def main():
    ref = torch.load(BASELINE)
    engine = Engine()
    model = engine.model
    ids = ref["input_ids"][0].tolist()                 # [6403, 1980, 253, 655]

    print(f"输入   {ref['prompt']!r}   {ids}")

    # ---- 一：参数量 -----------------------------------------------------------
    ours = model.weights.n_params
    print(f"\n一  参数量    我们{ours:,}   第一章{ref['n_params']:,}   "
          f"{'相同' if ours == ref['n_params'] else '不同'}")

    # ---- 二：31层逐层比 -------------------------------------------------------
    with torch.no_grad():
        states = hidden_states(model, ids)
    print(f"\n二  逐层比   {'下标':>4}{'绝对误差':>12}{'相对误差':>12}")
    worst = 0.0
    for i, (h, r) in enumerate(zip(states, ref["hidden_states"][:, 0])):
        abs_err = (h - r).abs().max().item()
        rel = abs_err / r.abs().max().item()
        worst = max(worst, rel)
        if i % 5 == 0 or i == 29:
            print(f"            {i:>4}{abs_err:>14.1e}{rel:>14.1e}")
    print(f"            31层里最大的相对误差   {worst:.1e}")

    # ---- 三：argmax ----------------------------------------------------------
    logits = model.to_logits(states[-1])               # (T, 49152)
    ours = logits.argmax(-1).tolist()
    theirs = ref["logits"][0].argmax(-1).tolist()
    print(f"\n三  argmax   我们{ours}   第一章{theirs}   "
          f"{'相同' if ours == theirs else '不同'}")

    # ---- 四：贪婪解码 ---------------------------------------------------------
    out = engine.generate(ids)
    theirs = ref["greedy"][0].tolist()
    text = engine.tok.decode(out)
    print(f"\n四  贪婪解码")
    print(fold_text("    我们     ", text))
    print(fold_text("    第一章   ", ref["greedy_text"]))
    print(f"    ID{'相同' if out == theirs else '不同'}，"
          f"文字{'相同' if text == ref['greedy_text'] else '不同'}")


if __name__ == "__main__":
    main()
