"""第四章 · 初始化对不对，看第一步的loss

模型什么都没学的时候，最合理的猜法是：8192个词元一视同仁，每个的概率都是
1/8192。这时的交叉熵是ln(8192) ≈ 9.01。初始化得当，第一步的loss应该就在
这附近；差得远，说明模型一开始就“很有把握地猜错”，训练要先花力气把这份
错误的自信磨掉，有时候还会直接训崩。

这里比三种初始化，在验证集的同一批数据上看第一步的loss：

    本书          正态分布，标准差0.02；o_proj、down_proj再除以√(2×层数)
    不缩小残差    全部标准差0.02，不对o_proj、down_proj特殊处理
    PyTorch默认   nn.Embedding是标准正态（标准差1），nn.Linear是均匀分布

运行：
    cd code
    uv run python book/ch04/init_loss.py
"""

import math

import numpy as np
import torch
import torch.nn as nn

from config import DATA, Config
from model import LLM

B, T = 8, 256


def torch_default(model):
    """什么都不做时的样子：每个子模块用PyTorch自带的初始化。

    lm_head和词嵌入是同一个矩阵，最后由nn.Embedding说了算：标准正态分布。
    """
    for m in model.modules():
        if isinstance(m, nn.Linear) and m is not model.lm_head:
            m.reset_parameters()
    model.model.embed_tokens.reset_parameters()


def no_residual_scaling(model):
    for p in model.parameters():
        if p.dim() == 2:
            nn.init.normal_(p, mean=0.0, std=model.config.initializer_range)


def width(s):
    """显示宽度：汉字占两列。"""
    return sum(2 if ord(c) > 0x2e80 else 1 for c in s)


def row(*cells):
    """第一列左对齐占12列，其余右对齐各占14列。"""
    first, *rest = cells
    return first + " " * (12 - width(first)) + "".join(" " * (14 - width(c)) + c for c in rest)


def measure(model, x, y):
    """第一步的loss、logits的标准差、最后一层输出（残差流）的标准差，
    以及有多少个位置，得分最高的恰好是当前这个词元自己。"""
    seen = {}
    hook = model.model.layers[-1].register_forward_hook(
        lambda m, i, out: seen.update(resid=out.std().item()))
    with torch.no_grad():
        logits, loss = model(x, y)
    hook.remove()
    selfish = (logits.argmax(-1) == x).float().mean().item()
    return loss.item(), logits.std().item(), seen["resid"], selfish


def main():
    data = np.memmap(DATA / "tokens" / "val.bin", dtype=np.uint16, mode="r")
    # 从验证集开头切B段，每段T+1个词元：前T个是输入，后T个是答案。
    chunk = torch.from_numpy(data[:B * (T + 1)].astype(np.int64)).view(B, T + 1)
    x, y = chunk[:, :-1], chunk[:, 1:]

    cfg = Config()
    print(f"均匀猜测的loss：ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.3f}\n")
    print(row("初始化", "第一步loss", "logits标准差", "残差流标准差", "猜成自己"))
    for name, tweak in [("本书", None), ("不缩小残差", no_residual_scaling),
                        ("PyTorch默认", torch_default)]:
        torch.manual_seed(0)
        model = LLM(cfg)
        if tweak:
            tweak(model)
        loss, s_logits, s_resid, selfish = measure(model, x, y)
        print(row(name, f"{loss:.3f}", f"{s_logits:.3f}", f"{s_resid:.3f}", f"{selfish:.1%}"))


if __name__ == "__main__":
    main()
