"""第五章 · 梯度累积和大batch，算出来的梯度一样吗

同样8段文本，两种算法：

    一次算：B=8，前向、反向各一次
    累积：  B=4，分两次前向、反向，每次的loss先除以2，梯度加在一起

两种算法的梯度在数学上相等，只差浮点误差。

运行（CPU，fp32）：
    cd code
    uv run python book/ch05/check_accum.py
"""

import numpy as np
import torch

from config import DATA, Config
from model import LLM

B, T, ACCUM = 8, 256, 2


def grads(model):
    return [p.grad.clone() for p in model.parameters()]


def main():
    data = np.memmap(DATA / "tokens" / "val.bin", dtype=np.uint16, mode="r")
    chunk = torch.from_numpy(data[:B * (T + 1)].astype(np.int64)).view(B, T + 1)
    x, y = chunk[:, :-1], chunk[:, 1:]
    torch.manual_seed(0)
    model = LLM(Config())

    model.zero_grad()
    model(x, y)[1].backward()
    once = grads(model)

    model.zero_grad()
    for xs, ys in zip(x.chunk(ACCUM), y.chunk(ACCUM)):
        (model(xs, ys)[1] / ACCUM).backward()
    accum = grads(model)

    err = max(((a - b).norm() / b.norm()).item() for a, b in zip(accum, once))
    print(f"一次算B={B}，和累积{ACCUM}次B={B // ACCUM}：梯度最大相对误差 {err:.1e}")

    # 常见的错误：忘了除以ACCUM，梯度就大了ACCUM倍。
    model.zero_grad()
    for xs, ys in zip(x.chunk(ACCUM), y.chunk(ACCUM)):
        model(xs, ys)[1].backward()
    wrong = grads(model)
    ratio = (sum(g.norm() ** 2 for g in wrong) / sum(g.norm() ** 2 for g in once)) ** 0.5
    print(f"忘了除以{ACCUM}：梯度范数是正确值的{ratio.item():.2f}倍")


if __name__ == "__main__":
    main()
