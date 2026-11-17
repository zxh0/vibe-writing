"""第五章 · 先过拟合一个batch

正式训练之前的体检：只拿一个batch，反复训。2,700万个参数去记4×256个词元，
绰绰有余，loss应该一路降到接近0。降不下去，一定是代码哪里写错了：
答案没错开一位、梯度没清零、学习率没生效、参数没交给优化器……

不用warmup、不用cosine，学习率固定；训练循环的其他部分和train.py一样。

运行：
    cd code
    uv run python book/ch05/overfit.py
"""

import json

import numpy as np
import torch

from adamw import AdamW, param_groups
from config import DATA, Config
from model import LLM
from train import RUNS, default_device

B, T, STEPS, LR = 4, 256, 150, 1e-3


def main():
    device = default_device()
    data = np.memmap(DATA / "tokens" / "train.bin", dtype=np.uint16, mode="r")
    # 训练集开头的4段，从头到尾只用这一批。
    chunk = torch.from_numpy(data[:B * (T + 1)].astype(np.int64)).view(B, T + 1).to(device)
    x, y = chunk[:, :-1], chunk[:, 1:]

    torch.manual_seed(0)
    model = LLM(Config()).to(device)
    opt = AdamW(param_groups(model, 0.0), lr=LR)
    out = RUNS / "overfit"
    out.mkdir(parents=True, exist_ok=True)
    print(f"设备 {device}，一个batch：{B}×{T}，学习率固定为{LR}")
    with open(out / "log.jsonl", "w") as log:
        for step in range(1, STEPS + 1):
            opt.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            log.write(json.dumps({"step": step, "loss": loss.item()}) + "\n")
            if step == 1 or step % 10 == 0:
                print(f"第{step:>4}步  loss {loss.item():.4f}", flush=True)

    # 记住了没有：拿这一批的开头，看模型能不能一字不差地背出后面。
    with torch.no_grad():
        model.eval()
        logits, loss = model(x, y)
        acc = (logits.argmax(-1) == y).float().mean().item()
    print(f"\n训完以后：loss {loss.item():.4f}，{acc:.1%}的位置猜中了下一个词元")


if __name__ == "__main__":
    main()
