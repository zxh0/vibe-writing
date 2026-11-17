"""第五章 · 手写AdamW，和torch.optim.AdamW核对

AdamW给每个参数的每一个数，都单独记两样东西：

    m：梯度的滑动平均，梯度最近大致朝哪边（一阶矩）
    v：梯度平方的滑动平均，梯度最近大致有多大（二阶矩）

每一步按 m / √v 走，梯度一直很大的数，步子被除小；一直很小的，步子被放大。
再加上权重衰减：每一步先把参数本身缩小一点点。

写成torch.optim.Optimizer的子类，是为了和torch自带的优化器用法一样：
param_groups、zero_grad、state_dict都是现成的，存checkpoint也不用另写。

运行（在CPU上，用fp32，几秒钟）：
    cd code
    uv run python book/ch05/adamw.py
"""

import copy
import math

import torch

from config import Config
from model import LLM


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:                         # 第一次见到这个参数
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]

                # 权重衰减：参数自己先缩小一点，和梯度无关。这就是AdamW的W。
                p.mul_(1 - lr * wd)
                # m = β1·m + (1-β1)·g，v = β2·v + (1-β2)·g²
                m.lerp_(g, 1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                # m、v都从0起步，头几步偏小，除以(1-β^t)校正回来。
                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t
                # p -= lr · (m/bc1) / (√(v/bc2) + eps)
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)


def param_groups(model, weight_decay):
    """矩阵（词嵌入、各个投影）做权重衰减；RMSNorm的权重是向量，不做。"""
    params = [p for p in model.parameters() if p.requires_grad]
    return [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]


def max_rel_err(a, b):
    """所有参数里，最大的 |a-b| / |b|（整个张量的范数之比）。"""
    return max(((x - y).norm() / y.norm()).item() for x, y in zip(a, b))


def main():
    torch.manual_seed(0)
    model_ours = LLM(Config())
    model_ref = copy.deepcopy(model_ours)
    kw = dict(lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    opt_ours = AdamW(param_groups(model_ours, 0.1), **kw)
    # foreach=False：torch的逐个张量的写法，运算顺序和我们的一样。
    opt_ref = torch.optim.AdamW(param_groups(model_ref, 0.1), foreach=False, **kw)

    steps, B, T = 10, 2, 64
    gen = torch.Generator().manual_seed(1)
    for step in range(1, steps + 1):
        ids = torch.randint(0, 8192, (B, T + 1), generator=gen)
        x, y = ids[:, :-1], ids[:, 1:]
        for model, opt in ((model_ours, opt_ours), (model_ref, opt_ref)):
            opt.zero_grad()
            model(x, y)[1].backward()
            opt.step()
        ours = list(model_ours.parameters())
        ref = list(model_ref.parameters())
        same = all(torch.equal(a, b) for a, b in zip(ours, ref))
        print(f"第{step:>2}步  参数完全相同：{same}  最大相对误差 {max_rel_err(ours, ref):.1e}")

    n = sum(p.numel() for p in model_ours.parameters())
    n_state = sum(s["m"].numel() + s["v"].numel() for s in opt_ours.state.values())
    print(f"\n参数 {n:,} 个，优化器状态（m和v）{n_state:,} 个，是参数的{n_state / n:.0f}倍")


if __name__ == "__main__":
    main()
