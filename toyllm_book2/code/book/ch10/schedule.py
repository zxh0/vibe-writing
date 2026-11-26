"""第五章 · 学习率：先warmup，再cosine

    第0步到第warmup步：从0线性升到max_lr
    之后到第total步：  沿半个余弦周期，从max_lr降到min_lr

和transformers的get_cosine_with_min_lr_schedule_with_warmup核对，每一步都要相同。

运行：
    cd code
    uv run python book/ch05/schedule.py
"""

import math

import torch


def lr_at(step, max_lr, warmup, total, min_lr=0.0):
    """第step步（从0数）的学习率。"""
    if step < warmup:
        return max_lr * (step / warmup)
    progress = (step - warmup) / max(1, total - warmup)   # 0 → 1
    progress = min(progress, 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))     # 1 → 0
    # 即 min_lr + (max_lr - min_lr) * cosine。先算出max_lr的几倍再乘，
    # 是为了和transformers的运算顺序一致，每一步逐位相同。
    ratio = min_lr / max_lr
    return max_lr * (cosine * (1 - ratio) + ratio)


def main():
    from transformers import get_cosine_with_min_lr_schedule_with_warmup

    max_lr, min_lr, warmup, total = 1e-3, 1e-4, 30, 300
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=max_lr)
    sched = get_cosine_with_min_lr_schedule_with_warmup(
        opt, num_warmup_steps=warmup, num_training_steps=total, min_lr=min_lr)

    ours, ref = [], []
    for step in range(total):
        ours.append(lr_at(step, max_lr, warmup, total, min_lr))
        ref.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()

    diff = max(abs(a - b) for a, b in zip(ours, ref))
    same = sum(a == b for a, b in zip(ours, ref))
    print(f"{total}步，完全相同的有{same}步，最大差 {diff:.1e}\n")
    print(f"{'步':>5}{'我们':>12}{'transformers':>14}")
    for step in (0, 1, 15, 29, 30, 31, 100, 165, 250, 299):
        print(f"{step:>5}{ours[step]:>13.3e}{ref[step]:>14.3e}")


if __name__ == "__main__":
    main()
