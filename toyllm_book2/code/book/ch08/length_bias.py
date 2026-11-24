"""第八章 · DPO是不是只学会了“长的就是好的”

偏好数据里，好回答比坏回答长的占了一多半。如果模型只学会“越长越好”，
它在验证集上也能对一多半。拿验证集的每一对，看DPO之后的模型：

    1. 好回答更长的那些对，和好回答更短的那些对，准确率各是多少；
    2. 两个回答的长度差，和隐式奖励的差距，有没有关系；
    3. 平均到每个词元，好回答、坏回答的对数概率各变了多少。

运行（在CPU上，两个模型各过一遍验证集，半个小时左右）：
    cd code
    uv run python book/ch08/length_bias.py
"""

import argparse
from pathlib import Path

import torch

from config import CODE, DATA
from dpo import n_pairs, read_pairs, sequence_logps, to_batch
from export import load


@torch.no_grad()
def logps(model, data, n, B=8):
    """每一对：好回答、坏回答的对数概率，以及两个回答各有几个词元。"""
    out = []
    for i in range(0, n, B):
        pairs = list(range(i, min(i + B, n)))
        x, y = to_batch(data, pairs, "cpu")
        p = sequence_logps(model, x, y)
        k = (y != -100).sum(-1)
        b = len(pairs)
        out.append(torch.stack([p[:b], p[b:], k[:b], k[b:]], -1))
    return torch.cat(out).double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, default=CODE / "runs" / "sft" / "model")
    ap.add_argument("--policy", type=Path,
                    default=CODE / "runs" / "dpo" / "model")
    ap.add_argument("--beta", type=float, default=0.1)
    args = ap.parse_args()

    data = read_pairs(DATA / "dpo", "val")
    n = n_pairs(data)
    ref = logps(load(args.ref).eval(), data, n)
    pol = logps(load(args.policy).eval(), data, n)
    nc, nr = pol[:, 2], pol[:, 3]
    chosen = args.beta * (pol[:, 0] - ref[:, 0])
    rejected = args.beta * (pol[:, 1] - ref[:, 1])
    margin = chosen - rejected
    right = margin > 0
    longer = nc > nr

    print(f"验证集{n:,}对，准确率{right.double().mean():.1%}")
    print(f"好回答更长的{longer.sum():,}对（{longer.double().mean():.1%}），"
          f"准确率{right[longer].double().mean():.1%}")
    print(f"好回答更短的{(~longer).sum():,}对，"
          f"准确率{right[~longer].double().mean():.1%}")
    r = torch.corrcoef(torch.stack([margin, nc - nr]))[0, 1]
    print(f"长度差（好减坏）和差距的相关系数 {r:+.2f}")
    per_c = ((pol[:, 0] - ref[:, 0]) / nc).mean()
    per_r = ((pol[:, 1] - ref[:, 1]) / nr).mean()
    print(f"平均每个词元的对数概率变化：好回答{per_c:+.3f}，坏回答{per_r:+.3f}")


if __name__ == "__main__":
    main()
