"""第八章 · 核对DPO的loss

三件事：

    1. dpo_loss：取一组手编的对数概率，按公式一步一步手算，和函数的结果比。
    2. sequence_logps：真模型、真数据，一个词元一个词元地取log p再加起来，
       和函数一次算出的结果比。
    3. 起点：策略模型和参考模型是同一个，隐式奖励全是0，
       每一对的loss都应该是-log σ(0) = log 2。

运行（在CPU上，十几秒）：
    cd code
    uv run python book/ch08/check_dpo.py
"""

import math

import torch

from config import CODE, DATA
from dpo import dpo_loss, read_pairs, sequence_logps, to_batch
from export import load


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def check_formula():
    pc, pr, rc, rr, beta = -10.0, -12.0, -11.0, -11.0, 0.1
    chosen = beta * (pc - rc)                     # 好回答比参考模型高了1
    rejected = beta * (pr - rr)                   # 坏回答比参考模型低了1
    by_hand = -math.log(sigmoid(chosen - rejected))
    t = lambda v: torch.tensor([v], dtype=torch.float64)
    loss, c, r = dpo_loss(t(pc), t(pr), t(rc), t(rr), beta)
    print("1. 手编的一对：好回答 -10（参考 -11），坏回答 -12（参考 -11），"
          f"beta {beta}")
    print(f"   隐式奖励  好 {chosen:+.3f}  坏 {rejected:+.3f}")
    print(f"   手算 {by_hand:.6f}   dpo_loss {loss.item():.6f}")


@torch.no_grad()
def check_logps(model, data):
    x, y = to_batch(data, [0, 1], "cpu")
    fast = sequence_logps(model, x, y)
    logp = model(x)[0].float().log_softmax(-1)
    slow = []
    for row in range(len(y)):
        total = 0.0
        for t in range(y.shape[1]):
            if y[row, t] != -100:
                total += logp[row, t, y[row, t]].item()
        slow.append(total)
    n = (y != -100).sum(-1).tolist()
    print("\n2. 验证集前两对，四段回答的对数概率（前两段好，后两段坏）")
    print("   词元数     逐个加起来   sequence_logps")
    for k, s, f in zip(n, slow, fast.tolist()):
        print(f"   {k:>6}  {s:13.4f}  {f:15.4f}")
    err = max(abs(s - f) / abs(s) for s, f in zip(slow, fast.tolist()))
    print(f"   最大相对误差 {err:.1e}")


@torch.no_grad()
def check_start(model, data):
    pairs = list(range(8))
    x, y = to_batch(data, pairs, "cpu")
    p = sequence_logps(model, x, y)
    B = len(pairs)
    loss, c, r = dpo_loss(p[:B], p[B:], p[:B], p[B:], 0.1)
    print("\n3. 策略模型就是参考模型：验证集前8对")
    print(f"   隐式奖励最大的绝对值 {torch.cat([c, r]).abs().max().item():g}")
    print(f"   loss {loss.item():.6f}   log 2 = {math.log(2):.6f}")


def main():
    model = load(CODE / "runs" / "sft" / "model").eval()
    data = read_pairs(DATA / "dpo", "val")
    check_formula()
    check_logps(model, data)
    check_start(model, data)


if __name__ == "__main__":
    main()
