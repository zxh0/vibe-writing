"""第九章 · 核对GRPO的几个零件

    1. 优势：一组手编的奖励，按公式手算，和advantages比；一组全对或全错，
       优势全是0。
    2. KL：策略模型就是参考模型时，每个词元的KL都是0，loss里只剩
       -优势 × log p。
    3. 方向：一组回答里，只有一个对的。在这一组上走一小步，对的那个回答的
       log p应该变大，错的大多变小。和对的开头一样的错回答（67和68都以6
       开头）可能跟着沾光：共用的那几个词元，概率是一起被推高的。

运行（在CPU上，几秒钟）：
    cd code
    uv run python book/ch09/check_grpo.py
"""

import torch

from chat_template import ChatTemplate
from config import CODE
from export import load
from grpo import advantages, grpo_loss, token_logps
from tokenizers import Tokenizer


def check_advantages():
    r = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    adv = advantages(r, 4)
    mean, std = 0.5, (1 / 3) ** 0.5             # 两对两错：样本标准差
    by_hand = [(x - mean) / (std + 1e-4) for x in [1.0, 0.0, 0.0, 1.0]]
    print("1. 奖励 [1 0 0 1] [1 1 1 1]")
    print("   手算  " + " ".join(f"{v:+.4f}" for v in by_hand))
    print("   函数  " + " ".join(f"{v:+.4f}" for v in adv.tolist()))


def batch(chat, answers):
    """同一道题的几个回答，拼成rollout返回的样子。"""
    q = [{"role": "user", "content": "What is 23 + 45?"}]
    prompt = chat.prompt(q)
    rows = [prompt + chat.tok.encode(a).ids + [2] for a in answers]
    T = max(len(x) for x in rows)
    ids = torch.zeros((len(rows), T), dtype=torch.long)
    learn = torch.zeros((len(rows), T - len(prompt)), dtype=torch.bool)
    for i, x in enumerate(rows):
        ids[i, :len(x)] = torch.tensor(x)
        learn[i, :len(x) - len(prompt)] = True
    return ids, len(prompt), learn


def main():
    check_advantages()
    path = CODE / "runs" / "sft" / "model"
    policy, ref = load(path), load(path).eval()
    chat = ChatTemplate(Tokenizer.from_file(str(path / "tokenizer.json")))
    answers = ["23 + 45 = 68", "23 + 45 = 58", "23 + 45 = 78",
               "23 + 45 = 67"]
    ids, P, learn = batch(chat, answers)
    adv = advantages(torch.tensor([1.0, 0.0, 0.0, 0.0]), 4)

    logp = token_logps(policy, ids, P, learn)
    with torch.no_grad():
        ref_logp = token_logps(ref, ids, P, learn)
    loss, kl = grpo_loss(logp, ref_logp, adv, learn, 0.04)
    print(f"\n2. 策略模型就是参考模型：KL {kl.item():g}")

    before = (logp * learn).sum(-1).detach()
    opt = torch.optim.SGD(policy.parameters(), lr=1e-3)
    loss.backward()
    opt.step()
    with torch.no_grad():
        after = (token_logps(policy, ids, P, learn) * learn).sum(-1)
    print("\n3. 走一小步，每个回答的log p")
    print("   回答                奖励   优势       之前       之后")
    for a, r, v, b, c in zip(answers, [1, 0, 0, 0], adv.tolist(),
                             before.tolist(), after.tolist()):
        print(f"   {a:<18}{r:>4}{v:>+8.3f}{b:>11.4f}{c:>11.4f}")


if __name__ == "__main__":
    main()
