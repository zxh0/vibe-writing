"""第九章 · 让模型做题：采样、判分、量正确率

rollout：一批题，每题让模型写G个回答，判分。强化学习里，模型自己写出
一个回答的过程叫一次rollout（展开）。

accuracy：固定的几组题，贪婪解码，量每种位数的正确率。冷启动和GRPO
都用它，前后才好比。

一批题的提示词要一样长，模型的generate才能B行一起写，所以同一批里的题
两个加数的位数都一样（数字是一个一个切的，位数一样，词元数就一样）。
"""

import random

import torch

from addition import make, question, reward
from chat_template import EOT, IM_END

STOP = (IM_END, EOT)


@torch.no_grad()
def rollout(model, chat, problems, G, max_new, temperature=1.0):
    """problems是[(a, b), ...]，每题写G个回答，第p题的第g个在第p*G+g行。

    返回一个dict：
        ids      (P*G, 提示词+回答)  提示词和写出的词元
        prompt   提示词的词元数
        learn    (P*G, 回答)  回答里要算进loss的词元：到<|im_end|>为止，
                 包括它；没停下的，整个回答都算
        texts    每个回答的文字（不含<|im_end|>）
        rewards  (P*G,)  每个回答的奖励，0或1
    """
    device = next(model.parameters()).device
    prompts = [chat.prompt([{"role": "user", "content": question(a, b)}])
               for a, b in problems]
    assert len({len(p) for p in prompts}) == 1, "同一批的提示词要一样长"
    ids = torch.tensor([p for p in prompts for _ in range(G)], device=device)
    P = ids.shape[1]
    ids = model.generate(ids, max_new, temperature=temperature, stop=STOP)
    answer = ids[:, P:]
    is_stop = (answer == IM_END) | (answer == EOT)
    stopped = is_stop.any(-1)
    # 第一个停止词元的位置；没停下的算到最后一个
    last = torch.full_like(stopped, answer.shape[1] - 1, dtype=torch.long)
    first = torch.where(stopped, is_stop.int().argmax(-1), last)
    pos = torch.arange(answer.shape[1], device=device)
    learn = pos[None, :] <= first[:, None]
    texts, rewards = [], []
    for row in range(ids.shape[0]):
        a, b = problems[row // G]
        n = first[row].item() + (0 if stopped[row] else 1)
        text = chat.tok.decode(answer[row, :n].tolist())
        texts.append(text)
        rewards.append(reward(text, a, b, stopped[row].item()))
    return {"ids": ids, "prompt": P, "learn": learn, "texts": texts,
            "rewards": torch.tensor(rewards, device=device)}


def fixed_problems(digits, n, seed=1234):
    """每种位数n道固定的题，两个加数位数相同。"""
    rng = random.Random(seed)
    return {d: [make(rng, d, d) for _ in range(n)] for d in digits}


def accuracy(model, chat, problems, max_new=24, batch=64):
    """贪婪解码，返回{位数: 正确率}。"""
    was_training = model.training
    model.eval()
    out = {}
    for d, probs in problems.items():
        right = 0.0
        for i in range(0, len(probs), batch):
            r = rollout(model, chat, probs[i:i + batch], 1, max_new,
                        temperature=0)
            right += r["rewards"].sum().item()
        out[d] = right / len(probs)
    model.train(was_training)
    return out
