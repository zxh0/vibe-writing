"""第一章 · 五分钟训一个字符级的小模型

一个文件，不依赖本书的其他代码。做的事情就是全书的缩影：

    1. 读入Tiny Shakespeare，每个字符编一个号，整篇文本变成一串整数
    2. 留出最后10%当验证集
    3. 随机初始化一个小Transformer
    4. 反复随机截取片段，预测下一个字符，按预测的好坏调整参数
    5. 每隔一段时间看一眼验证集上的loss，再让模型生成一小段

所有细节都留到后面各章：数据（第二章）、分词（第三章）、模型（第四章）、
训练循环（第五章）。这里只求短，能跑，看得到loss在降。

运行：
    cd code
    uv run python book/ch01/char_gpt.py
    uv run python book/ch01/char_gpt.py --steps 200      # 先试跑一下
"""

import argparse
import math
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# code/book/ch01/char_gpt.py → 上三层就是 code/
DATA = Path(__file__).resolve().parent.parent.parent / "data"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# 模型和训练的超参数。小到能在笔记本电脑的CPU上几分钟训完。
BLOCK = 64        # 一个片段多长，也就是模型最多往前看多少个字符
BATCH = 32        # 一步同时训多少个片段
D_MODEL = 128     # 每个字符用多少个数表示
N_LAYER = 4       # 几层Transformer块
N_HEAD = 4        # 每层几个注意力头
LR = 1e-3         # 学习率：每一步参数调多大


def load_text():
    """下载Tiny Shakespeare，已经下过就直接读。"""
    path = DATA / "tinyshakespeare.txt"
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return path.read_text(encoding="utf-8")


class Block(nn.Module):
    """一个Transformer块：注意力加前馈网络，各自包一层残差。"""

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL))

    def forward(self, x):                                          # (B, T, d)
        B, T, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).split(d, dim=-1)
        q, k, v = (t.view(B, T, N_HEAD, d // N_HEAD).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, h, T, d/h)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.ffn(self.norm2(x))


class CharGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, D_MODEL)   # 字符 → 向量
        self.pos = nn.Embedding(BLOCK, D_MODEL)        # 位置 → 向量
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYER))
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size)     # 向量 → 每个字符的分数

    def forward(self, ids):                                        # (B, T)
        T = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(T))              # (B, T, d)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))                             # (B, T, vocab)

    @torch.no_grad()
    def generate(self, ids, n, gen):
        """往后续写n个字符：每次取最后一个位置的分数，按概率抽一个。"""
        for _ in range(n):
            logits = self(ids[:, -BLOCK:])[:, -1]
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=gen)
            ids = torch.cat([ids, nxt], dim=1)
        return ids


def get_batch(data, gen):
    """随机挑BATCH个起点，各截一段。y是x往后错一位：每个位置的答案就是下一个字符。"""
    starts = torch.randint(len(data) - BLOCK - 1, (BATCH,), generator=gen)
    x = torch.stack([data[s:s + BLOCK] for s in starts])
    y = torch.stack([data[s + 1:s + BLOCK + 1] for s in starts])
    return x, y


@torch.no_grad()
def eval_loss(model, data, gen, n_batches=20):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, gen)
        logits = model(x)
        losses.append(F.cross_entropy(logits.flatten(0, 1), y.flatten()).item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--every", type=int, default=300, help="每隔多少步评估、生成一次")
    args = ap.parse_args()
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    # 1. 文本 → 整数。词表就是文本里出现过的所有字符。
    text = load_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text])
    decode = lambda ids: "".join(chars[i] for i in ids)

    # 2. 最后10%当验证集，训练时一次都不看。
    n = int(0.9 * len(data))
    train, val = data[:n], data[n:]

    # 3. 随机初始化的模型。
    model = CharGPT(len(chars))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"字符数     {len(text):,}")
    print(f"字符种类   {len(chars)}，随机猜的loss是ln({len(chars)}) = {math.log(len(chars)):.3f}")
    print(f"训练/验证  {len(train):,} / {len(val):,}")
    print(f"参数量     {n_params:,}")
    print()

    def report(step):
        prompt = torch.tensor([[stoi["\n"]]])
        sample = decode(model.generate(prompt, 200, gen)[0].tolist())
        print(f"--- 第{step}步  训练loss {eval_loss(model, train, gen):.3f}"
              f"  验证loss {eval_loss(model, val, gen):.3f}"
              f"  用时{time.perf_counter() - start:.0f}秒")
        print(sample.strip())
        print()

    # 4. 训练：前向、算loss、反向传播、更新参数，重复。
    start = time.perf_counter()
    report(0)
    for step in range(1, args.steps + 1):
        x, y = get_batch(train, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.every == 0 or step == args.steps:
            report(step)


if __name__ == "__main__":
    main()
