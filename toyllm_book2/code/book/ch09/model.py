"""第四章 · 训练用的Llama模型（第九章加了KV缓存）

数学和第一本完全一样：RMSNorm、RoPE、GQA、SwiGLU、残差连接、权重共享。
写法改成训练用的，有四处不同：

  - 张量多了最前面的batch维：(B, T, d)，一次算B条序列。
  - 没有KV Cache。训练时整段序列一起进来，因果掩码一次算完。
  - 权重是nn.Parameter，挂在nn.Module上，优化器才找得到、改得动。
  - 权重要自己初始化，不再从文件读。

参数名和HuggingFace的一字不差（model.layers.0.self_attn.q_proj.weight……），
所以SmolLM2的权重能直接装进来核对，存盘时也不用改名。

第九章的强化学习要生成大量回答，所以给generate加上了KV缓存（KVCache）：
先把提示词整段算一遍（prefill），之后每次只算新的一个词元（decode），
前面词元的k、v从缓存里取。训练时不传cache，和原来一模一样。

运行（打印每一部分的参数量）：
    cd code
    uv run python book/ch04/model.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class RMSNorm(nn.Module):
    """除以均方根，再乘一个可学习的权重。torch 2.2还没有nn.RMSNorm，自己写。"""

    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # 先升到fp32：fp16下，512个数的平方加起来，精度和范围都不够。
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(x.dtype)


def rope_table(cfg):
    """RoPE的cos、sin表，各(最长位置数, head_dim)。和第一本的算法一样。"""
    exponent = torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim
    inv_freq = 1.0 / cfg.rope_theta ** exponent             # (32,)
    positions = torch.arange(cfg.max_position_embeddings).float()
    angles = torch.outer(positions, inv_freq)                # (1024, 32)
    angles = torch.cat((angles, angles), dim=-1)             # (1024, 64)
    return angles.cos(), angles.sin()


def rotate_half(x):
    """(a, b) → (-b, a)，第i维和第i+32维配成一对。"""
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h, kv, hd = cfg.hidden_size, cfg.num_attention_heads, \
            cfg.num_key_value_heads, cfg.head_dim
        self.n_heads, self.n_kv_heads, self.head_dim = h, kv, hd
        self.q_proj = nn.Linear(d, h * hd, bias=False)     # 512 → 512
        self.k_proj = nn.Linear(d, kv * hd, bias=False)    # 512 → 128
        self.v_proj = nn.Linear(d, kv * hd, bias=False)    # 512 → 128
        self.o_proj = nn.Linear(h * hd, d, bias=False)     # 512 → 512

    def forward(self, x, cos, sin, cache=None, i=None):
        B, T, _ = x.shape
        h, kv, hd = self.n_heads, self.n_kv_heads, self.head_dim   # 8, 2, 64
        # 拆头，把头挪到第1维：(B, T, 512) → (B, 8, T, 64)
        q = self.q_proj(x).view(B, T, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, kv, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, kv, hd).transpose(1, 2)

        # RoPE：从第start个位置开始。训练时start是0，取表的前T行。
        start = cache.length if cache is not None else 0
        cos, sin = cos[start:start + T], sin[start:start + T]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        if cache is not None:
            # 新算的k、v接在缓存后面，拿回整段的：(B, 2, start+T, 64)
            k, v = cache.update(i, k, v)

        # GQA：每个kv头连续复制4份，对齐8个q头，(B, 2, T, 64) → (B, 8, T, 64)。
        k = k.repeat_interleave(h // kv, dim=1)
        v = v.repeat_interleave(h // kv, dim=1)

        # 整段一起算时，因果掩码交给is_causal。decode时只有一个新词元，
        # 它本来就该看到前面所有的词元，不用掩码。
        assert start == 0 or T == 1, "接着缓存往下算时，一次只能一个词元"
        out = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
        out = out.transpose(1, 2).reshape(B, T, -1)        # (B, T, 512)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU：down( silu(gate(x)) * up(x) )。"""

    def __init__(self, cfg):
        super().__init__()
        d, f = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(d, f, bias=False)       # 512 → 1408
        self.up_proj = nn.Linear(d, f, bias=False)         # 512 → 1408
        self.down_proj = nn.Linear(f, d, bias=False)       # 1408 → 512

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None, i=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, i)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class KVCache:
    """每层一份k、v，形状(B, 2, 已经算过的词元数, 64)。只存2个kv头，
    复制成8份是取出来以后的事，缓存小4倍。"""

    def __init__(self, n_layers):
        self.k = [None] * n_layers
        self.v = [None] * n_layers
        self.length = 0

    def update(self, i, k, v):
        if self.k[i] is not None:
            k = torch.cat([self.k[i], k], dim=2)
            v = torch.cat([self.v[i], v], dim=2)
        self.k[i], self.v[i] = k, v
        if i == len(self.k) - 1:              # 最后一层存完，长度才往前走
            self.length = k.shape[2]
        return k, v


class Backbone(nn.Module):
    """词嵌入、N层、最后一个RMSNorm。在HF的命名里，它就是那个`model.`前缀。"""

    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class LLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.config = cfg
        self.model = Backbone(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            # 权重共享：两个名字指向同一个Parameter，梯度会两边加起来。
            self.lm_head.weight = self.model.embed_tokens.weight
        # cos、sin表不是参数，不用学，也不存盘（persistent=False），
        # 但要跟着模型搬到GPU上，所以登记成buffer。
        cos, sin = rope_table(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        """初始化。推理时权重从文件读，训练时得自己给一个起点。

        - 线性层和词嵌入：均值0、标准差0.02的正态分布；
        - 每层最后写回残差的两个投影（o_proj、down_proj）：标准差再除以√(2×层数)，
          GPT-2的做法。每层往残差里加两次，层数多了，加起来的方差不至于越滚越大；
        - RMSNorm的权重：全是1，一开始什么都不缩放。
        """
        std = self.config.initializer_range
        small = std / math.sqrt(2 * self.config.num_hidden_layers)
        for name, p in self.named_parameters():
            if name.endswith("norm.weight"):
                nn.init.ones_(p)
            elif name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=small)
            else:
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, ids, targets=None, cache=None):
        """ids (B, T) → logits (B, T, 词表大小)；给了targets就顺便算loss。

        targets和ids一样形状，第t个位置放的是第t+1个词元，
        也就是每个位置“应该预测出来的那一个”。
        """
        T = ids.shape[1] + (cache.length if cache is not None else 0)
        assert T <= self.config.max_position_embeddings, "序列比RoPE表还长"
        x = self.model.embed_tokens(ids)
        for i, layer in enumerate(self.model.layers):
            x = layer(x, self.cos, self.sin, cache, i)
        logits = self.lm_head(self.model.norm(x))
        if targets is None:
            return logits, None
        # 交叉熵：B×T个位置各算一次-log p(正确答案)，取平均。在fp32里算。
        loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, n, temperature=1.0, top_k=None, stop=()):
        """接着ids (B, T)往下写最多n个词元，用KV缓存。

        先把ids整段算一遍，之后每步只算刚抽出的那一个词元。B行一起写，
        所以每行的提示词要一样长。每一行都写出过stop里的词元，就提前停下；
        先写完的行后面接着写，多出来的词元由调用的人截掉。
        """
        cache = KVCache(self.config.num_hidden_layers)
        done = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
        for step in range(n):
            new = ids if step == 0 else ids[:, -1:]   # prefill，然后decode
            logits = self(new, cache=cache)[0][:, -1, :].float()
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, top_k).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            ids = torch.cat([ids, nxt], dim=1)
            for s in stop:
                done |= nxt[:, 0] == s
            full = ids.shape[1] >= self.config.max_position_embeddings
            if done.all() or full:
                break
        return ids


def count(module):
    return sum(p.numel() for p in module.parameters())


def main():
    cfg = Config()
    model = LLM(cfg)
    layer = model.model.layers[0]
    rows = [
        ("词嵌入（和输出层共用）", count(model.model.embed_tokens)),
        ("每层：注意力 q、k、v、o", count(layer.self_attn)),
        ("每层：前馈 gate、up、down", count(layer.mlp)),
        ("每层：两个RMSNorm", count(layer.input_layernorm) * 2),
        ("每层合计", count(layer)),
        (f"{cfg.num_hidden_layers}层合计", count(model.model.layers)),
        ("最后的RMSNorm", count(model.model.norm)),
        ("全部", count(model)),
    ]
    for name, n in rows:
        # 汉字占两列，按显示宽度补齐
        pad = 28 - sum(2 if ord(c) > 0x2e80 else 1 for c in name)
        print(f"{name}{' ' * pad}{n:>12,}")

    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(ids[:, :-1], ids[:, 1:])
    print(f"\n输入 {tuple(ids[:, :-1].shape)} → logits {tuple(logits.shape)}，loss {loss.item():.3f}")


if __name__ == "__main__":
    main()
