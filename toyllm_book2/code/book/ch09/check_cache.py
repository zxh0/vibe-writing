"""第九章 · 核对KV缓存

两件事：

    1. logits：一段对话整段算一遍，和“先算提示词，再一个一个词元接着算”
       比，每个位置的logits应该一样（相对误差在fp32的舍入误差以内）。
    2. 生成：贪婪解码，带缓存的generate和每步整段重算的写法，写出的
       词元应该一个不差；再比一比速度。

运行（在CPU上，十几秒）：
    cd code
    uv run python book/ch09/check_cache.py
"""

import time

import torch
from tokenizers import Tokenizer

from chat_template import EOT, IM_END, ChatTemplate
from config import CODE
from export import load
from model import KVCache


def no_cache(model, ids, n):
    """不用缓存的贪婪解码：每一步都把整段重算一遍。"""
    for _ in range(n):
        nxt = model(ids)[0][:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return ids


@torch.no_grad()
def main():
    path = CODE / "runs" / "sft" / "model"
    model = load(path).eval()
    chat = ChatTemplate(Tokenizer.from_file(str(path / "tokenizer.json")))
    messages = [{"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital is Paris."}]
    ids = torch.tensor([chat.render(messages)[0]])
    T, P = ids.shape[1], 12

    full = model(ids)[0]
    cache = KVCache(model.config.num_hidden_layers)
    parts = [model(ids[:, :P], cache=cache)[0]]
    for t in range(P, T):
        parts.append(model(ids[:, t:t + 1], cache=cache)[0])
    step = torch.cat(parts, dim=1)
    err = ((step - full).norm() / full.norm()).item()
    print(f"1. {T}个词元：整段算，和先算{P}个、再一个一个算，"
          f"logits的相对误差 {err:.1e}")

    hi = [{"role": "user", "content": "Hi! How are you today?"}]
    prompt = torch.tensor([chat.prompt(hi)] * 4)
    n = 100
    t0 = time.perf_counter()
    a = no_cache(model, prompt, n)
    t1 = time.perf_counter()
    b = model.generate(prompt, n, temperature=0)
    t2 = time.perf_counter()
    print(f"2. 贪婪解码{prompt.shape[0]}行、各{n}个词元：写出的词元"
          f"{'完全相同' if torch.equal(a, b) else '不同'}")
    print(f"   每步重算 {t1 - t0:.2f}秒   KV缓存 {t2 - t1:.2f}秒")
    c = model.generate(prompt, n, temperature=0, stop=(IM_END, EOT))
    same = torch.equal(c, b[:, :c.shape[1]])
    print(f"   遇到<|im_end|>就停：写了{c.shape[1] - prompt.shape[1]}个词元，"
          f"和前面的开头{'一样' if same else '不一样'}")


if __name__ == "__main__":
    main()
