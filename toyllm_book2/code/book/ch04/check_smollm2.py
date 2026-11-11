"""第四章 · 训练之前，先和transformers对答案

把SmolLM2-135M的权重装进我们自己的模型，再用transformers加载同一份权重，
喂同一批数据，比三样东西：

  1. logits：前向算得对不对；
  2. loss：交叉熵和错位（第t个位置预测第t+1个）对不对；
  3. 每个参数的梯度：反向传播对不对。训练靠的就是梯度，前向对了、
     梯度不对（比如权重共享没接上），模型照样训不好，还不报错。

全部在CPU上、用fp32算。transformers只在这个脚本里出现，是dev依赖。

运行（先下载SmolLM2-135M，见code/download_model.sh）：
    cd code
    uv run python book/ch04/check_smollm2.py
"""

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

from config import MODELS, Config
from model import LLM

SMOLLM2 = MODELS / "SmolLM2-135M"
TEXT = [
    "The quick brown fox jumps over the lazy dog. It was a sunny day, and the fox "
    "was looking for something to eat. Photosynthesis converts sunlight into energy.",
    "Once upon a time, there was a little girl who lived in a village near the forest. "
    "Whenever she went out, the little girl wore a red riding cloak.",
]


def rel_err(a, b):
    """相对误差：差的最大值，除以参考值的最大值。"""
    return ((a - b).abs().max() / b.abs().max()).item()


def main():
    torch.manual_seed(0)
    cfg = Config.load(SMOLLM2)
    print(f"SmolLM2-135M：d={cfg.hidden_size}，{cfg.num_hidden_layers}层，"
          f"{cfg.num_attention_heads}头（KV {cfg.num_key_value_heads}头），"
          f"FFN {cfg.intermediate_size}，词表{cfg.vocab_size:,}")

    # 我们的模型：按配置建好，再把文件里的权重原样装进去。参数名一字不差，
    # 唯一缺的是lm_head：权重共享，文件里不存，它就是embed_tokens。
    ours = LLM(cfg)
    missing, unexpected = ours.load_state_dict(load_file(SMOLLM2 / "model.safetensors"),
                                               strict=False)
    assert missing == ["lm_head.weight"] and unexpected == [], (missing, unexpected)
    ref = AutoModelForCausalLM.from_pretrained(SMOLLM2, dtype=torch.float32,
                                               attn_implementation="sdpa")

    # 两条一样长的序列，凑成一个batch。
    tok = Tokenizer.from_file(str(SMOLLM2 / "tokenizer.json"))
    rows = [tok.encode(t).ids for t in TEXT]
    T = min(len(r) for r in rows)
    ids = torch.tensor([r[:T] for r in rows])
    print(f"输入         {tuple(ids.shape)}，两条序列各{T}个词元")

    # 我们：输入是前T-1个，答案是后T-1个，错开一位。
    logits, loss = ours(ids[:, :-1], ids[:, 1:])
    loss.backward()
    # transformers：把整段当labels传进去，它在内部错位。
    out = ref(ids, labels=ids)
    out.loss.backward()

    print(f"logits       相对误差 {rel_err(logits, out.logits[:, :-1]):.1e}")
    print(f"loss         我们 {loss.item():.6f}，transformers {out.loss.item():.6f}")

    ref_params = dict(ref.named_parameters())
    worst, n = (0.0, ""), 0
    for name, p in ours.named_parameters():
        e = rel_err(p.grad, ref_params[name].grad)
        worst = max(worst, (e, name))
        n += 1
    print(f"梯度         {n}个参数，最大相对误差 {worst[0]:.1e}")
    print(f"             出现在{worst[1]}")


if __name__ == "__main__":
    main()
