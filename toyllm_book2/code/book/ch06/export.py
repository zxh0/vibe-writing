"""第四章 · 存成第一本能读的模型目录

和SmolLM2的目录长得一样，四个文件：

    config.json             模型的形状，Config.save写的
    generation_config.json  起始符、结束符。第一本的引擎直接打开它，不能缺
    model.safetensors       权重，fp32。权重共享时不存lm_head
    tokenizer.json          第三章训练的分词器，原样复制

读回来用load：按config.json建模型，再把权重装进去。

运行（把一个随机初始化的模型存到code/models/toyllm2-init/，再读回来生成几个词元）：
    cd code
    uv run python book/ch04/export.py
"""

import json
import shutil
import textwrap

import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

from config import DATA, MODELS, Config
from model import LLM

TOKENIZER = DATA / "tokenizer" / "tokenizer.json"


def save(model, out_dir, tokenizer=TOKENIZER):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = model.config
    cfg.save(out_dir)
    gen = {"bos_token_id": cfg.bos_token_id, "eos_token_id": cfg.eos_token_id}
    (out_dir / "generation_config.json").write_text(json.dumps(gen, indent=2) + "\n")

    state = model.state_dict()
    if cfg.tie_word_embeddings:
        # lm_head.weight和model.embed_tokens.weight是同一块内存。safetensors不肯
        # 存两个共享内存的张量，SmolLM2的文件里也只有后者，所以去掉前者。
        del state["lm_head.weight"]
    # 不管训练时用什么精度，存盘都是fp32；contiguous是safetensors的要求。
    state = {k: v.detach().float().contiguous().cpu() for k, v in state.items()}
    save_file(state, out_dir / "model.safetensors", metadata={"format": "pt"})
    shutil.copy(tokenizer, out_dir / "tokenizer.json")


def load(model_dir):
    cfg = Config.load(model_dir)
    model = LLM(cfg)
    missing, unexpected = model.load_state_dict(load_file(model_dir / "model.safetensors"),
                                                strict=False)
    allowed = ["lm_head.weight"] if cfg.tie_word_embeddings else []
    assert missing == allowed and unexpected == [], (missing, unexpected)
    return model


def main():
    torch.manual_seed(0)
    out = MODELS / "toyllm2-init"
    save(LLM(Config()), out)
    for f in sorted(out.iterdir()):
        print(f"{f.name:<24}{f.stat().st_size:>12,} 字节")

    model = load(out).eval()
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    ids = torch.tensor([tok.encode("Once upon a time").ids])
    for name, temperature in [("贪婪", 0), ("随机采样", 1.0)]:
        torch.manual_seed(0)
        out_ids = model.generate(ids, 20, temperature=temperature)[0].tolist()
        print(f"\n{name}生成20个词元（折行是为了排版加的）：")
        print(textwrap.indent(textwrap.fill(repr(tok.decode(out_ids)), 74), "  "))


if __name__ == "__main__":
    main()
