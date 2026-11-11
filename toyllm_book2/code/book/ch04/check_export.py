"""第四章 · 导出的目录，transformers认不认

用transformers加载export.py存下的目录，和我们自己读回来的模型比：
同一段输入的logits，以及贪婪生成的20个词元。

运行（先跑export.py）：
    cd code
    uv run python book/ch04/check_export.py
"""

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

from config import MODELS
from export import load

MODEL_DIR = MODELS / "toyllm2-init"


def main():
    ours = load(MODEL_DIR).eval()
    ref = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                               attn_implementation="sdpa").eval()
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    ids = torch.tensor([tok.encode("Once upon a time").ids])

    with torch.no_grad():
        a, b = ours(ids)[0], ref(ids).logits
    err = ((a - b).abs().max() / b.abs().max()).item()
    print(f"logits       相对误差 {err:.1e}")

    mine = ours.generate(ids, 20, temperature=0)[0].tolist()
    theirs = ref.generate(ids, max_new_tokens=20, do_sample=False,
                          eos_token_id=None, pad_token_id=0)[0].tolist()
    print(f"贪婪生成     一模一样：{mine == theirs}")


if __name__ == "__main__":
    main()
