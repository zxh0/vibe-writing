"""第七章 · 和模型说几句话（第九章改用带KV缓存的generate）

问题按对话模板包起来，接上“<|im_start|>assistant\\n”，让模型往下写，
写出<|im_end|>就停。基座模型从没见过这两个特殊词元，也照样能问，
正好看看SFT之前和之后有什么不同。

生成用的是模型自己的generate。第九章给它加了KV缓存，
回答一两百个词元，在CPU上一两秒。

运行：
    cd code
    # 默认几个问题，基座模型和SFT之后的模型各答一遍
    uv run python book/ch07/chat.py --model runs/pretrain/model runs/sft/model
    # 自己问；不加问题就进入多轮对话，空行退出
    uv run python book/ch07/chat.py "What is the capital of France?"
    uv run python book/ch07/chat.py --chat
"""

import argparse
import textwrap
from pathlib import Path

import torch
from tokenizers import Tokenizer

from chat_template import EOT, IM_END, ChatTemplate
from config import CODE
from export import load

QUESTIONS = [
    "What is the capital of France?",
    "Give me three tips for staying healthy.",
    "Write a short poem about the sea.",
    "What is 12 plus 7?",
    "Hi! How are you today?",
]


@torch.no_grad()
def reply(model, chat, messages, max_new=200, temperature=0.0, top_k=None):
    """接着对话往下写，写出<|im_end|>或者<|endoftext|>就停。返回回答的文字。"""
    device = next(model.parameters()).device
    ids = torch.tensor([chat.prompt(messages)], device=device)
    start = ids.shape[1]
    ids = model.generate(ids, max_new, temperature=temperature, top_k=top_k,
                         stop=(IM_END, EOT))
    if ids[0, -1].item() in (IM_END, EOT):
        ids = ids[:, :-1]
    return chat.tok.decode(ids[0, start:].tolist())


def show(name, text):
    print(f"[{name}]")
    for line in text.splitlines() or [""]:
        print(textwrap.fill(line, 76, initial_indent="  ",
                            subsequent_indent="  ") or "  ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="*", help="不给就用默认的几个问题")
    ap.add_argument("--model", type=Path, nargs="+",
                    default=[CODE / "runs" / "sft" / "model"], help="导出的模型目录")
    ap.add_argument("--chat", action="store_true", help="多轮对话，空行退出")
    ap.add_argument("--temperature", type=float, default=0.0, help="0就是贪婪")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    models = []
    for path in args.model:
        tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        name = path.parent.name if path.name == "model" else path.name
        models.append((name, load(path).eval(), ChatTemplate(tok)))
    gen = dict(max_new=args.max_new, temperature=args.temperature,
               top_k=args.top_k)

    if args.chat:
        name, model, chat = models[0]
        messages = []
        while question := input("你：").strip():
            messages.append({"role": "user", "content": question})
            torch.manual_seed(args.seed)
            answer = reply(model, chat, messages, **gen)
            messages.append({"role": "assistant", "content": answer})
            show(name, answer)
        return

    for question in args.questions or QUESTIONS:
        print(f"问：{question}")
        for name, model, chat in models:
            torch.manual_seed(args.seed)
            show(name, reply(model, chat, [{"role": "user", "content": question}],
                             **gen))
        print()


if __name__ == "__main__":
    main()
