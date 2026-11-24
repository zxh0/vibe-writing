"""第九章 · 冷启动：用很少的加法题做一次SFT

第七章的模型做加法几乎全错：它先写一大段“To solve this, ...”，写不到
答案。强化学习要靠“有时候对”才学得到东西，一组回答全错，就没有信号。
所以先教它格式：用1到3位数的题，按“a + b = c”的样子回答。

只训一小会儿：冷启动要做的是让模型“有时候做对”，不是“总能做对”，
剩下的交给GRPO。4位、5位数的题一道也不给它看，最后看强化学习能不能
把本事推到没见过的位数上。每隔几步存一个模型目录，好挑一个合适的起点。

训练和第七章一样：对话模板、只在回答上算loss。题是现出的，用不着
准备数据文件；每题只有二三十个词元，一批补齐到最长的那题就行。

运行：
    cd code
    uv run python book/ch09/cold_start.py --device cpu --steps 300
    # 在T4上（默认值）
    python book/ch09/cold_start.py --runs /content/drive/MyDrive/toyllm2/runs
"""

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from tokenizers import Tokenizer

from adamw import param_groups
from addition import make, question, solution
from chat_template import EOT, ChatTemplate
from config import CODE
from export import load as load_model
from export import save as export_model
from rollout import accuracy, fixed_problems
from schedule import lr_at


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_batch(rng, chat, B, max_digits, device):
    """B道题，两个加数各取1到max_digits位，拼成对话，补齐到一样长。"""
    seqs = []
    for _ in range(B):
        a, b = make(rng, rng.randint(1, max_digits), rng.randint(1, max_digits))
        seqs.append(chat.render([
            {"role": "user", "content": question(a, b)},
            {"role": "assistant", "content": solution(a, b)}]))
    T = max(len(ids) for ids, _ in seqs)
    rows = torch.full((B, T), EOT)
    learn = torch.zeros((B, T), dtype=torch.bool)
    for i, (ids, mask) in enumerate(seqs):
        rows[i, :len(ids)] = torch.tensor(ids)
        learn[i, :len(ids)] = torch.tensor(mask, dtype=torch.bool)
    x, y = rows[:, :-1], rows[:, 1:].masked_fill(~learn[:, 1:], -100)
    return x.to(device), y.to(device)


def show(acc, label):
    cells = "  ".join(f"{d}位 {v:6.1%}" for d, v in acc.items())
    print(f"{label}  {cells}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="cold")
    ap.add_argument("--init", type=Path,
                    default=CODE / "runs" / "sft" / "model")
    ap.add_argument("--runs", type=Path, default=CODE / "runs")
    ap.add_argument("--max-digits", type=int, default=3,
                    help="训练题的加数最多几位")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=250,
                    help="每隔几步存一个模型目录<runs>/<name>/step-N/")
    ap.add_argument("--eval-n", type=int, default=200,
                    help="每种位数用几道题量正确率")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tok_path = args.init / "tokenizer.json"
    chat = ChatTemplate(Tokenizer.from_file(str(tok_path)))
    model = load_model(args.init).to(args.device)
    opt = torch.optim.AdamW(param_groups(model, 0.1), lr=args.lr,
                            betas=(0.9, 0.95), eps=1e-8)
    fp16 = args.device == "cuda"
    amp = (torch.autocast("cuda", dtype=torch.float16) if fp16
           else nullcontext())
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    tests = fixed_problems([1, 2, 3, 4, 5], args.eval_n)

    run_dir = args.runs / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "log.jsonl").open("w")

    def record(rec):
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()

    with amp:
        acc = accuracy(model, chat, tests)
    show(acc, "第    0步")
    record({"step": 0, "acc": acc})
    t0 = time.perf_counter()
    for step in range(args.steps):
        lr = lr_at(step, args.lr, args.warmup, args.steps, args.lr / 10)
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = get_batch(rng, chat, args.batch, args.max_digits, args.device)
        with amp:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        done = step + 1
        rec = {"step": done, "loss": loss.item(), "lr": lr,
               "time": time.perf_counter() - t0}
        if done % 50 == 0 or done == args.steps:
            print(f"第{done:>5}步  loss {loss.item():.4f}  "
                  f"{time.perf_counter() - t0:.0f}秒", flush=True)
        if done % args.eval_every == 0 or done == args.steps:
            with amp:
                rec["acc"] = accuracy(model, chat, tests)
            show(rec["acc"], f"第{done:>5}步")
        record(rec)
        if done % args.save_every == 0 and done < args.steps:
            export_model(model, run_dir / f"step-{done}",
                         tok_path)

    log.close()
    out = run_dir / "model"
    export_model(model, out, tok_path)
    print(f"模型目录存到 {out}")


if __name__ == "__main__":
    main()
