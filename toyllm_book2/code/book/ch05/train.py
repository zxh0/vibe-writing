"""第五章 · 一个最小的训练循环

每一步做五件事：

    1. 按进度算出这一步的学习率（warmup加cosine）
    2. 从train.bin里随机截B段，每段T+1个词元，前T个是输入，后T个是答案
    3. 前向算loss，反向算梯度；梯度累积时，这一步重复accum次
    4. 梯度裁剪：总范数超过clip，就整体缩小到clip
    5. 优化器按梯度更新参数

每一步往runs/<名字>/log.jsonl写一行，隔一段时间在验证集上算一次loss，
最后存一个checkpoint，用--resume可以接着训。

这一章只在Mac上跑几百步，用fp32。搬到Colab的T4上、换成fp16，是第六章的事。

运行：
    cd code
    uv run python book/ch05/train.py --name base
    uv run python book/ch05/train.py --name lr1e-2 --lr 1e-2
    uv run python book/ch05/train.py --name cut --stop-at 150   # 训到一半停下
    uv run python book/ch05/train.py --name cut --resume        # 接着训完
"""

import argparse
import json
import math
import textwrap
import time

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.nn.utils import clip_grad_norm_

from adamw import AdamW, param_groups
from config import CODE, DATA, Config
from model import LLM
from schedule import lr_at

RUNS = CODE / "runs"


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch(data, B, T, gen, device):
    """随机截B段，每段T+1个词元。错开一位，就是输入和答案。"""
    starts = torch.randint(0, len(data) - T - 1, (B,), generator=gen).tolist()
    chunk = np.stack([data[s:s + T + 1] for s in starts]).astype(np.int64)
    chunk = torch.from_numpy(chunk).to(device)
    return chunk[:, :-1], chunk[:, 1:]


@torch.no_grad()
def evaluate(model, data, B, T, n_batches, device):
    """验证集开头的n_batches批，每次都是同样的数据，前后才好比。"""
    model.eval()
    total = 0.0
    for i in range(n_batches):
        start = i * B * (T + 1)
        chunk = data[start:start + B * (T + 1)].astype(np.int64)
        chunk = torch.from_numpy(chunk).view(B, T + 1).to(device)
        total += model(chunk[:, :-1], chunk[:, 1:])[1].item()
    model.train()
    return total / n_batches


def save_checkpoint(path, model, opt, step):
    """接着训需要的最少的东西：参数、优化器的m和v、训到了第几步。"""
    torch.save({"model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step}, path)


def load_checkpoint(path, model, opt, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]


def make_optimizer(model, args):
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr)
    return AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base", help="记录写到runs/<name>/")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8, help="每次前向的序列条数B")
    ap.add_argument("--seq", type=int, default=256, help="每条序列的词元数T")
    ap.add_argument("--accum", type=int, default=1, help="梯度累积：几次前向更新一次")
    ap.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    ap.add_argument("--lr", type=float, default=1e-3, help="最大学习率")
    ap.add_argument("--min-lr", type=float, default=None, help="默认是lr的十分之一")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0, help="0表示不裁剪")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-batches", type=int, default=10)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-at", type=int, default=None,
                    help="训到第几步就存盘退出，模拟训练被打断")
    args = ap.parse_args()
    if args.min_lr is None:
        args.min_lr = args.lr / 10
    return args


def main():
    args = parse_args()
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    train = np.memmap(DATA / "tokens" / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(DATA / "tokens" / "val.bin", dtype=np.uint16, mode="r")

    torch.manual_seed(args.seed)
    model = LLM(Config()).to(args.device)
    opt = make_optimizer(model, args)
    start = 0
    if args.resume:
        start = load_checkpoint(out / "ckpt.pt", model, opt, args.device)
        print(f"从第{start}步接着训")
    # 取数据用的随机数，种子带上起始步数，接着训时不会把前面的数据重抽一遍。
    gen = torch.Generator().manual_seed(args.seed * 100003 + start)
    log = open(out / "log.jsonl", "a" if args.resume else "w")
    tokens_per_step = args.batch * args.seq * args.accum
    print(f"设备 {args.device}，每步 {args.batch}×{args.seq}×{args.accum} = "
          f"{tokens_per_step:,} 个词元，共{args.steps}步")

    if start == 0:
        val_loss = evaluate(model, val, args.batch, args.seq, args.eval_batches, args.device)
        log.write(json.dumps({"step": 0, "val_loss": val_loss}) + "\n")
        print(f"第{0:>4}步  验证loss {val_loss:.3f}")

    t0 = time.perf_counter()
    done = start
    for step in range(start, args.steps):
        # 1. 学习率
        lr = lr_at(step, args.lr, args.warmup, args.steps, args.min_lr)
        for group in opt.param_groups:
            group["lr"] = lr

        # 2、3. 取数据，前向，反向。累积accum次，每次的loss先除以accum，
        # 梯度加起来，正好是accum倍大的batch的平均梯度。
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(args.accum):
            x, y = get_batch(train, args.batch, args.seq, gen, args.device)
            _, loss = model(x, y)
            (loss / args.accum).backward()
            loss_sum += loss.item()
        loss = loss_sum / args.accum

        # 4. 梯度裁剪。返回的是裁剪之前的总范数；
        # 不裁剪时上限给无穷大，只为算出范数。
        max_norm = args.clip if args.clip > 0 else math.inf
        norm = clip_grad_norm_(model.parameters(), max_norm).item()

        # 5. 更新参数
        opt.step()

        done = step + 1
        rec = {"step": done, "loss": loss, "lr": lr, "grad_norm": norm,
               "time": time.perf_counter() - t0}
        if done % args.eval_every == 0 or done == args.steps:
            rec["val_loss"] = evaluate(model, val, args.batch, args.seq,
                                       args.eval_batches, args.device)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if "val_loss" in rec:
            print(f"第{done:>4}步  loss {loss:.3f}  验证loss {rec['val_loss']:.3f}  "
                  f"梯度范数 {norm:.2f}  学习率 {lr:.1e}", flush=True)
        if not math.isfinite(loss):
            print("loss不是有限的数了，训练发散，停下")
            break
        if done == args.stop_at:
            print(f"在第{done}步停下")
            break

    elapsed = time.perf_counter() - t0
    speed = (done - start) * tokens_per_step / elapsed
    print(f"用时{elapsed:.0f}秒，平均每秒{speed:,.0f}个词元（含验证）")
    save_checkpoint(out / "ckpt.pt", model, opt, done)
    print(f"checkpoint存到 {(out / 'ckpt.pt').relative_to(CODE)}")

    tok = Tokenizer.from_file(str(DATA / "tokenizer" / "tokenizer.json"))
    prompt = "Once upon a time"
    ids = torch.tensor([tok.encode(prompt).ids], device=args.device)
    torch.manual_seed(args.seed)
    text = tok.decode(model.generate(ids, 30, temperature=0.8, top_k=50)[0].tolist())
    print("生成（折行是为了排版加的）：")
    print(textwrap.indent(textwrap.fill(repr(text), 74), "  "))


if __name__ == "__main__":
    main()
