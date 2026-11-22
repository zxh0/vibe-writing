"""第七章 · 监督微调（SFT）：第六章的train.py改几处

训练循环、混合精度、断点续训和第六章一模一样，改的只有四处：

    1. 起点：不是随机初始化，而是读第六章导出的基座模型（--init）。
    2. 数据：prepare_sft.py装好的定长行，每次随机取B行；不在回答里的
       位置，答案换成-100，cross_entropy见到-100就跳过，不算loss。
    3. 定期生成：不再续写开头，而是按对话模板问几个问题。
    4. 导出：结束符改成<|im_end|>，第一本的引擎读generation_config.json，
       抽到它就停。

学习率比预训练小，模型已经会说话了，只是学一种新的格式，
步子太大会把预训练学到的东西冲掉。

运行：
    cd code
    # 在Mac上，用很小的batch试一试（fp32）
    uv run python book/ch07/sft.py --name sft-try --batch 2 --accum 1 \\
        --steps 20 --warmup 2 --eval-every 10 --eval-batches 2 \\
        --sample-every 20 --ckpt-every 10
    # 在T4上（默认值），断了以后再跑一遍，--resume会接着训
    python book/ch07/sft.py --init .../runs/pretrain/model \\
        --runs /content/drive/MyDrive/toyllm2/runs --resume --export
"""

import argparse
import json
import math
import os
import textwrap
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.nn.utils import clip_grad_norm_

from adamw import param_groups
from chat import reply
from chat_template import IM_END, ChatTemplate
from config import CODE, DATA
from export import load as load_model
from export import save as export_model
from schedule import lr_at

QUESTIONS = ["What is the capital of France?",
             "Give me three tips for staying healthy.",
             "Hi! How are you today?"]
ROW = 1025          # 每行1024个输入，错开一位是答案


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_scaler(enabled, init_scale):
    """损失缩放器。torch 2.3起叫torch.amp.GradScaler，2.2只有torch.cuda.amp里的。"""
    try:
        return torch.amp.GradScaler("cuda", init_scale=init_scale, enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.GradScaler(init_scale=init_scale, enabled=enabled)


def read_rows(data_dir, split):
    """prepare_sft.py写的词元和掩码，每行1025个。"""
    rows = np.fromfile(data_dir / f"{split}.bin", dtype=np.uint16)
    mask = np.fromfile(data_dir / f"{split}_mask.bin", dtype=np.uint8)
    return rows.reshape(-1, ROW), mask.reshape(-1, ROW)


def to_batch(rows, mask, device):
    """错开一位是输入和答案；掩码是0的位置，答案换成-100，不算loss。"""
    rows = torch.from_numpy(rows.astype(np.int64))
    learn = torch.from_numpy(mask[:, 1:].astype(bool))
    x, y = rows[:, :-1], rows[:, 1:].masked_fill(~learn, -100)
    return x.to(device), y.to(device)


def get_batch(data, B, gen, device):
    """随机取B行。"""
    rows, mask = data
    idx = torch.randint(0, len(rows), (B,), generator=gen).numpy()
    return to_batch(rows[idx], mask[idx], device)


@torch.no_grad()
def evaluate(model, data, B, n_batches, device, amp):
    """验证集开头的n_batches批，每次都是同样的数据，前后才好比。
    loss只在回答的词元上取平均，和训练时一样。"""
    model.eval()
    rows, mask = data
    total = 0.0
    for i in range(n_batches):
        part = slice(i * B, (i + 1) * B)
        x, y = to_batch(rows[part], mask[part], device)
        with amp:
            total += model(x, y)[1].item()
    model.train()
    return total / n_batches


def sample(model, chat, device, seed, n=60):
    """按对话模板问几个问题，每个最多答n个词元。随机数在一个分叉出来的
    状态里用，用完还原，所以生成不会改变训练本身；种子固定，不同阶段的
    样例才好比。"""
    model.eval()
    out = []
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for q in QUESTIONS:
            a = reply(model, chat, [{"role": "user", "content": q}], n,
                      temperature=0.8, top_k=50)
            out.append(f"{q} → {a}")
    model.train()
    return out


def write_log(path, records):
    """把日志整个重写一遍。和checkpoint一样，先写临时文件再改名，
    读到的log.jsonl总是完整的。"""
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                           for r in records))
    os.replace(tmp, path)


def save_checkpoint(path, model, opt, scaler, gen, step, elapsed, records,
                    args):
    """接着训需要的全部东西，和第六章一样。先写到临时文件，写完再改名。"""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "elapsed": elapsed,
        "data_rng": gen.get_state(),
        "cpu_rng": torch.get_rng_state(),
        "log": records,
        "args": {k: str(v) if isinstance(v, Path) else v
                 for k, v in vars(args).items()},
    }
    if torch.cuda.is_available():
        ckpt["cuda_rng"] = torch.cuda.get_rng_state_all()
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, opt, scaler, gen, args):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    gen.set_state(ckpt["data_rng"])
    torch.set_rng_state(ckpt["cpu_rng"])
    if "cuda_rng" in ckpt and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    same = ("batch", "accum", "steps", "lr", "min_lr", "warmup",
            "weight_decay", "clip", "dtype", "seed")
    for k in same:
        if ckpt["args"].get(k) != getattr(args, k):
            print(f"注意：{k}原来是{ckpt['args'].get(k)}，现在是{getattr(args, k)}")
    return ckpt["step"], ckpt["elapsed"], ckpt["log"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sft", help="记录写到<runs>/<name>/")
    ap.add_argument("--init", type=Path, default=CODE / "runs" / "pretrain" / "model",
                    help="第六章导出的基座模型目录")
    ap.add_argument("--data", type=Path, default=DATA / "sft",
                    help="prepare_sft.py的输出目录")
    ap.add_argument("--runs", type=Path, default=CODE / "runs")
    ap.add_argument("--epochs", type=float, default=1.0,
                    help="把训练集过几遍，用来算默认的步数")
    ap.add_argument("--steps", type=int, default=None, help="给了就不按epochs算")
    ap.add_argument("--batch", type=int, default=32, help="每次前向的行数B")
    ap.add_argument("--accum", type=int, default=2, help="梯度累积：几次前向更新一次")
    ap.add_argument("--lr", type=float, default=3e-4, help="最大学习率")
    ap.add_argument("--min-lr", type=float, default=None, help="默认是lr的十分之一")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0, help="0表示不裁剪")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default=None,
                    help="默认CUDA上fp16，其他设备fp32")
    ap.add_argument("--init-scale", type=float, default=2.0 ** 16)
    ap.add_argument("--compile", action="store_true", help="用torch.compile，要torch 2.x新版")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--sample-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="有checkpoint就接着训，没有就从头开始")
    ap.add_argument("--export", action="store_true",
                    help="训完存成模型目录：<runs>/<name>/model/")
    args = ap.parse_args()
    if args.min_lr is None:
        args.min_lr = args.lr / 10
    if args.dtype is None:
        args.dtype = "fp16" if args.device == "cuda" else "fp32"
    assert args.dtype == "fp32" or args.device == "cuda", "fp16只在CUDA上用"
    return args


def main():
    args = parse_args()
    out = args.runs / args.name
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path, log_path = out / "ckpt.pt", out / "log.jsonl"
    train = read_rows(args.data, "train")
    val = read_rows(args.data, "val")
    tok_path = args.init / "tokenizer.json"
    chat = ChatTemplate(Tokenizer.from_file(str(tok_path)))
    if args.steps is None:
        args.steps = math.ceil(args.epochs * len(train[0]) / (args.batch * args.accum))

    # 起点是基座模型。种子只管取数据的顺序和dropout之类，这里没有随机初始化。
    torch.manual_seed(args.seed)
    model = load_model(args.init).to(args.device)
    fused = args.device == "cuda"
    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr,
                            betas=(0.9, 0.95), eps=1e-8, fused=fused)
    fp16 = args.dtype == "fp16"
    scaler = make_scaler(fp16, args.init_scale)
    amp = torch.autocast("cuda", dtype=torch.float16) if fp16 else nullcontext()
    gen = torch.Generator().manual_seed(args.seed)
    fwd = torch.compile(model) if args.compile else model

    start, elapsed, records = 0, 0.0, []
    if args.resume and ckpt_path.exists():
        start, elapsed, records = load_checkpoint(ckpt_path, model, opt, scaler,
                                                  gen, args)
        print(f"从第{start}步接着训，之前已经训了{elapsed / 60:.1f}分钟")

    n_params = sum(p.numel() for p in model.parameters())
    rows_per_step = args.batch * args.accum
    tokens_per_step = rows_per_step * (ROW - 1)
    learn = train[1][:, 1:].mean()
    print(f"设备 {args.device}，{args.dtype}，参数{n_params:,}个，起点 {args.init}")
    print(f"训练集{len(train[0]):,}行，其中{learn:.1%}的位置算loss；"
          f"每步{rows_per_step}行，共{args.steps:,}步，"
          f"约{args.steps * rows_per_step / len(train[0]):.2f}遍")

    if start == 0:
        val_loss = evaluate(model, val, args.batch, args.eval_batches,
                            args.device, amp)
        records.append({"step": 0, "val_loss": val_loss,
                        "samples": sample(model, chat, args.device, args.seed)})
        print(f"第{0:>5}步  验证loss {val_loss:.3f}")
        for text in records[-1]["samples"]:
            print(textwrap.indent(textwrap.fill(repr(text), 74), "    "))

    # 吞吐量只算训练本身的时间，不含验证、生成和存盘。
    window_time, window_tokens, skipped = 0.0, 0, 0
    for step in range(start, args.steps):
        t_step = time.perf_counter()
        lr = lr_at(step, args.lr, args.warmup, args.steps, args.min_lr)
        for group in opt.param_groups:
            group["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(args.accum):
            x, y = get_batch(train, args.batch, gen, args.device)
            with amp:
                _, loss = fwd(x, y)
            scaler.scale(loss / args.accum).backward()
            loss_sum += loss.item()
        loss = loss_sum / args.accum

        scaler.unscale_(opt)
        max_norm = args.clip if args.clip > 0 else math.inf
        norm = clip_grad_norm_(model.parameters(), max_norm).item()
        scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skip = scaler.get_scale() < scale
        skipped += skip

        done = step + 1
        took = time.perf_counter() - t_step
        elapsed += took
        window_time += took
        window_tokens += tokens_per_step
        rec = {"step": done, "loss": loss, "lr": lr, "grad_norm": norm,
               "time": elapsed}
        if fp16:
            rec["scale"] = scale
        if skip:
            rec["skipped"] = True
            print(f"第{done:>5}步  梯度溢出，跳过这一步，放大倍数{scale:g} → "
                  f"{scaler.get_scale():g}")
        if done % args.log_every == 0:
            speed = window_tokens / window_time
            window_time, window_tokens = 0.0, 0
            rec["tok_per_s"] = speed
            if args.device == "cuda":
                rec["mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
            print(f"第{done:>5}步  loss {loss:.3f}  梯度范数 {norm:5.2f}  "
                  f"学习率 {lr:.1e}  每秒{speed:,.0f}个词元", flush=True)
        if done % args.eval_every == 0 or done == args.steps:
            rec["val_loss"] = evaluate(model, val, args.batch, args.eval_batches,
                                       args.device, amp)
            print(f"第{done:>5}步  验证loss {rec['val_loss']:.3f}  "
                  f"用时{elapsed / 60:.1f}分钟")
        if done % args.sample_every == 0 or done == args.steps:
            rec["samples"] = sample(model, chat, args.device, args.seed)
            for text in rec["samples"]:
                print(textwrap.indent(textwrap.fill(repr(text), 74), "    "))
        records.append(rec)
        if done % args.log_every == 0:
            write_log(log_path, records)
        if not math.isfinite(loss):
            print("loss不是有限的数了，训练发散，停下")
            break
        if done % args.ckpt_every == 0 or done == args.steps:
            save_checkpoint(ckpt_path, model, opt, scaler, gen, done, elapsed,
                            records, args)

    write_log(log_path, records)
    print(f"训练用时{elapsed / 60:.1f}分钟（不含验证、生成和存盘），"
          f"这一次运行里梯度溢出、跳过了{skipped}步")
    if args.device == "cuda":
        print(f"显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if args.export:
        # 对话模型答完一轮写<|im_end|>，推理时抽到它就该停。
        model.config.eos_token_id = IM_END
        export_model(model, out / "model", tok_path)
        print(f"模型目录存到 {out / 'model'}")


if __name__ == "__main__":
    main()
