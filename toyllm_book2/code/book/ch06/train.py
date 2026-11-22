"""第六章 · 预训练：把第五章的循环搬到T4上

和第五章比，多了五样东西：

    1. 混合精度：CUDA上前向和反向用fp16算，参数和优化器状态还是fp32。
       T4没有bf16，fp16的范围又窄，要用GradScaler做损失缩放。
    2. 吞吐量：每隔几步打印一次每秒多少个词元，按6N折算成每秒多少次浮点运算。
    3. 定期生成：每隔一段，用几句固定的开头续写，看模型学到了什么。
    4. 真正的断点续训：checkpoint里还存了GradScaler、取数据的随机数状态、
       已经用掉的时间和到这一步为止的全部日志，每隔一段存一次。接着训时，
       日志从checkpoint里取，从checkpoint那一步重新训，曲线和没断过一样。
    5. 数据和记录的位置可以指定：在Colab上，记录写到Google Drive，
       会话断了也不丢。

优化器换成torch自带的AdamW，CUDA上用它的fused实现，快一些。
第五章已经核对过，它和我们手写的那个算的是同一件事。

运行：
    cd code
    # 在Mac上，用很小的batch试一试（fp32）
    uv run python book/ch06/train.py --name try --batch 4 --seq 128 --accum 1 \\
        --steps 60 --warmup 6 --eval-every 20 --sample-every 30 --ckpt-every 20
    # 在T4上，完整的预训练（默认值就是它），断了以后加--resume接着训
    python book/ch06/train.py --name pretrain --runs /content/drive/MyDrive/toyllm2/runs
    python book/ch06/train.py --name pretrain --runs ... --resume
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
from config import CODE, DATA, Config
from export import save as export_model
from model import LLM
from schedule import lr_at

PROMPTS = ["Once upon a time", "The capital of France is",
           "Photosynthesis is the process"]


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


def get_batch(data, B, T, gen, device):
    """随机截B段，每段T+1个词元。错开一位，就是输入和答案。"""
    starts = torch.randint(0, len(data) - T - 1, (B,), generator=gen).tolist()
    chunk = np.stack([data[s:s + T + 1] for s in starts]).astype(np.int64)
    chunk = torch.from_numpy(chunk).to(device)
    return chunk[:, :-1], chunk[:, 1:]


@torch.no_grad()
def evaluate(model, data, B, T, n_batches, device, amp):
    """验证集开头的n_batches批，每次都是同样的数据，前后才好比。"""
    model.eval()
    total = 0.0
    for i in range(n_batches):
        start = i * B * (T + 1)
        chunk = data[start:start + B * (T + 1)].astype(np.int64)
        chunk = torch.from_numpy(chunk).view(B, T + 1).to(device)
        with amp:
            total += model(chunk[:, :-1], chunk[:, 1:])[1].item()
    model.train()
    return total / n_batches


def sample(model, tok, device, seed, n=40):
    """每句开头续写n个词元。随机数在一个分叉出来的状态里用，用完还原，
    所以生成不会改变训练本身；种子固定，不同阶段的样例才好比。"""
    model.eval()
    out = []
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for prompt in PROMPTS:
            ids = torch.tensor([tok.encode(prompt).ids], device=device)
            ids = model.generate(ids, n, temperature=0.8, top_k=50)
            out.append(tok.decode(ids[0].tolist()))
    model.train()
    return out


def write_log(path, records):
    """把日志整个重写一遍。和checkpoint一样，先写临时文件再改名，
    读到的log.jsonl总是完整的。"""
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                           for r in records))
    os.replace(tmp, path)


def read_log(path, step):
    """旧的checkpoint里没有日志，只好从log.jsonl里读，比checkpoint新的不要。"""
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [r for r in records if r["step"] <= step]


def save_checkpoint(path, model, opt, scaler, gen, step, elapsed, records,
                    args):
    """接着训需要的全部东西。先写到临时文件，写完再改名：
    写到一半会话断了，原来那个checkpoint还是好的。

    日志也存在里面。log.jsonl是一直在改的文件，在Colab挂载的Drive上，
    会话一断，它不一定保得住；checkpoint保住了，到这一步为止的日志就在。"""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "elapsed": elapsed,
        "data_rng": gen.get_state(),
        "cpu_rng": torch.get_rng_state(),
        "log": records,
        # 路径存成字符串：checkpoint里只放基本类型，新版torch默认的
        # weights_only=True也能读。
        "args": {k: str(v) if isinstance(v, Path) else v
                 for k, v in vars(args).items()},
    }
    if torch.cuda.is_available():
        ckpt["cuda_rng"] = torch.cuda.get_rng_state_all()
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, opt, scaler, gen, args):
    # 先全部读到CPU上：随机数状态只能是CPU上的张量，模型和优化器的状态，
    # load_state_dict会自己搬到参数所在的设备。文件是我们自己存的，
    # weights_only=False没有风险，旧的checkpoint里还存着Path，也读得进来。
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    gen.set_state(ckpt["data_rng"])
    torch.set_rng_state(ckpt["cpu_rng"])
    if "cuda_rng" in ckpt and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    # 影响训练结果的参数，接着训时要和原来一样，不一样就提醒一声。
    same = ("batch", "seq", "accum", "steps", "lr", "min_lr", "warmup",
            "weight_decay", "clip", "dtype", "seed")
    for k in same:
        if ckpt["args"].get(k) != getattr(args, k):
            print(f"注意：{k}原来是{ckpt['args'].get(k)}，现在是{getattr(args, k)}")
    return ckpt["step"], ckpt["elapsed"], ckpt.get("log")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pretrain", help="记录写到<runs>/<name>/")
    ap.add_argument("--data", type=Path, default=DATA,
                    help="里面要有tokens/train.bin、tokens/val.bin、tokenizer/tokenizer.json")
    ap.add_argument("--runs", type=Path, default=CODE / "runs")
    # 默认值是T4上的完整预训练：每步32×1024×4 = 131,072个词元，
    # 3,815步约5亿个词元（26.7M参数的约20倍）。
    ap.add_argument("--steps", type=int, default=3815)
    ap.add_argument("--batch", type=int, default=32, help="每次前向的序列条数B")
    ap.add_argument("--seq", type=int, default=1024, help="每条序列的词元数T")
    ap.add_argument("--accum", type=int, default=4, help="梯度累积：几次前向更新一次")
    ap.add_argument("--lr", type=float, default=1e-3, help="最大学习率")
    ap.add_argument("--min-lr", type=float, default=None, help="默认是lr的十分之一")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0, help="0表示不裁剪")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default=None,
                    help="默认CUDA上fp16，其他设备fp32")
    ap.add_argument("--init-scale", type=float, default=2.0 ** 16,
                    help="GradScaler一开始把loss放大多少倍")
    ap.add_argument("--compile", action="store_true", help="用torch.compile，要torch 2.x新版")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="有checkpoint就接着训，没有就从头开始")
    ap.add_argument("--kill-at", type=int, default=None,
                    help="训到第几步直接退出，不存盘，模拟会话断开")
    ap.add_argument("--untie", action="store_true",
                    help="输出层不和词嵌入共用矩阵（对照实验用）")
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
    train = np.memmap(args.data / "tokens" / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(args.data / "tokens" / "val.bin", dtype=np.uint16, mode="r")
    tok = Tokenizer.from_file(str(args.data / "tokenizer" / "tokenizer.json"))

    # 词表大小以分词器为准，对照实验会换成4096、16384的词表。
    cfg = Config(vocab_size=tok.get_vocab_size(),
                 tie_word_embeddings=not args.untie)
    torch.manual_seed(args.seed)
    model = LLM(cfg).to(args.device)
    fused = args.device == "cuda"
    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr,
                            betas=(0.9, 0.95), eps=1e-8, fused=fused)
    fp16 = args.dtype == "fp16"
    scaler = make_scaler(fp16, args.init_scale)
    # autocast：矩阵乘法这类运算自动换成fp16，softmax、归一化这类留在fp32。
    amp = torch.autocast("cuda", dtype=torch.float16) if fp16 else nullcontext()
    gen = torch.Generator().manual_seed(args.seed)
    fwd = torch.compile(model) if args.compile else model

    # 日志先记在内存里：每隔log_every步整个写一遍log.jsonl，方便随时看；
    # 存checkpoint时也存一份，接着训时从checkpoint里取。
    start, elapsed, records = 0, 0.0, []
    if args.resume and ckpt_path.exists():
        start, elapsed, records = load_checkpoint(ckpt_path, model, opt, scaler,
                                                  gen, args)
        if records is None:
            records = read_log(log_path, start)
        print(f"从第{start}步接着训，之前已经训了{elapsed / 60:.1f}分钟")

    def write(rec):
        records.append(rec)

    n_params = sum(p.numel() for p in model.parameters())
    tokens_per_step = args.batch * args.seq * args.accum
    print(f"设备 {args.device}，{args.dtype}，参数{n_params:,}个")
    print(f"每步 {args.batch}×{args.seq}×{args.accum} = {tokens_per_step:,} 个词元，"
          f"共{args.steps:,}步，{args.steps * tokens_per_step / 1e6:,.1f}M个词元")

    if start == 0:
        val_loss = evaluate(model, val, args.batch, args.seq, args.eval_batches,
                            args.device, amp)
        write({"step": 0, "val_loss": val_loss})
        print(f"第{0:>5}步  验证loss {val_loss:.3f}")

    # 吞吐量只算训练本身的时间，不含验证、生成和存盘。
    window_time, window_tokens, skipped = 0.0, 0, 0
    for step in range(start, args.steps):
        t_step = time.perf_counter()
        # 1. 学习率
        lr = lr_at(step, args.lr, args.warmup, args.steps, args.min_lr)
        for group in opt.param_groups:
            group["lr"] = lr

        # 2、3. 取数据，前向，反向。scaler.scale把loss放大，
        # 反向传播得到的梯度也跟着放大，fp16里很小的梯度才不会变成0。
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(args.accum):
            x, y = get_batch(train, args.batch, args.seq, gen, args.device)
            with amp:
                _, loss = fwd(x, y)
            scaler.scale(loss / args.accum).backward()
            loss_sum += loss.item()
        loss = loss_sum / args.accum

        # 4. 裁剪之前，先把梯度缩回原来的大小。梯度里有inf或nan时，
        # 算出来的范数也是inf或nan，这一步会被scaler跳过。
        scaler.unscale_(opt)
        max_norm = args.clip if args.clip > 0 else math.inf
        norm = clip_grad_norm_(model.parameters(), max_norm).item()

        # 5. 更新参数。梯度溢出了，scaler就跳过这一步，并把放大倍数减半；
        # 连续2000步没溢出，再翻一倍。
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
                # 显存峰值，GiB。训完时也会打印，但那只在笔记本的输出里
                rec["mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
            print(f"第{done:>5}步  loss {loss:.3f}  梯度范数 {norm:5.2f}  "
                  f"学习率 {lr:.1e}  每秒{speed:,.0f}个词元  "
                  f"约{6 * n_params * speed / 1e12:.1f} TFLOPS", flush=True)
        if done % args.eval_every == 0 or done == args.steps:
            rec["val_loss"] = evaluate(model, val, args.batch, args.seq,
                                       args.eval_batches, args.device, amp)
            print(f"第{done:>5}步  验证loss {rec['val_loss']:.3f}  "
                  f"已训{done * tokens_per_step / 1e6:,.1f}M个词元，"
                  f"用时{elapsed / 60:.1f}分钟")
        if done % args.sample_every == 0 or done == args.steps:
            rec["samples"] = sample(model, tok, args.device, args.seed)
            for text in rec["samples"]:
                print(textwrap.indent(textwrap.fill(repr(text), 74), "    "))
        write(rec)
        if done % args.log_every == 0:
            write_log(log_path, records)
        if not math.isfinite(loss):
            print("loss不是有限的数了，训练发散，停下")
            break
        if done == args.kill_at:
            print(f"在第{done}步模拟断开：直接退出，不存盘")
            return
        if done % args.ckpt_every == 0 or done == args.steps:
            save_checkpoint(ckpt_path, model, opt, scaler, gen, done, elapsed,
                            records, args)

    write_log(log_path, records)
    print(f"训练用时{elapsed / 60:.1f}分钟（不含验证、生成和存盘），"
          f"这一次运行里梯度溢出、跳过了{skipped}步")
    if args.device == "cuda":
        print(f"显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if args.export:
        export_model(model, out / "model", args.data / "tokenizer" / "tokenizer.json")
        print(f"模型目录存到 {out / 'model'}")


if __name__ == "__main__":
    main()
