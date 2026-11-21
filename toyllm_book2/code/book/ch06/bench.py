"""第六章 · 吞吐量：每秒能训多少个词元

不读数据，用随机的词元，只测一次完整的训练步（前向、反向、裁剪、AdamW）
要多久。换几种设置各测一遍：

    精度：     fp32，或者fp16混合精度加GradScaler
    batch：    每次前向几条序列
    compile：  要不要用torch.compile（只在Colab的新版torch上测）

每种设置先空跑几步热身，再计时。热身也计一次时：用compile时，编译就发生在
热身的第一步里，热身用的秒数，减去不用compile时的热身秒数，大致就是编译
用了多久。显存不够的设置记为OOM，接着测下一种。
最后按6N（每个词元每个参数6次浮点运算）折算成TFLOPS。

运行：
    cd code
    uv run python book/ch06/bench.py --device cpu --batch 4 --seq 128   # Mac上试一试
    python book/ch06/bench.py                                         # T4上测全套
"""

import argparse
import time
from contextlib import nullcontext

import torch
from torch.nn.utils import clip_grad_norm_

from adamw import param_groups
from config import Config
from model import LLM


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def measure(B, T, dtype, compile_, device, steps, warmup):
    """返回每秒词元数、显存峰值（GiB）和热身用的秒数。"""
    torch.manual_seed(0)
    model = LLM(Config()).to(device)
    opt = torch.optim.AdamW(param_groups(model, 0.1), lr=1e-4, betas=(0.9, 0.95),
                            fused=device == "cuda")
    fp16 = dtype == "fp16"
    scaler = make_scaler(fp16)
    amp = torch.autocast("cuda", dtype=torch.float16) if fp16 else nullcontext()
    fwd = torch.compile(model) if compile_ else model
    ids = torch.randint(0, model.config.vocab_size, (B, T + 1), device=device)
    x, y = ids[:, :-1], ids[:, 1:]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def step():
        opt.zero_grad(set_to_none=True)
        with amp:
            _, loss = fwd(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

    t0 = time.perf_counter()
    for _ in range(warmup):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    speed = steps * B * T / (time.perf_counter() - t0)
    mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    return speed, mem, warm


def rjust(text, width):
    """按显示宽度右对齐：汉字占两列。"""
    return " " * (width - sum(2 if ord(c) > 0x2e80 else 1 for c in text)) + text


def main():
    cuda = torch.cuda.is_available()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if cuda else "cpu")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--compile", action="store_true", help="另外再测fp16加compile")
    args = ap.parse_args()

    on_cuda = args.device == "cuda"
    settings = [("fp32", False)]
    if on_cuda:
        settings.append(("fp16", False))
        if args.compile:
            settings.append(("fp16", True))
    n_params = sum(p.numel() for p in LLM(Config()).parameters())
    if on_cuda:
        print(torch.cuda.get_device_name(), f"torch {torch.__version__}")
    print(f"参数{n_params:,}个，每条序列{args.seq}个词元\n")
    print("精度  " + rjust("compile", 8) + rjust("B", 5) + rjust("每秒词元", 12)
          + rjust("TFLOPS", 9) + rjust("显存GiB", 10) + rjust("热身秒", 9))
    for dtype, compile_ in settings:
        for B in args.batch:
            try:
                speed, mem, warm = measure(B, args.seq, dtype, compile_, args.device,
                                     args.steps, args.warmup)
            except torch.cuda.OutOfMemoryError:
                print(f"{dtype:<6}{str(compile_):>8}{B:>5}{'OOM':>12}")
                torch.cuda.empty_cache()
                continue
            tflops = 6 * n_params * speed / 1e12
            print(f"{dtype:<6}{str(compile_):>8}{B:>5}{speed:>12,.0f}"
                  f"{tflops:>9.1f}{mem:>10.2f}{warm:>9.1f}", flush=True)


if __name__ == "__main__":
    main()
