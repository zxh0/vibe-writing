"""第八章 · DPO：第七章的sft.py改几处

混合精度、checkpoint和断点续训和第七章一模一样，改的是这几处：

    1. 两个模型：策略模型（要训的那个）和参考模型，都从第七章导出的
       SFT模型读进来；参考模型冻住，只做前向。
    2. 数据：prepare_dpo.py分好的一对一对的对话。每次取B对，好回答和
       坏回答各B段，补齐到同样长，前B行是好的，后B行是坏的。
    3. loss：先算出每段对话里回答部分的对数概率，再按DPO的公式算。
    4. 日志：多记准确率、好坏回答的隐式奖励和它们的差。

运行：
    cd code
    # 在Mac上，用很小的batch试一试（fp32）
    uv run python book/ch08/dpo.py --name dpo-try --batch 2 --accum 1 \\
        --steps 20 --warmup 2 --eval-every 10 --eval-pairs 32 \\
        --sample-every 20 --ckpt-every 10
    # 在T4上（默认值），断了以后再跑一遍，--resume会接着训
    python book/ch08/dpo.py --init .../runs/sft/model \\
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
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.nn.utils import clip_grad_norm_

from adamw import param_groups
from chat import reply
from chat_template import EOT, ChatTemplate
from config import CODE, DATA
from export import load as load_model
from export import save as export_model
from schedule import lr_at

QUESTIONS = ["What is the capital of France?",
             "Give me three tips for staying healthy.",
             "Hi! How are you today?"]


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_scaler(enabled, init_scale):
    """损失缩放器。torch 2.3起叫torch.amp.GradScaler，
    2.2只有torch.cuda.amp里的。"""
    try:
        return torch.amp.GradScaler("cuda", init_scale=init_scale,
                                    enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.GradScaler(init_scale=init_scale, enabled=enabled)


def read_pairs(data_dir, split):
    """prepare_dpo.py写的三个文件：词元、掩码、每段的起点。"""
    ids = np.fromfile(data_dir / f"{split}.bin", dtype=np.uint16)
    mask = np.fromfile(data_dir / f"{split}_mask.bin", dtype=np.uint8)
    offsets = np.load(data_dir / f"{split}_offsets.npy")
    return ids, mask, offsets


def n_pairs(data):
    return (len(data[2]) - 1) // 2


def to_batch(data, pairs, device):
    """取出这几对，好回答的B段在前，坏回答的B段在后，补齐到同样长。
    长度向上取到64的倍数，形状的种类少一些。和第七章一样，错开一位是
    输入和答案，掩码是0的位置（问题、模板、补齐）答案换成-100。"""
    ids, mask, offsets = data
    seqs = [2 * p for p in pairs] + [2 * p + 1 for p in pairs]
    lens = [offsets[s + 1] - offsets[s] for s in seqs]
    T = min(math.ceil(max(lens) / 64) * 64, 1025)
    rows = np.full((len(seqs), T), EOT, dtype=np.int64)
    learn = np.zeros((len(seqs), T), dtype=bool)
    for i, (s, n) in enumerate(zip(seqs, lens)):
        rows[i, :n] = ids[offsets[s]:offsets[s + 1]]
        learn[i, :n] = mask[offsets[s]:offsets[s + 1]]
    rows, learn = torch.from_numpy(rows), torch.from_numpy(learn)
    x, y = rows[:, :-1], rows[:, 1:].masked_fill(~learn[:, 1:], -100)
    return x.to(device), y.to(device)


def sequence_logps(model, x, y):
    """每一行回答部分的对数概率：每个词元的log p加起来。

    cross_entropy算的就是-log p，reduction="none"让它每个位置给一个数，
    答案是-100的位置给0，正好不算。所以一行加起来，取负号就是这一行
    回答的对数概率。在fp32里算。"""
    logits = model(x)[0].float()
    nll = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                          reduction="none")
    return -nll.view(y.shape).sum(-1)


def dpo_loss(pc, pr, rc, rr, beta):
    """DPO的loss。pc、pr是策略模型给好回答、坏回答的对数概率，
    rc、rr是参考模型给的。隐式奖励是beta乘以“比参考模型高了多少”：

        loss = -log σ( beta·(pc - rc) - beta·(pr - rr) )

    返回loss（B对的平均）和两个隐式奖励（每对一个，不求梯度）。"""
    chosen = beta * (pc - rc)
    rejected = beta * (pr - rr)
    loss = -F.logsigmoid(chosen - rejected).mean()
    return loss, chosen.detach(), rejected.detach()


def pair_stats(pc, pr, chosen, rejected):
    """记进日志的几个数，都是这几对的平均。"""
    return {"acc": (chosen > rejected).float().mean().item(),
            "margin": (chosen - rejected).mean().item(),
            "chosen": chosen.mean().item(), "rejected": rejected.mean().item(),
            "logp_chosen": pc.mean().item(), "logp_rejected": pr.mean().item()}


def step_pairs(policy, ref, data, pairs, beta, device, amp):
    """一批：策略模型和参考模型各算一遍，返回loss和统计。"""
    x, y = to_batch(data, pairs, device)
    B = len(pairs)
    with torch.no_grad(), amp:
        r = sequence_logps(ref, x, y)
    with amp:
        p = sequence_logps(policy, x, y)
    loss, chosen, rejected = dpo_loss(p[:B], p[B:], r[:B], r[B:], beta)
    return loss, pair_stats(p[:B].detach(), p[B:].detach(), chosen, rejected)


@torch.no_grad()
def evaluate(policy, ref, data, B, n, beta, device, amp):
    """验证集的前n对，每次都是同样的数据，前后才好比。"""
    policy.eval()
    total, stats = 0.0, {}
    batches = [list(range(i, min(i + B, n))) for i in range(0, n, B)]
    for pairs in batches:
        loss, s = step_pairs(policy, ref, data, pairs, beta, device, amp)
        w = len(pairs) / n
        total += loss.item() * w
        for k, v in s.items():
            stats[k] = stats.get(k, 0.0) + v * w
    policy.train()
    return {"val_loss": total, **{f"val_{k}": v for k, v in stats.items()}}


def sample(model, chat, device, seed, n=100):
    """和第七章一样问几个问题，随机数在分叉出来的状态里用，用完还原。
    顺便记下每个回答有多少个词元，看DPO会不会让回答变长。"""
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
    """把日志整个重写一遍，先写临时文件再改名。"""
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                           for r in records))
    os.replace(tmp, path)


def save_checkpoint(path, model, opt, scaler, gen, step, elapsed, records,
                    args):
    """接着训需要的全部东西，和第七章一样。参考模型不用存，它没变过。"""
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
    same = ("batch", "accum", "steps", "lr", "min_lr", "warmup", "beta",
            "weight_decay", "clip", "dtype", "seed")
    for k in same:
        if ckpt["args"].get(k) != getattr(args, k):
            print(f"注意：{k}原来是{ckpt['args'].get(k)}，"
                  f"现在是{getattr(args, k)}")
    return ckpt["step"], ckpt["elapsed"], ckpt["log"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dpo", help="记录写到<runs>/<name>/")
    ap.add_argument("--init", type=Path,
                    default=CODE / "runs" / "sft" / "model",
                    help="第七章导出的SFT模型目录，策略和参考都从它开始")
    ap.add_argument("--data", type=Path, default=DATA / "dpo",
                    help="prepare_dpo.py的输出目录")
    ap.add_argument("--runs", type=Path, default=CODE / "runs")
    ap.add_argument("--beta", type=float, default=0.1,
                    help="能偏离参考模型多远，越小走得越远")
    ap.add_argument("--epochs", type=float, default=1.0,
                    help="把训练集过几遍，用来算默认的步数")
    ap.add_argument("--steps", type=int, default=None,
                    help="给了就不按epochs算")
    ap.add_argument("--batch", type=int, default=8, help="每次前向的对数B")
    ap.add_argument("--accum", type=int, default=4,
                    help="梯度累积：几次前向更新一次")
    ap.add_argument("--lr", type=float, default=5e-6, help="最大学习率")
    ap.add_argument("--min-lr", type=float, default=None,
                    help="默认是lr的十分之一")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0, help="0表示不裁剪")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default=None,
                    help="默认CUDA上fp16，其他设备fp32")
    ap.add_argument("--init-scale", type=float, default=2.0 ** 16)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-pairs", type=int, default=None,
                    help="验证用前几对，默认全部")
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
    train = read_pairs(args.data, "train")
    val = read_pairs(args.data, "val")
    n_train = n_pairs(train)
    n_val = min(args.eval_pairs or n_pairs(val), n_pairs(val))
    tok_path = args.init / "tokenizer.json"
    chat = ChatTemplate(Tokenizer.from_file(str(tok_path)))
    if args.steps is None:
        args.steps = math.ceil(args.epochs * n_train
                               / (args.batch * args.accum))

    torch.manual_seed(args.seed)
    policy = load_model(args.init).to(args.device)
    ref = load_model(args.init).to(args.device).eval()
    ref.requires_grad_(False)
    fused = args.device == "cuda"
    opt = torch.optim.AdamW(param_groups(policy, args.weight_decay),
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                            fused=fused)
    fp16 = args.dtype == "fp16"
    scaler = make_scaler(fp16, args.init_scale)
    amp = torch.autocast("cuda", dtype=torch.float16) if fp16 else nullcontext()
    gen = torch.Generator().manual_seed(args.seed)

    start, elapsed, records = 0, 0.0, []
    if args.resume and ckpt_path.exists():
        start, elapsed, records = load_checkpoint(ckpt_path, policy, opt,
                                                  scaler, gen, args)
        print(f"从第{start}步接着训，之前已经训了{elapsed / 60:.1f}分钟")

    pairs_per_step = args.batch * args.accum
    print(f"设备 {args.device}，{args.dtype}，beta {args.beta}，"
          f"起点 {args.init}")
    print(f"训练集{n_train:,}对，验证用{n_val:,}对；每步{pairs_per_step}对，"
          f"共{args.steps:,}步，"
          f"约{args.steps * pairs_per_step / n_train:.2f}遍")

    def run_eval(rec):
        rec.update(evaluate(policy, ref, val, args.batch, n_val, args.beta,
                            args.device, amp))
        print(f"第{rec['step']:>5}步  验证loss {rec['val_loss']:.4f}  "
              f"准确率 {rec['val_acc']:.1%}  差距 {rec['val_margin']:+.3f}  "
              f"好 {rec['val_chosen']:+.3f}  坏 {rec['val_rejected']:+.3f}")

    if start == 0:
        rec = {"step": 0}
        run_eval(rec)
        rec["samples"] = sample(policy, chat, args.device, args.seed)
        for text in rec["samples"]:
            print(textwrap.indent(textwrap.fill(repr(text), 74), "    "))
        records.append(rec)

    window_time, window_pairs, skipped = 0.0, 0, 0
    for step in range(start, args.steps):
        t_step = time.perf_counter()
        lr = lr_at(step, args.lr, args.warmup, args.steps, args.min_lr)
        for group in opt.param_groups:
            group["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_sum, stats = 0.0, {}
        for _ in range(args.accum):
            pairs = torch.randint(0, n_train, (args.batch,),
                                  generator=gen).tolist()
            loss, s = step_pairs(policy, ref, train, pairs, args.beta,
                                 args.device, amp)
            scaler.scale(loss / args.accum).backward()
            loss_sum += loss.item()
            for k, v in s.items():
                stats[k] = stats.get(k, 0.0) + v / args.accum
            window_pairs += len(pairs)
        loss = loss_sum / args.accum

        scaler.unscale_(opt)
        max_norm = args.clip if args.clip > 0 else math.inf
        norm = clip_grad_norm_(policy.parameters(), max_norm).item()
        scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skip = scaler.get_scale() < scale
        skipped += skip

        done = step + 1
        took = time.perf_counter() - t_step
        elapsed += took
        window_time += took
        rec = {"step": done, "loss": loss, "lr": lr, "grad_norm": norm,
               "time": elapsed, **stats}
        if fp16:
            rec["scale"] = scale
        if skip:
            rec["skipped"] = True
            print(f"第{done:>5}步  梯度溢出，跳过这一步，"
                  f"放大倍数{scale:g} → {scaler.get_scale():g}")
        if done % args.log_every == 0:
            rec["pairs_per_s"] = window_pairs / window_time
            window_time, window_pairs = 0.0, 0
            if args.device == "cuda":
                rec["mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
            print(f"第{done:>5}步  loss {loss:.4f}  准确率 {stats['acc']:.0%}  "
                  f"差距 {stats['margin']:+.3f}  梯度范数 {norm:5.2f}  "
                  f"每秒{rec['pairs_per_s']:.1f}对", flush=True)
        if done % args.eval_every == 0 or done == args.steps:
            run_eval(rec)
        if done % args.sample_every == 0 or done == args.steps:
            rec["samples"] = sample(policy, chat, args.device, args.seed)
            for text in rec["samples"]:
                print(textwrap.indent(textwrap.fill(repr(text), 74), "    "))
        records.append(rec)
        if done % args.log_every == 0:
            write_log(log_path, records)
        if not math.isfinite(loss):
            print("loss不是有限的数了，训练发散，停下")
            break
        if done % args.ckpt_every == 0 or done == args.steps:
            save_checkpoint(ckpt_path, policy, opt, scaler, gen, done, elapsed,
                            records, args)

    write_log(log_path, records)
    print(f"训练用时{elapsed / 60:.1f}分钟（不含验证、生成和存盘），"
          f"这一次运行里梯度溢出、跳过了{skipped}步")
    if args.device == "cuda":
        print(f"显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if args.export:
        export_model(policy, out / "model", tok_path)
        print(f"模型目录存到 {out / 'model'}")


if __name__ == "__main__":
    main()
