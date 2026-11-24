"""第九章 · GRPO：让模型自己做加法题，只告诉它对还是错

每一步：

    1. 出P道题，两个加数的位数一样，位数从--digits里随机挑一个。
    2. 每道题让模型写G个回答（一组），温度1.0，用带KV缓存的generate。
    3. 判分：对了1，错了0。
    4. 组内比较：每个回答的优势 = (奖励 - 组内平均) / (组内标准差 + 很小的数)。
       一组全对或全错，优势全是0，这组只剩KL项起作用。
    5. loss：每个词元上是 -优势 × log p，加上beta乘以和参考模型的KL，
       在一个回答的词元上取平均，再在所有回答上取平均。

每批回答只用来更新一次，写回答的就是正在训练的这个模型，所以不需要
PPO的重要性采样和截断（ratio恒等于1）。参考模型是冷启动之后的模型，
冻住，只用来算KL。

运行：
    cd code
    # 在Mac上试几步（fp32）
    uv run python book/ch09/grpo.py --name grpo-try --device cpu \\
        --prompts 4 --group 4 --steps 10 --eval-every 5 --eval-n 20
    # 在T4上（默认值），断了以后再跑一遍，--resume会接着训
    python book/ch09/grpo.py --init .../runs/cold/model \\
        --runs /content/drive/MyDrive/toyllm2/runs --resume --export
"""

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.nn.utils import clip_grad_norm_

from adamw import param_groups
from addition import make, question
from chat_template import ChatTemplate
from config import CODE
from export import load as load_model
from export import save as export_model
from rollout import accuracy, fixed_problems, rollout
from schedule import lr_at


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_scaler(enabled, init_scale):
    """损失缩放器。torch 2.3起叫torch.amp.GradScaler，
    2.2只有torch.cuda.amp里的。"""
    try:
        return torch.amp.GradScaler("cuda", init_scale=init_scale,
                                    enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.GradScaler(init_scale=init_scale, enabled=enabled)


def advantages(rewards, G):
    """组内比较：减去组内平均，除以组内标准差。rewards (P*G,)。"""
    r = rewards.view(-1, G)
    adv = (r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-4)
    return adv.flatten()


def token_logps(model, ids, P, learn):
    """回答里每个词元的log p，(B, 回答长度)。不算loss的位置是0。

    第t个位置预测第t+1个词元，所以回答的第一个词元由提示词的最后
    一个位置预测：取logits的第P-1到倒数第二个位置。"""
    logits = model(ids[:, :-1])[0][:, P - 1:].float()
    target = ids[:, P:].masked_fill(~learn, -100)
    nll = F.cross_entropy(logits.flatten(0, 1), target.flatten(),
                          reduction="none")
    return -nll.view(target.shape)


def grpo_loss(logp, ref_logp, adv, learn, beta):
    """每个词元：-优势 × log p + beta × KL，先在回答内平均，再在回答间平均。

    KL用的是k3估计：exp(ref - logp) - (ref - logp) - 1，总是非负，
    策略和参考模型一样时是0。"""
    diff = ref_logp - logp
    kl = torch.exp(diff) - diff - 1
    per_token = -adv[:, None] * logp + beta * kl
    m = learn.float()
    per_answer = (per_token * m).sum(-1) / m.sum(-1)
    kl_mean = ((kl * m).sum(-1) / m.sum(-1)).mean()
    return per_answer.mean(), kl_mean.detach()


def write_log(path, records):
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                           for r in records))
    os.replace(tmp, path)


def save_checkpoint(path, model, opt, scaler, rng, step, elapsed, records,
                    args):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "elapsed": elapsed,
        "py_rng": rng.getstate(),
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


def load_checkpoint(path, model, opt, scaler, rng):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    rng.setstate(ckpt["py_rng"])
    torch.set_rng_state(ckpt["cpu_rng"])
    if "cuda_rng" in ckpt and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    return ckpt["step"], ckpt["elapsed"], ckpt["log"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="grpo", help="记录写到<runs>/<name>/")
    ap.add_argument("--init", type=Path,
                    default=CODE / "runs" / "cold" / "model",
                    help="冷启动之后的模型，策略和参考都从它开始")
    ap.add_argument("--runs", type=Path, default=CODE / "runs")
    ap.add_argument("--digits", default="2,3",
                    help="训练题的位数，每一步随机挑一个")
    ap.add_argument("--prompts", type=int, default=32, help="每步几道题P")
    ap.add_argument("--group", type=int, default=8, help="每题几个回答G")
    ap.add_argument("--max-new", type=int, default=24,
                    help="每个回答最多几个词元")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.04,
                    help="KL项的系数，0表示不要KL")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default=None,
                    help="默认CUDA上fp16，其他设备fp32")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-digits", default="1,2,3,4,5")
    ap.add_argument("--eval-n", type=int, default=200,
                    help="每种位数用几道固定的题量正确率")
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()
    if args.dtype is None:
        args.dtype = "fp16" if args.device == "cuda" else "fp32"
    assert args.dtype == "fp32" or args.device == "cuda", "fp16只在CUDA上用"
    return args


def main():
    args = parse_args()
    out = args.runs / args.name
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path, log_path = out / "ckpt.pt", out / "log.jsonl"
    tok_path = args.init / "tokenizer.json"
    chat = ChatTemplate(Tokenizer.from_file(str(tok_path)))
    digits = [int(d) for d in args.digits.split(",")]
    tests = fixed_problems([int(d) for d in args.eval_digits.split(",")],
                           args.eval_n)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    policy = load_model(args.init).to(args.device)
    ref = load_model(args.init).to(args.device).eval()
    ref.requires_grad_(False)
    opt = torch.optim.AdamW(param_groups(policy, 0.0), lr=args.lr,
                            betas=(0.9, 0.95), eps=1e-8,
                            fused=args.device == "cuda")
    fp16 = args.dtype == "fp16"
    scaler = make_scaler(fp16, 2.0 ** 16)
    amp = (torch.autocast("cuda", dtype=torch.float16) if fp16
           else nullcontext())

    start, elapsed, records = 0, 0.0, []
    if args.resume and ckpt_path.exists():
        start, elapsed, records = load_checkpoint(ckpt_path, policy, opt,
                                                  scaler, rng)
        print(f"从第{start}步接着训，之前已经训了{elapsed / 60:.1f}分钟")
    P, G = args.prompts, args.group
    print(f"设备 {args.device}，{args.dtype}，起点 {args.init}")
    print(f"每步{P}道题 × {G}个回答，位数从{digits}里挑，"
          f"beta {args.beta}，共{args.steps}步")

    def run_eval(rec):
        with amp:
            acc = accuracy(policy, chat, tests, args.max_new)
        rec["acc"] = {str(d): v for d, v in acc.items()}
        cells = "  ".join(f"{d}位 {v:6.1%}" for d, v in acc.items())
        print(f"第{rec['step']:>5}步  正确率  {cells}", flush=True)

    if start == 0:
        rec = {"step": 0}
        run_eval(rec)
        records.append(rec)

    for step in range(start, args.steps):
        t0 = time.perf_counter()
        lr = lr_at(step, args.lr, args.warmup, args.steps, args.lr / 10)
        for group in opt.param_groups:
            group["lr"] = lr

        d = rng.choice(digits)
        problems = [make(rng, d, d) for _ in range(P)]
        policy.eval()
        with amp:
            r = rollout(policy, chat, problems, G, args.max_new,
                        args.temperature)
        policy.train()
        t_gen = time.perf_counter() - t0
        adv = advantages(r["rewards"], G)
        learn = r["learn"]
        with amp:
            logp = token_logps(policy, r["ids"], r["prompt"], learn)
            with torch.no_grad():
                ref_logp = token_logps(ref, r["ids"], r["prompt"], learn)
        loss, kl = grpo_loss(logp, ref_logp, adv, learn, args.beta)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        norm = clip_grad_norm_(policy.parameters(), args.clip).item()
        scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skip = scaler.get_scale() < scale

        done = step + 1
        took = time.perf_counter() - t0
        elapsed += took
        groups = r["rewards"].view(P, G)
        mixed = ((groups.min(-1).values == 0)
                 & (groups.max(-1).values == 1)).float().mean().item()
        rec = {"step": done, "digits": d, "loss": loss.item(), "lr": lr,
               "reward": r["rewards"].mean().item(), "mixed": mixed,
               "kl": kl.item(), "grad_norm": norm,
               "length": learn.sum(-1).float().mean().item(),
               "time": elapsed, "gen_time": t_gen}
        if skip:
            rec["skipped"] = True
        if done % 10 == 0:
            print(f"第{done:>5}步  {d}位  奖励 {rec['reward']:.3f}  "
                  f"有信号的组 {mixed:4.0%}  KL {rec['kl']:.4f}  "
                  f"长度 {rec['length']:4.1f}  {took:.1f}秒", flush=True)
        if done % args.eval_every == 0 or done == args.steps:
            run_eval(rec)
            rec["samples"] = [f"{question(*problems[p])} → {r['texts'][p * G]}"
                              for p in range(min(P, 4))]
        records.append(rec)
        if done % 10 == 0:
            write_log(log_path, records)
        if done % args.ckpt_every == 0 or done == args.steps:
            save_checkpoint(ckpt_path, policy, opt, scaler, rng, done,
                            elapsed, records, args)

    write_log(log_path, records)
    print(f"训练用时{elapsed / 60:.1f}分钟（不含验证和存盘）")
    if args.device == "cuda":
        print(f"显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if args.export:
        export_model(policy, out / "model", tok_path)
        print(f"模型目录存到 {out / 'model'}")


if __name__ == "__main__":
    main()
