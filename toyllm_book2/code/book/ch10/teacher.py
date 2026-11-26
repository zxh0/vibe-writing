"""第十章 · 老师写回答：Qwen3-0.6B回答prompts.jsonl里的每个问题

老师用transformers跑，不用第一本的引擎：引擎一次只写一条，这里要写
两万多条，得几十个问题一起写。几十个问题长短不一，左边补齐，让所有
问题的最后一个词元对齐，回答从同一列开始往后写。问题先按长度排好，
一批里的长短差不多，补齐的就少。

非思考模式（enable_thinking=False）：对话模板会替老师写好一段空的
<think></think>，它直接作答。采样参数用Qwen3给非思考模式推荐的：
温度0.7，top-p 0.8，top-k 20。

每批写完就追加到--out，每行一个回答：

    {"id": 问题的id, "answer": 回答, "finished": 是否自己停下,
     "tokens": 老师写了几个词元}

断了以后重跑，已经写过的问题会跳过。验证集的问题先写。

在Colab上，--out写在本地盘，每隔几批整个复制一份到Drive（--backup，
先写临时文件再改名）：第六章发现，Drive上一直在改的文件，断线时可能
丢；整个换掉的文件不会。重新连上以后，本地盘是空的，就先从备份复制回来。

运行：
    cd code
    # 在Mac上试几个问题（CPU，fp32，几分钟）
    uv run python book/ch10/teacher.py --model models/Qwen3-0.6B \\
        --limit 8 --batch 4 --out data/distill/teacher-try.jsonl
    # 在T4上（fp16），两万多个问题
    python book/ch10/teacher.py --prompts .../prompts.jsonl \\
        --out /content/data/distill/teacher.jsonl \\
        --backup /content/drive/MyDrive/toyllm2/data/distill/teacher.jsonl
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B",
                    help="HuggingFace上的名字，或者下载好的目录")
    ap.add_argument("--prompts", type=Path,
                    default=DATA / "distill" / "prompts.jsonl")
    ap.add_argument("--out", type=Path,
                    default=DATA / "distill" / "teacher.jsonl")
    ap.add_argument("--batch", type=int, default=64, help="几个问题一起写")
    ap.add_argument("--max-new", type=int, default=512,
                    help="每个回答最多几个词元（老师的词元）")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--backup", type=Path, default=None,
                    help="每隔几批把--out整个复制到这里")
    ap.add_argument("--backup-every", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="只写前几个，试跑用")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model = model.to(args.device).eval()
    stop = model.generation_config.eos_token_id         # <|im_end|>等
    stop = set(stop if isinstance(stop, list) else [stop])

    def backup():
        if args.backup is not None:
            tmp = args.backup.with_suffix(".tmp")
            shutil.copy(args.out, tmp)
            os.replace(tmp, args.backup)

    if args.backup is not None and args.backup.exists() and not args.out.exists():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.backup, args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prompts = [json.loads(line) for line in args.prompts.open()]
    done = set()
    if args.out.exists():
        done = {json.loads(line)["id"] for line in args.out.open()}
    todo = [p for p in prompts if p["id"] not in done]
    if args.limit is not None:
        todo = todo[:max(0, args.limit - len(done))]
    # 验证集在前；同一份里按问题的长度排，一批里长短差不多。
    todo.sort(key=lambda p: (p["split"] != "val", len(p["prompt"])))
    print(f"老师 {args.model}，{args.device}，{dtype}；"
          f"{len(prompts):,}个问题，已经写过{len(done):,}个，"
          f"这次写{len(todo):,}个", flush=True)

    torch.manual_seed(args.seed + len(done))
    t0, n_done, n_tok = time.perf_counter(), 0, 0
    with args.out.open("a") as f:
        for i in range(0, len(todo), args.batch):
            batch = todo[i:i + args.batch]
            texts = [tok.apply_chat_template(
                [{"role": "user", "content": p["prompt"]}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
                for p in batch]
            enc = tok(texts, return_tensors="pt", padding=True).to(args.device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=args.max_new, do_sample=True,
                    temperature=args.temperature, top_p=args.top_p,
                    top_k=args.top_k, pad_token_id=tok.pad_token_id)
            new = out[:, enc["input_ids"].shape[1]:].tolist()
            for p, ids in zip(batch, new):
                end = next((k for k, t in enumerate(ids) if t in stop), None)
                ids = ids if end is None else ids[:end]
                f.write(json.dumps({
                    "id": p["id"],
                    "answer": tok.decode(ids, skip_special_tokens=True).strip(),
                    "finished": end is not None, "tokens": len(ids)},
                    ensure_ascii=False) + "\n")
                n_tok += len(ids)
            f.flush()
            n_done += len(batch)
            if (i // args.batch + 1) % args.backup_every == 0:
                backup()
            took = time.perf_counter() - t0
            left = (len(todo) - n_done) / (n_done / took)
            print(f"  {len(done) + n_done:>6,}/{len(prompts):,}  "
                  f"每秒{n_done / took:4.1f}个回答、{n_tok / took:5.0f}个词元  "
                  f"已用{took / 60:5.1f}分钟  还要{left / 60:5.1f}分钟",
                  flush=True)
    backup()
    print(f"写到 {args.out}")


if __name__ == "__main__":
    main()
