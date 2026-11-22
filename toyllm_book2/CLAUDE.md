# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

《自己动手训练LLM》 (title provisional) — the sister book of `../toyllm_book1/`. Book 1 hand-writes an engine that runs someone else's model; this one trains a ~26M Llama-style model from scratch on a free Colab T4, exports it, and runs it on book 1's engine. The two books meet at one interface only: a HuggingFace-style model directory.

Read the working docs before anything substantial. They are drafts that change with discussion, not manuscript:

| File | What it holds |
|------|---------------|
| `idea.md` | positioning, why this book, the route (第一部分 跑通 / 第二部分 后训练), dataset candidates, prior art |
| `design.md` | the code design: hardware constraints (Colab T4, the author's Intel Mac), the one model config, datasets, `code/` layout, modules, the interface to book 1, the validation table, PoC questions, per-chapter code plan, and its 待定 list |
| `TODO.md` | open manuscript decisions; **not part of the book**, the linters skip it |
| `ref.md` | links |

See also the workspace `../CLAUDE.md`: prose rules, notation, and the shared lint skills apply here.

## Layout

| Path | Role |
|------|------|
| `chapters/_front.md`, `_preface.md`, `chNN_slug.md` | the manuscript, `ch01`–`ch13` |
| `code/` | one uv project (torch pinned to 2.2.2 for the author's Intel Mac; code uses only torch-2.2 APIs so it also runs on Colab's newer torch) |
| `code/book/chNN/` | the book's code, one directory per chapter, each runnable on its own |
| `code/data/`, `code/models/` | corpora and weights, not in git; `code/download_model.sh` fetches SmolLM2-135M, used as the reference answer |
| `draw/chNN.drawio`, `images/chNN/` | diagram sources and exported PNGs; `python3 export_drawio.py chNN` exports one chapter and shrinks the PNGs |

`merge.py` (a copy of book 1's) concatenates `chapters/` into `Book.md`, which is gitignored; never edit `Book.md` by hand.

## Chapter status

`ch01`–`ch05` (概览, 数据, 分词, 模型, 训练循环) are drafted with code in `code/book/ch01`–`ch05`; ch05 runs entirely on the Mac (300-step runs on MPS). `ch06` (预训练) is drafted from the full T4 run (3,815 steps, 3.43 h pure training, val 3.127, one disconnect after step 1630, resumed from the step-1600 checkpoint); bench.py numbers from a Colab T4 run are in; the 对照实验 are written (fp16 overflow, weight tying, vocab size; data amount moved to ch12, data source left to readers in ch02); ch06 has no TODOs left. `code/book/ch06/train.py` adds fp16 + `GradScaler`, throughput in tok/s and 6N TFLOPS, periodic samples, a complete checkpoint (scaler, data RNG, elapsed time and the whole log so far; written atomically; on resume the log is rebuilt from the checkpoint, because on Colab's Drive mount a constantly-rewritten `log.jsonl` was lost across disconnects while `ckpt.pt` survived; `log.jsonl` itself is rewritten atomically every `--log-every` steps for watching) and `--data`/`--runs`/`--export`; `bench.py` measures throughput; `eval.py` evaluates the exported model against SmolLM2-135M (a transformers logits check, bits per byte on the first 100 val documents, ten greedy probe prompts; about two minutes on CPU); `plot_log.py` (PEP 723, matplotlib) draws `images/ch06/loss.png`, `grad.png`, `speed.png` from `code/runs/pretrain/log.jsonl`; the 对照实验 (fp16 overflow, `--untie`, vocab 4k/8k/16k; 400-step runs, `RUN_EXPERIMENTS` in `colab.ipynb`) use `retokenize.py`, which decodes the first 100M tokens of `train.bin` and re-encodes them into `data/vocab-{4k,8k,16k}/` (same documents for all three; the 8k copy is asserted identical), and `compare_bpb.py`, which compares exported models by bits per byte on val docs that fit 1023 tokens under every tokenizer; `train.py` now takes the vocab size from the tokenizer; `check_resume.py` compares two logs (a killed-and-resumed CPU run is bit-identical to an uninterrupted one); `colab.ipynb` mounts Drive, sparse-clones the public mirror https://github.com/zxh0/vibe-writing (so ch06 must be synced there first), copies the token files from Drive to local disk and calls those scripts. `ch07`–`ch13` (SFT, DPO, RL, 蒸馏, LoRA, 规模, 基准测试) are still one-paragraph stubs. `code/book/ch07/` is written but not yet run on the T4: `chat_template.py` (ChatML, loss mask on assistant content plus `<|im_end|>`), `prepare_sft.py` (smol-smoltalk, Apache 2.0; conversations over 1025 tokens cut at the last assistant turn that fits; best-fit packing into 1025-token rows, about 6% padding; writes `data/sft/{train,val}.bin` plus `_mask.bin`), `sft.py` (ch06 `train.py` adapted: `--init` from the base model dir, masked targets set to -100, exports with `eos_token_id` = 2), `chat.py` (our own `generate`, base vs SFT side by side) and `colab.ipynb`. ch04 opens with a self-contained Llama overview (formulas for RMSNorm, attention with causal mask, RoPE half-split and `rotate_half`, SwiGLU), so a reader does not need book 1; keep later chapters that way too. `ch13` (added 2026-11-22, last chapter, under 走出去) explains how benchmark scores are produced, with SmolLM2-135M and Qwen3-0.6B as subjects; our model only appears to show it sits on the random line, so ch06 keeps just bpb and ten probes. `design.md`'s 目录结构 also plans `poc/`, `colab/`, `runs/` and `reference/`; none exist yet.

## Conventions specific to this book

- **Notation is book 1's**: Goodfellow, so matrices are **bold uppercase**. The `符号记法/粗体大写` lint rule is off for this book in `../.claude/lint-scope.json`; do not "fix" bold matrices.
- **The author is “我” in the prose** (decided 2026-11-22), never “作者”: 我的Mac, 我选了后者. 作者 stays only for other people's work (a paper's authors).
- **Never write 对拍**, in chapters, docs, comments or code. The author dislikes the word. Use 核对、核对结果、和参考实现核对、对答案. Book 1 still has 对拍 in places, so do not carry it over when borrowing text or code.
- **Verify against a reference**, as book 1 does: every component is checked against `transformers` / `torch` (logits, loss and gradients to relative error < 1e-5 in fp32, the exact standards are in `design.md`'s 验证 table). `transformers` is a dev dependency only, used in checking scripts, never in the chapters' main code.
- **Self-contained**: code never imports or references book 1's files; copy what is needed. The shared lint skills in `../.claude/skills/` are the one exception.
- **Each chapter builds on book 1** rather than re-teaching it: the architecture chapters point back to the book-1 chapter that explained a component.

## Before committing a chapter

```bash
python3 ../.claude/skills/check-formulas/scripts/check.py chapters/chNN_slug.md
python3 ../.claude/skills/check-prose/scripts/check.py    chapters/chNN_slug.md
```
