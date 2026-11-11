# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU 学习笔记 (title provisional) — a book the author writes while learning GPUs, built around one question: **how do you make a very large matrix multiplication fast?** It starts on the CPU, writes a first CUDA kernel, then optimizes FP32 matmul step by step towards cuBLAS, adds Tensor Cores and low precision, and ends with general operators and multi-GPU.

Read `README.md` first: its 写作原则 are this book's standing rules (one benchmark through the whole book, hardware introduced only when a measurement asks for it, how terms are bolded and when to keep them in English, avoid “不是X，而是Y”). `outline.md` is the detailed plan (five parts, 17 chapters), `research.md` the survey of similar material, `ref.md` links. `tpu_book` is this book's pair and shares its language (matmul, arithmetic intensity, roofline).

See also the workspace `../CLAUDE.md`: prose rules, notation, and the shared lint skills apply here.

## Layout

| Path | Role |
|------|------|
| `chapters/_front.md`, `chNN_slug.md`, `appendix.md` | the manuscript, `ch01`–`ch17` |
| `Book.md` | **generated** by `python3 merge.py` (`--check` to verify), gitignored. Never edit by hand |
| `code/chNN/` | per-chapter code: Python and C on the CPU (`ch01`–`ch04`), CUDA from `ch05` on. Chapters with measurements have a `Makefile`, a `bench.sh` and its saved output `results.txt` |
| `draw/chNN.drawio`, `images/chNN/` | diagram sources, one page per figure, page name = image file name; `python3 export_drawio.py chNN` exports and shrinks |
| `tools/` | standalone HTML tools the chapters point readers to (`matmul-demo/`, `loop-order/`, `cuda-grid/`, `gpu-arch/`, `t4-explorer/`) |

## Chapter status

`ch01`–`ch08` are written (矩阵乘法, CPU, roofline, 为什么用GPU, 第一个CUDA程序, 执行模型, 访存合并, 共享内存分块). `ch09`–`ch17` are still outlines.

## Running the code

`bench.sh` runs every experiment of its chapter and writes `results.txt`. **Every number a chapter quotes comes from that file**; when the code changes, rerun and update both. Do not invent or estimate a measurement. The CPU chapters run on the author's Mac. The Mac has no NVIDIA GPU, so the CUDA chapters run on a Colab T4, driven by each chapter's `colab.ipynb`.

## Conventions specific to this book

- **Notation**: matrices are **bold uppercase** ($\mathbf{A}$, $\mathbf{B}$, $\mathbf{C}$), elements plain lowercase with 0-based subscripts ($a_{ik}$), as ch01 declares. The `符号记法/粗体大写` lint rule is off for this book in `../.claude/lint-scope.json`.
- **The benchmark table**: per `outline.md`, every chapter adds a row (GFLOPS and percent of hardware peak) to one running table.

## Before committing a chapter

```bash
python3 ../.claude/skills/check-formulas/scripts/check.py chapters/chNN_slug.md
python3 ../.claude/skills/check-prose/scripts/check.py    chapters/chNN_slug.md
python3 merge.py
```
