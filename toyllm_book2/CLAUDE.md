# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

《自己动手训练LLM》 (title provisional) — the sister book of `../toyllm_book1/`. Book 1 hand-writes an engine that runs someone else's model; this one trains a ~26M Llama-style model from scratch on a free Colab T4, exports it, and runs it on book 1's engine. The two books meet at one interface only: a HuggingFace-style model directory.

Read the working docs before anything substantial. They are drafts that change with discussion, not manuscript:

| File | What it holds |
|------|---------------|
| `idea.md` | positioning, why this book, the route (第一部分 跑通 / 第二部分 后训练), dataset candidates, prior art |
| `design.md` | the code design: hardware constraints (Colab T4, the author's Intel Mac), the one model config, datasets, `code/` layout, modules, the interface to book 1, the validation table, PoC questions, per-chapter code plan, and its 待定 list |
| `chapters/todo.md` | open manuscript decisions; **not part of the book**, the linters skip it |
| `ref.md` | links |

See also the workspace `../CLAUDE.md`: prose rules, notation, and the shared lint skills apply here.

## Layout

| Path | Role |
|------|------|
| `chapters/_front.md`, `_preface.md`, `chNN_slug.md` | the manuscript, `ch01`–`ch12` |
| `code/` | one uv project (torch pinned to 2.2.2 for the author's Intel Mac; code uses only torch-2.2 APIs so it also runs on Colab's newer torch) |
| `code/book/chNN/` | the book's code, one directory per chapter, each runnable on its own |
| `code/data/`, `code/models/` | corpora and weights, not in git; `code/download_model.sh` fetches SmolLM2-135M, used as the reference answer |
| `draw/chNN.drawio`, `images/chNN/` | diagram sources and exported PNGs; `python3 export_drawio.py chNN` exports one chapter and shrinks the PNGs |

There is no `merge.py` / `Book.md` here yet.

## Chapter status

`ch01`–`ch04` (概览, 数据, 分词, 模型) are drafted with code in `code/book/ch01`–`ch04`. `ch05`–`ch12` (训练循环, 预训练, SFT, DPO, RL, 蒸馏, LoRA, 规模) are still one-paragraph stubs. `design.md`'s 目录结构 also plans `poc/`, `colab/`, `runs/` and `reference/`; none exist yet.

## Conventions specific to this book

- **Notation is book 1's**: Goodfellow, so matrices are **bold uppercase**. The `符号记法/粗体大写` lint rule is off for this book in `../.claude/lint-scope.json`; do not "fix" bold matrices.
- **Never write 对拍**, in chapters, docs, comments or code. The author dislikes the word. Use 核对、核对结果、和参考实现核对、对答案. Book 1 still has 对拍 in places, so do not carry it over when borrowing text or code.
- **Verify against a reference**, as book 1 does: every component is checked against `transformers` / `torch` (logits, loss and gradients to relative error < 1e-5 in fp32, the exact standards are in `design.md`'s 验证 table). `transformers` is a dev dependency only, used in checking scripts, never in the chapters' main code.
- **Self-contained**: code never imports or references book 1's files; copy what is needed. The shared lint skills in `../.claude/skills/` are the one exception.
- **Each chapter builds on book 1** rather than re-teaching it: the architecture chapters point back to the book-1 chapter that explained a component.

## Before committing a chapter

```bash
python3 ../.claude/skills/check-formulas/scripts/check.py chapters/chNN_slug.md
python3 ../.claude/skills/check-prose/scripts/check.py    chapters/chNN_slug.md
```
