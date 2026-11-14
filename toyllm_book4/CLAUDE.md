# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

自己动手写深度学习框架 (title provisional, English working title *Build a Toy PyTorch from Scratch*) — the fourth toyllm book; it opens the black box books 1 and 2 use: `../toyllm_book1/` runs a model on PyTorch's forward pass, `../toyllm_book2/` trains one with PyTorch's backward pass and optimizers, and this one writes that framework, a toy PyTorch called **pytoych**: tensors, autograd, `nn`, `optim`, `data`. The endpoint is a small model with book 1's architecture (RMSNorm, RoPE, SwiGLU, GQA, tied embeddings) trained in pytoych, exported as a model directory, and run in book 1's engine.

**Still a plan.** Read `README.md` first: its 写作原则 are this book's standing rules (one concept per chapter, problem before part, pytoych's API copies PyTorch's so a script runs with `import pytoych as torch`, every operator's forward and gradient is checked against real torch, every chapter ends comparing with PyTorch's actual design while keeping documented facts apart from guesses, pytoych itself depends only on NumPy). Its 技术选择（待定） section lists choices not yet made, and its PoC section (P0 speed, P1 curve alignment, P2 export) lists the questions that must be answered before the part-four chapters are fixed; do not treat any of them as decided. `outline.md` is the plan (five parts, 22 chapters, one appendix), `research.md` the survey of similar material (DeZero, MiniTorch, needle, TinyTorch are the closest; the 本书的位置 section says how this book differs).

See also the workspace `../CLAUDE.md`: prose rules and the shared lint skills apply here. Notation follows the toyllm books (Goodfellow: bold uppercase = matrix), so `../.claude/lint-scope.json` disables `符号记法/粗体大写` for this book.

## Layout and status

`chapters/` holds `_front.md`, `_preface.md`, `ch01`–`ch22`, `afterword.md` and `x_a_grad_table.md`, the same naming as the toyllm books. Every chapter is still a frame: an H1 title, a one-paragraph lead, section headings with at most a line under them, and the fixed tail 代码实现 / 对答案 / 对照PyTorch / 本章小结. There is no `code/`, `draw/`, `images/`, `aigc/`, `merge.py` or site entry yet; when they arrive, follow `../toyllm_book1/`'s layout (`code/book/chNN/`, one self-contained directory per chapter, each built from the previous one; `draw/chNN.drawio` exported by `export_drawio.py`).

## Environment constraints

The author's machine is a 2019 Intel MacBook Pro, so the reference torch is pinned to 2.2.2 with `numpy<2`, as in book 1 (see `../toyllm_book1/code/CLAUDE.md`). torch is a test-only oracle; pytoych must never import it.
