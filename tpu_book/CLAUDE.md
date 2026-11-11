# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

从零设计 TPU (title provisional) — the pair of `../gpu_book/`. That book asks how to write matmul fast on an existing general-purpose chip; this one asks what a chip built *for* matmul should look like, and reinvents the TPU one design decision at a time, from a single MAC to systolic arrays, TPU v1, training support (v2/v3) and pod-scale interconnect.

**Still a plan.** Read `README.md` first: its 写作原则 are this book's standing rules (every structure is motivated by a problem, every decision comes with a back-of-envelope cost, a cycle-level Python simulator grows through the book, published facts are kept apart from the book's own guesses). `outline.md` is the plan (five parts, 19 chapters plus an appendix), `research.md` the survey of similar material.

See also the workspace `../CLAUDE.md`: prose rules, notation, and the shared lint skills apply here.

## Layout and status

`chapters/chNN_slug.md` (`ch01`–`ch19`) and `appendix.md` hold only per-chapter outlines so far, each opening with 本章要回答的问题. There is no `code/`, `draw/`, `images/`, `merge.py` or `_front.md` yet; when they arrive, follow `gpu_book`'s layout.

The outlines were written in an older style: spaces between CJK and Latin, 第 N 章 in the H1, a few em-dashes. Bring a chapter in line with `../CLAUDE.md` when it is turned into prose, not before.
