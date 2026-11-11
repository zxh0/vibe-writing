# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

自己动手写智能体 (title provisional, English working title *Build a Toy Coding Agent from Scratch*) — the third book of the toy series after `../toyllm_book1/` (how a model runs) and `../toyllm_book2/` (how it is trained): this one is how a model is *used*. Chapter 1 gives a working agent loop of a few dozen lines; every later chapter adds one thing (tool calls, permissions, a terminal UI, context compaction, retrieval, MCP, skills, subagents, hooks, evaluation) and stays runnable.

**Still a plan.** Read `README.md` first: its 写作原则 are this book's standing rules (one concept per chapter, problem before part, no agent frameworks and minimal SDKs, raw HTTP/SSE and ANSI escape codes before any library, compare each chapter with Claude Code, Codex, Pi and mini-swe-agent while keeping published facts apart from guesses, test the loop by replaying recorded replies). Its 技术选择（待定） section lists choices not yet made; do not treat them as decided. `outline.md` is the plan (five parts, 21 chapters), `research.md` the survey of similar material, `ref.md` links.

See also the workspace `../CLAUDE.md`: prose rules, notation, and the shared lint skills apply here.

## Layout and status

`chapters/` holds `_front.md`, `_preface.md`, `ch01`–`ch21` and `afterword.md`, the same naming as the toyllm books. Every chapter is still a frame: a title and a line or two of key points. There is no `code/`, `draw/`, `images/` or `merge.py` yet; when they arrive, follow the toyllm books: `code/book/chNN/`, one self-contained directory per chapter, each built from the previous one.
