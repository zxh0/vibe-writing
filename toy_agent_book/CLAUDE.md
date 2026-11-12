# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

自己动手写智能体 (title provisional, English working title *Build a Toy Coding Agent from Scratch*) — the third book of the toy series after `../toyllm_book1/` (how a model runs) and `../toyllm_book2/` (how it is trained): this one is how a model is *used*. Chapter 1 gives a working agent loop of a few dozen lines; every later chapter adds one thing (tool calls, permissions, a terminal UI, the system prompt, context compaction, retrieval, MCP, skills, subagents, hooks, evaluation) and stays runnable.

**Still a plan.** Read `README.md` first: its 写作原则 are this book's standing rules (one concept per chapter, problem before part, no agent frameworks and minimal SDKs, raw HTTP/SSE and ANSI escape codes before any library, compare each chapter with Claude Code, Codex, Pi and mini-swe-agent while keeping published facts apart from guesses, test the loop by replaying recorded replies). Its 技术选择（待定） section lists choices not yet made; do not treat them as decided. `outline.md` is the plan (five parts, 22 chapters), `research.md` the survey of similar material, `ref.md` links.

See also the workspace `../CLAUDE.md`: prose rules, notation, and the shared lint skills apply here.

## Layout and status

`chapters/` holds `_front.md`, `_preface.md`, `ch01`–`ch22` and `afterword.md`, the same naming as the toyllm books. Chapters 1 and 2 have first drafts, with code in `code/book/ch01/` and `code/book/ch02/` (stdlib only, an OpenAI-compatible Chat Completions endpoint set by `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `MODEL`; the book's model is DeepSeek V4 Flash). Everything that needs a real DeepSeek response is still marked （待补） or （待核实）: the author's network blocks `api.deepseek.com` from the machine Claude runs on, so those runs are the author's to do. Never invent that output. `ch02/chat_template.py` uses the Qwen3 tokenizer and its output in the chapter is real. Every other chapter is still a frame. Diagrams follow the toyllm books: `draw/chNN.drawio`, exported by `python3 export_drawio.py chNN` into `images/chNN/` (gitignored repo-wide). There is no `merge.py` yet. Code follows the toyllm books: `code/book/chNN/`, one self-contained directory per chapter, each built from the previous one.
