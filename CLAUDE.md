# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Chinese-language vibe-writing workspace: several books in progress, side by side, plus a folder of standalone 图解 articles. Each book is a self-contained sibling directory.

| Path | What | Status | Its own CLAUDE.md |
|------|------|--------|-------------------|
| `llm_notes/` | standalone 图解 articles, `YYYY-MM-DD-Slug.md` | ongoing | — |
| `toyllm_book1/` | 《自己动手写LLM推理引擎》 — hand-write an inference engine for SmolLM2-135M on CPU | drafting, the most mature | ✓ (plus `code/CLAUDE.md`) |
| `toyllm_book2/` | 《自己动手训练LLM》 — sister book: train a small model from scratch, then run it on book 1's engine | drafting, ch01–ch04 | ✓ |
| `toyllm_book3/` | 《自己动手写智能体》 — third toyllm book: build a toy coding agent from scratch | plan, chapters are frames | ✓ |
| `toyllm_book4/` | 《自己动手写深度学习框架》 — fourth toyllm book: write a toy PyTorch (PyToych) on NumPy, checked against real torch, ending with a model book 1's engine can run | plan, chapters are frames | ✓ |
| `ann_book/` | 《人人能懂的人工神经网络》 — neural networks for general readers, told as history from the 1940s neuron to Transformers | drafting | ✓ |
| `gpu_book/` | GPU 学习笔记 — one question: how to make a huge matmul fast | drafting, ch01–ch08 | ✓ |
| `tpu_book/` | 从零设计 TPU — the pair of `gpu_book`: design a chip for matmul from first principles | plan, chapters are outlines | ✓ |

**Every book has its own CLAUDE.md; read it before working on that book.** It holds only what differs from this file (layout, status, book-specific conventions) and points to the book's `README.md`/`outline.md`/`idea.md` for the plan rather than repeating it. This file holds what is true across the whole workspace. A new book gets a CLAUDE.md in the same shape.

Book layout is mostly shared: `chapters/` is the manuscript source, `code/` the examples, `draw/*.drawio` the diagram sources with exported PNGs in `images/`, `aigc/` AI-generated art, and `merge.py` generates a single-file `Book.md` (never edit `Book.md` by hand).

## Git

This directory is the repo root (`github.com/zxh0/vibe-writing`, formerly `ann-book`), and `main` is the only branch; writing is committed straight to it. Keep commits scoped to the area being worked on, with the 【区域】 prefix the history uses (【仓库】, 【toyllm】, 【站点】…).

## Site

One VitePress site for the whole workspace, published at https://zxh0.github.io/vibe-writing/ by `.github/workflows/deploy.yml` on every push to `main` (GitHub Pages, source = GitHub Actions). The homepage lists the notes and every book side by side; `/notes` is the notes, `/<route>` each book's own 进度和目录 page, `/<route>/tools` a book's 小工具 if it has any. Locally:

```bash
npm ci                   # once
npm run dev              # prepare:site + vitepress dev
npm run build            # prepare:site + vitepress build -> docs/.vitepress/dist
```

`scripts/build-site.mjs` is the only content pipeline; its `NOTES` and `BOOKS` tables are the single place that maps a repo directory to a route (`ann_book` → `/ann`, `toyllm_book1` → `/llm-engine`, `toyllm_book2` → `/llm-train`, `toyllm_book3` → `/agent`, `toyllm_book4` → `/framework`, `gpu_book` → `/gpu`, `tpu_book` → `/tpu`), with each book's title, status, blurb, cover and `published` list. A book's English subtitle is read from the `> …` line under the title in its `chapters/_front.md` (the homepage card and the landing page's `<BookSubtitle />` both show it), so a title change is made there only; `subtitle` is set in `BOOKS` only for books whose title page has no English title (gpu, tpu, ann). Hand-written site files are `docs/index.md` (homepage), one landing page per area (`docs/notes.md`, `docs/ann.md`, `docs/llm-engine.md`, …) and `docs/.vitepress/` (config, theme, the `AreaCards`/`BookToc`/`NoteList`/`ToolCards` components). Everything else under `docs/` (`docs/<repo dir>/`, `docs/public/`, `sidebar.json`) is regenerated on every run and gitignored.

- **Notes publish automatically**: every `llm_notes/YYYY-MM-DD-Slug.md` gets a page; the sidebar runs oldest-first, the list page newest-first.
- **Book chapters go up one at a time**: `published` in a book's `BOOKS` entry lists the chapter files that get a page; the rest only show as grey titles on the book's landing page. Publishing a chapter means adding its filename there. Chapter files are `chNN_*.md` plus `afterword.md`, `appendix.md`, `x_?_*.md` (`_front.md`, `_preface.md`, `todo.md` never publish); `ann_book` is the exception with `NN_slug.md` names and `##` chapter titles that get promoted one level.
- **Tools** (`<book>/tools/<dir>/index.html`) are copied verbatim to `/<route>/tools/<dir>/` and listed on `/<route>/tools`; the `tools` map in the `BOOKS` entry gives each a display title, the blurb is the first paragraph of its README. Adding a tool = a directory plus one line in that map.
- Generated page files keep the repo path (`docs/ann_book/chapters/03_ch01_neuron.md`) and are mapped to clean URLs (`/ann/ch01-neuron`) through VitePress `rewrites`, so the edit link and the last-updated lookup (`transformPageData` in `config.mts`, since generated files have no git history of their own) resolve straight back to the source file.
- `<img src="../images/chNN/x.png" width="N">` tags become Markdown images under `/<route>/images/` with the width carried through (`style="zoom:50%"` is dropped); `../aigc/` likewise under `/<route>/aigc/`; notes' `./images/` under `/notes/images/`.

## llm_notes/

Standalone deep-dive articles, published separately from the books. One Markdown file per article, named `YYYY-MM-DD-Slug.md` by publication date, with draw.io sources in `llm_notes/draw/`, exported assets in `llm_notes/images/<slug>/` and the series' AI-generated cover in `llm_notes/aigc/front.png` (every area keeps its AI art in its own `aigc/`). `llm_notes/README.md` lists them all, oldest first, and is the index to keep in sync when adding one; the workspace `README.md` links only to the folder.

These are **hand-written by the author and deliberately not AI-polished**. When editing them, limit changes to what was asked (e.g. a typo fix) and preserve the author's voice.

## Notation Conventions

The default, used by `ann_book` and every book that does not say otherwise:

- Scalars: lowercase — $x$, $w$, $b$, $\eta$
- Vectors: **bold** lowercase — `\mathbf{x}`, `\mathbf{h}`, `\mathbf{b}` (Greek: `\boldsymbol{\alpha}`, since `\mathbf` has no effect on Greek)
- Matrices: uppercase, never bold — $W$, $Q$, $K$, $V$
- Layer indices: superscripts

Bold is reserved for vectors, so bold uppercase (`\mathbf{W}`) and arrow vectors (`\vec{x}`) are both wrong. Enforced by the `check-prose` rule 符号记法统一.

Exceptions: `toyllm_book1`, `toyllm_book2`, `toyllm_book4` and `gpu_book` follow Goodfellow (bold lowercase = vector, **bold uppercase = matrix**, plain = scalar), and `llm_notes` follows each paper's own notation. Both are scopes in `.claude/lint-scope.json`, see below.

## Content Language

Prose is **Simplified Chinese**; code, identifiers, filenames, and diagram names are English. No space between CJK and Latin/digits (`ReLU函数`, not `ReLU 函数`); full-width Chinese quotes `“”`, never `""`. The author dislikes em-dashes (`——`) — use a comma, a colon, or split the sentence.

The author's standing preferences for book prose, given while revising toyllm and gpu_book chapters, apply to every book (not to the hand-written `llm_notes/`):

- **Never open a paragraph with non-Chinese text**: no code span, English word, acronym (GPU, T4) or digit first. Lead with a Chinese word: 在`engine.py`里…, 所谓occupancy，…, 在GPU上，…, 每个block申请16KiB时…. Terms that read as Chinese (KV缓存…) are fine. Before finishing an edit, list the openers outside code blocks, e.g. ``awk '/^```/{c=!c;next} !c && /^[A-Za-z0-9`(]/' chapters/chNN_*.md``, and reword them (ignore `$$` lines, tables, images and headings).
- **No very short paragraphs**, especially a lone line like “下面是输出：”. Merge it into a neighbour, fold two adjacent display formulas into one `aligned` block so the bridge sentence disappears, or give the lead-in real content (what the script does, what to look for).
- **The author's editor drops the blank line before a `$$` block** that follows a line ending in “：”. That is a side effect, not intent: after the author edits, run `check-formulas`, put the blank line back, and mention it in one line.

## Diagrams

The author opens `draw/*.drawio` in draw.io and re-saves them, which reformats the XML and may carry hand tweaks. For a small change (a label, a unit), patch the committed file in place (string replacement on the `value` attributes) and re-export with `python3 export_drawio.py chNN`. Never regenerate a whole saved file from a script: that once produced a 2000-line diff and would silently drop manual edits. Generate from scratch only for a brand-new page or file.

## Custom Skills

One set of skills in `.claude/skills/`, shared by every book and the notes; no book keeps its own copy (all areas share one style and one layout). Each linter takes files, directories, or several paths at once, recurses into directories, and skips dot-dirs and the generated `Book.md`.

```bash
python3 .claude/skills/check-formulas/scripts/check.py ann_book     # LaTeX formatting for GitHub rendering
python3 .claude/skills/check-prose/scripts/check.py ann_book        # Chinese body text + toy_*.py examples
uv run --no-project .claude/skills/shrink-images/scripts/shrink.py --dry-run ann_book/images
```

From inside a book directory the same scripts are `../.claude/skills/...`. The books' `export_drawio.py` call `shrink-images` from there too.

`check-formulas` rules: inline `$...$` needs surrounding spaces, blank lines around block formulas, `\begin{aligned}` for multi-line, `\tag{}` after `\end{aligned}`, no `\operatorname`, multi-letter function names in `\mathrm{}`.

`check-prose` rules: 符号记法统一 (notation), 尽量不用破折号 (em-dash), 公式用 `$`/代码用反引号 (math-vs-code), 论文引用 (a paper's first mention in the book carries `（arXiv:编号）`, later ones don't; "first" follows chapter-file order), 图片宽度 (`<img width="半个像素宽度">`, never `style="zoom:50%"`), 代码块宽度 (≤80 columns, CJK counts as two, for print), 玩具代码规范 (toy-code, applies to `toy_*.py` by filename), plus a semantic check that a chapter's code blocks match the `.py` file verbatim.

`polish-prose` turns a drafted chapter into finished text. Its workflow and red lines apply to every book; its 术语约定/符号约定 sections are the toyllm series' own.

`shrink-images` compresses PNGs in place: lossless flatten + metadata strip, then an 8-bit palette only for line art (draw.io exports). It needs Pillow, declared inline (PEP 723) so `uv run` builds the env; the system `python3` is 3.9 with no Pillow. Details in its `SKILL.md`.

Each `check.py` has a `fix.py` beside it that auto-fixes the mechanical rules. Both take `--dry-run`. **Always `--dry-run` first on `llm_notes/`**, those are hand-written. Run both linters after editing any chapter, before committing.

### Per-area exemptions

Where a book's style really differs, the difference is a scope in `.claude/lint-scope.json`, never a forked script (rules dispatch on file type, not path):

| Scope | Disabled | Why |
|-------|----------|-----|
| `llm_notes/**` | 符号记法, 图片宽度, 论文引用 | hand-written: each paper's own notation, images still use `zoom`, citation style is a book convention |
| `ann_book/**` | 图片宽度 | images still use `zoom`; drop once migrated |
| `toyllm_book1/**`, `toyllm_book2/**`, `toyllm_book4/**`, `gpu_book/**` | 符号记法/粗体大写 | Goodfellow notation, matrices are bold uppercase |

`exclude` (`CLAUDE.md`, `**/todo.md`, …) lists files not checked at all; a file inherits the union of every matching scope's `disable` list. Exempted counts are always reported (“另有 N 处被 .claude/lint-scope.json 豁免”), never silently dropped. `--all-rules` puts the scope exemptions back for one run; `exclude` still applies.

The config is found by walking up from each file to the nearest `.claude/lint-scope.json`, so **do not add a `.claude/` inside a book**: it would silently shadow the root config for that book.
