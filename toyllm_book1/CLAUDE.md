# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Chinese-language book in progress, 《自己动手写LLM推理引擎》 (title settled 2026-11-13, no subtitle; the preface explains that 推理 here is inference, and the book calls reasoning 思考) — teaching how LLM **inference** works by hand-writing an engine for it. Target model is SmolLM2-135M on PyTorch 2.2 / CPU.

The thesis, stated in `chapters/_preface.md`: 从训练入手会比较难，从推理入手就比较简单；从 GPU 入手会比较难，从 CPU 入手就比较简单. Everything follows from that — the model choice, the hardware, the chapter order.

Three standing conventions from the preface that the manuscript must honour:

1. **No backpropagation.** The target reader is 略懂 AI/DL but 害怕微积分和反向传播. Anything whose real explanation lives in gradients gets the result only, plus its consequence for inference code — never the derivation. (E.g. pre-norm vs post-norm is argued from "is there a rescale on the trunk", not from gradient flow.)
2. **Every chapter diffs against `transformers`.** Chapter 1 dumps a baseline; every later chapter compares to it. The strictness varies deliberately: `==` on ints (ch03) → bit-identical (ch05) → relative error < 1e-5 (ch07 onward, once RoPE's cos table introduces a 1-ulp gap) → back to `torch.equal` (ch10, because argmax chains amplify).
3. **Shapes are the subject.** Most bugs here are shape bugs, not math bugs, and they fail silently. Chapters spell out `(T, 576) → (T, 9, 64) → (T, 576)` at every transformation.

## Layout

| Path | Role |
|------|------|
| `chapters/_front.md` | 扉页 — title, English subtitle, version stamp, cover image |
| `chapters/_preface.md` | 前言 — motivation, target reader, reading conventions, tech choices, TOC. Also absorbed the old `ch00.md` (准备). Sorts before `ch01` thanks to the `_` prefix. |
| `chapters/chNN_slug.md` | manuscript source, one file per chapter, `chNN` matches 第N章. Fourteen chapters, `ch01`–`ch14`. |
| `chapters/afterword.md`, `chapters/x_X_slug.md` | 后记 (not written yet) and the appendices (`X` = one lowercase letter: `x_a_matrix` 矩阵公式, `x_b_formulas` 引擎公式汇总, `x_c_papers` 论文列表, `x_d_glossary` 术语表; reviewed with the author 2026-11-13: A and B put the text before each formula, B's symbol table has a Qwen3-0.6B column, D is ordered by chapter, not pinyin). The `x_` prefix makes the appendices sort after the chapters by filename too; `merge.py` still orders explicitly, because `afterword` sorts before `ch`. `check-prose`'s 论文引用 rule counts the afterword as part of the book order and skips appendices, since a paper list repeats every arXiv number by design |
| `images/chNN/` | exported PNGs, referenced from chapters as `../images/chNN/...` inside `<img ... width="N" />` tags, where N is half the PNG's pixel width (every image is 2x: Retina screenshots, and `export_drawio.py` exports at `SCALE = 2`). Never `style="zoom:50%"`: GitHub strips `style` and `zoom` is non-standard CSS. `check-prose` checks the width (rule `图片宽度`) and its `fix.py` rewrites it after an image is re-exported |
| `draw/chNN.drawio` | draw.io sources for those PNGs. A plotted figure (a function curve, not a diagram) has no drawio page: its source is a `plot_*.py` in that chapter's code directory, e.g. `code/book/ch08/plot_af.py` → `images/ch08/af.png`. A plot of measured timings splits measuring from drawing: `code/book/ch11/speed.py` (project env) writes `speed.json`, and `plot_speed.py` (PEP 723) draws `images/ch11/speed.png` from it, so the figure and the numbers ch11 quotes come from one run; re-running `speed.py` means re-quoting every timing in ch11 (and the 12.7GB/s echoed in ch12–ch14). matplotlib is not a project dependency; the script declares it with PEP 723, so run it as `uv run book/ch08/plot_af.py` (no `python`). A plot that needs the model (real logits) cannot use PEP 723, which drops the project's torch; `code/book/ch10/plot_sampling.py` → `images/ch10/filters.png`, `temperature.png` runs in the project env instead, as `uv run --with matplotlib --with "numpy<2" python book/ch10/plot_sampling.py` (without the numpy pin, the overlay pulls numpy 2 and torch 2.2 breaks) |
| `Book.md` | **generated** single-file manuscript; never edit by hand — see `merge.py` below. Gitignored (root `.gitignore` lists `Book.md`): regenerate it locally to check the merge, it never enters a commit |
| `TODO.md` | the book's open decisions and to-dos, the only place they are recorded; **not part of the book**, deleted before publication. Remove an item once it is done rather than striking it through |
| `reviews/` | read-through reviews, one dated file each: `2026-11-14-Claude通读意见.md` (Claude's full read-through of the finished manuscript; it superseded the 2026-11-01 file, which was deleted, and keeps those old ids in parentheses where an item is still open), `2026-11-04-审读意见.md` and `2026-11-04-市场定位.md` (an independent review) |
| `merge.py` | the generator: `python3 merge.py` (regenerate), `--check` (exit 1 if `Book.md` is out of sync with `chapters/`, version stamp ignored), `-o` (write elsewhere) |
| `code/` | one uv project holding two code trees — see `code/CLAUDE.md` |
| `code/book/` | the book's code: one directory per chapter, `ch01/`…`ch14/`, all written. **New work goes here.** Like ch09, `ch12/` adds no engine code: its engine files are byte-identical copies of `ch11/`, plus `online_softmax.py` and `flash_attn.py` (a pure-PyTorch tiled attention; on CPU it is *faster* than the one-shot version at block sizes 128–256, because it skips future blocks and fits in cache). `ch14/` adds `device` / `dtype` / `attn_impl` to `Model`; with the defaults it is bit-identical to `ch13/`. The author's machine has no CUDA: ch14's Mac numbers come from CPU and MPS (torch 2.2's MPS has no bf16), and its CUDA numbers from Colab's free T4, produced by `ch14/colab.ipynb` (it sparse-clones the public mirror https://github.com/zxh0/vibe-writing, so changes must be synced there first) with the raw output kept in `ch14/t4/`. The T4 (sm75) has no native bf16, so its speed and batching runs use fp16, and PyTorch's FlashAttention kernel needs sm80+, so on the T4 `F.scaled_dot_product_attention()` picks memory-efficient attention; the prose says so. `attend_sdpa` adds a batch dim before the call: the fused kernels only take 4-D input, and a 3-D `(9, T, 64)` silently falls back to the math kernel that builds the whole T×T table (no error, same result, only slower; ch14 tells this story). ch09 adds no engine code: its four engine files are byte-identical copies of `ch08/` (an untouched file keeps the header of the chapter that last changed it, as `weights.py` does everywhere), plus three inspection scripts (`residual_stream.py`, `logit_lens.py`, `sinkhorn.py`), so `ch10/` still builds on `ch08/`. In `ch11/`, `NO_CACHE` must reproduce ch10 bit for bit, and the engine feeds `ids[cache.length:]` instead of branching on the cache. |
| `code/poc/` | the original proof of concept. **Frozen reference, not the book's code.** |
| `code/models/`, `code/reference/` | weights and baseline tensors, shared by both trees, gitignored |
| `tools/` | standalone HTML tools the chapters point readers to (`sinpe-heatmap/`, `rope-angles/`, `safetensors-viewer/`, `embed-dims/`, `param-cubes/`): one `index.html` each, no build step, plus a `README.md` |
| `export_drawio.py` | exports one chapter's `draw/chNN.drawio` page by page into `images/chNN/`, then shrinks the PNGs |

**The skills and lint config are shared, not copied.** `check-formulas`, `check-prose`, `polish-prose` and `shrink-images` live in the workspace root's `.claude/skills/`, and this book's exemption (`符号记法/粗体大写`, for the Goodfellow notation) is a scope in the root `.claude/lint-scope.json`. Apart from that, keep the book self-contained: its code, images and tools never import or reference a sibling book.

`merge.py` concatenates `_front.md`, `_preface.md`, `chNN_*.md`, `afterword.md` and `x_X_*.md`, in that order and by filename within each group (any other name is skipped), and, on the way through: demotes every heading one level outside code fences (`#` → `##`, so the book title in `_front.md` stays the only H1), inserts the page-break `<div>` between chapters, rewrites every relative asset path `../` → `./`, inserts the 自动生成 banner under the H1, and stamps `版本：vYYYY.MM.DD` with today's date. So **chapter files keep their H1 chapter titles, must not contain the page-break div or the banner**, and asset paths in `chapters/` are always written `../images/...` / `../aigc/...`. Run it after editing any chapter; the linters skip `Book.md` by name. The table of contents lives in `_preface.md` under 本书结构; chapter order is otherwise carried by the filenames.

**Read `code/CLAUDE.md` before writing any code.** It records the environment pins, the verified SmolLM2 config values, and the tokenizer / RoPE / KV-cache gotchas that were found the hard way. Reuse those findings rather than rediscovering them — but do not copy `poc/`'s file layout or API, since `book/` is deliberately a different take (one self-contained directory per chapter). Extending `poc/` is not the task.

## Chapters

Order follows the data flow of one forward pass, split into two parts.

**跑通 (1–10)** — walk the pipeline left to right, one component per chapter:

| Ch | Topic | PoC reference |
|---|---|---|
| ch01 | LLM推理引擎概览 — run it with `transformers` first, dump the baseline | `poc/steps/00_reference.py` |
| ch02 | 模型文件与权重 | `poc/toyllm/config.py`, `weights.py`, `poc/steps/01_load_and_inspect.py` |
| ch03 | 分词（Tokenization） | `poc/toyllm/tokenizer.py`, `poc/steps/02_tokenizer.py` |
| ch04 | 词嵌入（Embedding） — one matrix read both ways | `poc/toyllm/weights.py`, `model.py` |
| ch05 | 归一化（Normalization） — RMSNorm, pre-norm vs post-norm | `layers.RMSNorm`, `poc/steps/03_rmsnorm.py` |
| ch06 | 注意力机制（Attention） — GQA + causal mask | `layers.Attention`, `poc/steps/06_attention.py` |
| ch07 | 位置编码（Positional Encoding） — classic PE, then RoPE | `layers.RotaryEmbedding`, `poc/steps/05_rope.py` |
| ch08 | 前馈网络（FFN） — SwiGLU, with a brief look at MoE | `layers.SwiGLU`, `poc/steps/04_swiglu.py` |
| ch09 | 残差连接 — wraps both sublayers; the Block is complete here | `poc/toyllm/model.py`, `poc/steps/07_forward.py` |
| ch10 | 采样（Sampling） — plus the generate loop | `poc/toyllm/sampling.py`, `poc/steps/09_sampling.py` |

**跑快 (11–12)** — same data flow, different execution:

| Ch | Topic | PoC reference |
|---|---|---|
| ch11 | KV缓存（KV Cache） — prefill / decode | `layers.KVCache` / `NoCache`, `poc/steps/08_kv_cache.py` |
| ch12 | FlashAttention — online softmax + tiling | none (`code/book/ch12/flash_attn.py`) |

**走出去 (13–14)** — same engine, another model, another device:

| Ch | Topic | PoC reference |
|---|---|---|
| ch13 | 换一个模型 — Qwen3-0.6B, chapter-by-chapter review of what changes; `--model` picks the directory under `code/models/` | none |
| ch14 | GPU — device, half precision, batching, FlashAttention measured; CUDA numbers from a Colab T4 | none |

Note the chapter order **differs from the order the PoC was built in**, so a chapter's material usually lives in a differently-numbered step script. Three chapters have no single PoC counterpart: ch04 (the embedding/lm_head pair was never its own step), ch09 (the PoC never separated residuals from the Block), and ch12.

Chapters used to open with a `> **本章代码**` scaffolding blockquote (what to build, shapes, what to diff against, known traps). Every chapter has had it removed; do not reintroduce one.

The diffing criteria live only in the code and in this file. **The prose never mentions the baseline file or the tolerance table.** When a chapter needs to refer to the reference output, it points at something the reader has actually seen, e.g. 「和第一章那句话一字不差」 rather than 「和第一章那份基准逐字节相同」.

**When writing a chapter, run its code first and quote real output.** The numbers in ch02 (272 tensors, 269.1 MB, 134,515,008 params, 3,540,096 per block) come from `poc/steps/01_load_and_inspect.py` and must keep matching it.

## Commands

Everything runnable lives under `code/`, and must go through `uv run` (system `python3` is 3.9 with no torch):

```bash
cd code
uv sync --group dev                                # .venv + deps + pytest oracles
uv run python book/ch01/hello_5l.py              # run one chapter's code (book/ is where new work goes)
uv run python poc/steps/07_forward.py              # run one PoC milestone script
uv run pytest                                      # all tests (testpaths = poc/tests)
uv run pytest poc/tests/test_rope.py -x            # one test file
```

Downloads (weights, `uv sync`) must bypass Claude Code's proxy — prefix with
`env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy`. See `code/CLAUDE.md`.

## Writing conventions

- **Language and punctuation.** Prose is Simplified Chinese; code, identifiers, filenames and diagram names are English. No space between CJK and Latin/digits (`SmolLM2模型`, not `SmolLM2 模型`). Full-width Chinese quotes `“”`, never `""`. No em-dashes (`——`): use a comma, a colon, or split the sentence.
- **Notation: matrices are bold uppercase.** This book follows Goodfellow's scheme: bold lowercase = vector ($\mathbf{q}_m$, $\mathbf{x}$), **bold uppercase = matrix** ($\mathbf{Q}$, $\mathbf{W}_Q$, $\mathbf{R}_m$), unbolded = a plain count or scalar ($d$, $T$, $m\theta_i$). ch06 and ch07 are written this way; do not "fix" bold matrices to plain. The root `../.claude/lint-scope.json` disables `符号记法/粗体大写` for that reason. Greek vectors still need `\boldsymbol`, and `\vec{}` is still wrong.

- **Em-dashes are cleared**: the whole manuscript has no `——` left. Do not introduce new ones.
- **Never write 对拍 in prose.** The author dislikes the word; the chapters say 核对 (核对结果, 和`transformers`核对). Code comments and the per-chapter READMEs under `code/book/` still say 对拍 in places; leave them unless asked, since a sweep touches every chapter's copies.
- **The model is always `SmolLM2`, never `SmolLM`.** This typo has recurred several times; grep for `SmolLM[^2]` after editing.
- **Large integers fall into three classes**, and ch02/ch03 already follow this:
  - *Identifiers* never take a separator, because they say *which one*, not *how many*: token ids (46119), line numbers (第49347行), merge ranks (45867), code points (`U+0120`).
  - *Spec values* a reader can look up in `config.json` / `tokenizer.json` stay bare, so prose and file read the same: 49152, 576, 8192, 30.
  - *Quantities* take a comma every three digits from four digits up: 134,515,008 params, 269,060,552 bytes, 30,536 bytes, 35,136 params.
  - One exception: a quantity being reconciled against a spec value in the same passage follows the spec value and stays bare. ch03's 48900 merge rules sits inside `49152 - 17 - 235 = 48900`, so both stay bare rather than mixing `49,152` with `48,900`.
  - Numbers inside code blocks, JSON excerpts and quoted script output are verbatim and never reformatted.
- **Paper citations**: the first time a paper appears in the book (file-name order: `_preface`, `ch01`, …) it is written 《Title》（arXiv:NNNN.NNNNN）, with the title exactly as on arXiv; every later mention, in any chapter, is just 《Title》. A paper mentioned only by author and year gets its title added. `check-prose` enforces both halves (rule `论文引用`).
- Bold marks a term's first formal definition, given as **中文**（English）: **词元**（Token）, **前馈网络**（Feed-Forward Network，简称FFN）. Sampling parameters keep their conventional lowercase-hyphenated spelling in prose (`top-k`, `top-p`) and snake_case in code (`top_k`, `top_p`).
- Chapter titles are 中文（English）where a standard English term exists — 分词（Tokenization）, 归一化（RMSNorm） — and plain English only where no settled Chinese term exists (FlashAttention). **KV Cache is written 「KV缓存」 everywhere in prose** (author's decision, 2026-11-01), including ch11's title; the English stays only in its first definition, **KV缓存**（KV Cache）, in the glossary's English column, in code (`KVCache`, `kv_cache.py`), and in the ASCII engine diagrams at the end of chapters. **Those diagrams are English only** (ids → embed → … → sampler → id, `append to`, `KV Cache`): CJK characters are double-width and break the box alignment, so they never carry Chinese labels.
- Chapters end with a `## 本章小结` section (小结 = summary; 小节 would mean subsection — the whole book was corrected from that typo, so do not reintroduce it).
- Chapter files may still hold **raw pasted terminal or AI output as scratch material**, and `_preface.md` keeps its open questions in a trailing HTML comment. Do not assume everything in a chapter file is intended final text.
- The preface is the author's own voice, first person. Preserve their wording when editing it; add rather than rewrite.

### Linters

Both live in the workspace root's `.claude/skills/` and read their per-file rule scoping from the root `.claude/lint-scope.json`. Run them from the book root (this directory):

```bash
python3 ../.claude/skills/check-formulas/scripts/check.py chapters/chNN_slug.md
python3 ../.claude/skills/check-prose/scripts/check.py    chapters/chNN_slug.md
```

Both accept a directory (`chapters/`) or the whole repo, and both have a `fix.py` beside them that takes `--dry-run`. Run `check-formulas` after editing any chapter — inline `$...$` needs surrounding spaces, and it is easy to miss by hand. `check-prose` is clean on `chapters/` as of 2026-11-13, 代码块宽度 included. Keep it that way: quoted code is verbatim, so shorten the `.py` (and every chapter's identical copy) and re-quote; for quoted script output, change the script's print format and re-run, never edit the output by hand. Scripts that print long lists or strings use small `fold()` / `fold_text()` helpers (comma- and space-aware wrapping at 78 columns), and tables with Chinese headers pad by display width (`ljust`/`rjust`), as in ch09 and ch14. Its toy-code rules dispatch on the `toy_*.py` filename, so they do not touch `code/`.
