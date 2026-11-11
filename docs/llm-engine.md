# 自己动手写LLM推理引擎

> Build a Toy LLM Inference Engine (from Scratch)

::: warning 施工中（WIP）
书名待定，正文还在写。写完一章发一章，目录里能点开的就是已经发布的。还没发布的章节可以直接读仓库里的
[toyllm_book1/chapters/](https://github.com/zxh0/vibe-writing/tree/main/toyllm_book1/chapters)，不过那些还是草稿，可能有很多逻辑混乱、甚至胡言乱语的地方。
:::

![自己动手写LLM推理引擎](/llm-engine/cover.png)

## 这本书想做什么

用 Python 和 PyTorch，从零手写一个能跑的 LLM 推理引擎：加载真实的模型权重，把一段文字变成词元（Token），算完三十层 Transformer，采样出下一个词，再把它变回文字，然后循环。

全程不使用 `transformers` 的建模代码。解析文件这两件杂活，分词和读权重，交给 HF 的 `tokenizers` 和 `safetensors` 两个库；其余的 RMSNorm、RoPE、GQA、SwiGLU、KV 缓存、采样、生成循环，全部自己写。目标模型是 **SmolLM2-135M**，跑在一台**普通笔记本的 CPU** 上。慢，但每一个数字都能和 HuggingFace 对上。

主张很简单：

> 从训练入手，会比较难；从推理入手，就比较简单，然后再去学训练。
>
> 从 GPU 入手，会比较难；从 CPU 入手，就比较简单，然后再去学 GPU。

「从零手写理解 LLM」这个赛道其实很挤，但已有的资料几乎全是**训练视角**：搭一个模型，然后训练它。这本书换一个视角：只讲推理，于是梯度、数据、漫长的等待全都不用管，只剩下模型结构这一件事。而且每一步都可验证，自己写的每个部件，当场就能和 `transformers` 对答案。

## 目录

十四章，分成三段，最后还有附录。已经发布的章节可以点开读：

**跑通**：沿数据流从左走到右，每章造一个部件。

<BookToc book="llm-engine" :from="0" :to="10" />

**跑快**：不改变数据流，改变执行方式。

<BookToc book="llm-engine" :from="10" :to="12" />

**走出去**：同一个引擎，换一个模型，换一种硬件。

<BookToc book="llm-engine" :from="12" :to="14" />

**附录**

<BookToc book="llm-engine" :from="14" />

## 随书代码

随书代码在仓库的 [toyllm_book1/code/](https://github.com/zxh0/vibe-writing/tree/main/toyllm_book1/code) 下，一章一个目录。另外 `code/poc/` 里是最早的概念验证，已经跑通全流程并有一批测试，作为正确性参照保留。

## 小工具

写书时顺手做的几个纯前端单页工具，打开就能用，数据不出浏览器。见[小工具](/llm-engine/tools)页。
