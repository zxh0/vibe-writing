# 自己动手训练LLM

> Train a Toy LLM (from Scratch)

::: warning 构思和初稿
书名待定，还在构思和写初稿，目前没有发布的章节。草稿在仓库的
[toyllm_book2/chapters/](https://github.com/zxh0/vibe-writing/tree/main/toyllm_book2/chapters) 下，构思见
[idea.md](https://github.com/zxh0/vibe-writing/blob/main/toyllm_book2/idea.md)。
:::

![自己动手训练LLM](/llm-train/cover.png)

## 这本书想做什么

这是《自己动手写LLM推理引擎》的姊妹篇。第一本手写引擎，跑别人训好的模型；这一本从零训练一个自己的模型，最后再用第一本的引擎把它跑起来。两本书首尾相接。

第一本的主张是「从推理入手，然后再去学训练」。这一本就是那个「然后」。方法和第一本一样：一点一点来，每一步都看得见、对得上。

「从零训练一个中文小模型」本身已经不新，MiniMind 和 nanochat 都把代码给全了，缺的是一本中文书：把每一步讲清楚、讲出为什么，而且只用免费的 Colab 就能从头复现。这本书想占三个位置：

- **和第一本首尾相接。** 架构沿用 SmolLM2，只缩小尺寸；训练出的模型存成同样的 `config.json` 加 `model.safetensors`，第一本的引擎不改一行就能加载。
- **每个选择都做对照实验。** 同一个小模型少训一些 token，在 Colab 上一组只要十几到几十分钟，正好把「为什么」跑出来给读者看：不做 warmup 会怎样，词表 6k 对比 32k，有没有权重共享，数据清洗前后的差别。
- **验证。** 延续第一本对答案的做法：先用 SmolLM2 的权重把 loss 和梯度对上，再开始训练。

## 目录

<BookToc book="llm-train" />
