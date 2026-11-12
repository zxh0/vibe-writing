# 自己动手写智能体

> Build a Toy Coding Agent from Scratch

::: warning 写作计划
书名待定，目前还是写作计划，各章只有框架，没有发布的章节。大纲在仓库的
[toy_agent_book/](https://github.com/zxh0/vibe-writing/tree/main/toy_agent_book) 下。
:::

![自己动手写智能体](/agent/cover.png)

## 这本书想做什么

平时用 Coding Agent 很多，大致也知道它是怎么工作的：一个循环，模型说要调什么工具，程序就去调，结果再交回给模型。但「大致知道」和「亲手写过」差得很远。就像 toy 系列的前两本：知道 Transformer 长什么样，不等于能写出一个推理引擎。所以这本书还是同一个办法，自己从零写一个玩具版的 Coding Agent，一点一点来，每一步都看得见、跑得通。

它和别的「从零写 Agent」材料最大的不同，是能从模型内部往外讲：读过第一本的人知道模型只会预测下一个 token，那工具调用、上下文窗口、prompt caching 在 token 层面到底是怎么回事，这本书可以讲到底。

这个系列的第一本《自己动手写LLM推理引擎》讲模型怎么跑，第二本《自己动手训练LLM》讲模型怎么训。这一本讲模型怎么用：给模型配上工具和循环，让它替人干活。三本书合起来，正好是一个模型从「训出来」到「跑起来」再到「干起活来」的全过程。

写法上：一章一个概念，先有问题再有部件；不用框架，少用 SDK，API 直接看 HTTP 请求和 SSE 事件，界面从 ANSI 转义序列写起；每章结尾对照 Claude Code、Codex、Pi、mini-swe-agent 在同一个问题上的选择。

## 目录

<BookToc book="agent" />
