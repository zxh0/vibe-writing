# 调研

> 工作草稿。2026年11月11日做的第一轮调研，大部分条目只看了标题、摘要和二手介绍，标了“待核实”的数字和日期，写正文前要回到原文确认。

## 一句话结论

“自己写一个Coding Agent”这件事，英文世界已经有很多好材料，而且几乎都在说同一件事：核心就是一个循环，几百行代码就能跑起来。缺的是两样东西：一本把每个设计决策讲出“为什么”、一章一个概念循序渐进的中文书；以及一本能从模型内部（token、对话模板、KV缓存）往外讲Agent的书。后一点正好接上toyllm系列的前两本。

## 博客和文章

### 从零写一个Agent

这几篇是本书最直接的同类，都在证明“Agent没有魔法”：

* **Thorsten Ball，How to Build an Agent**（Amp博客，2025年4月，副标题The Emperor Has No Clothes）。用Go写，三个工具（`read_file`、`list_files`、`edit_file`）加一个循环，三四百行，大部分是样板代码。这篇几乎是这个话题的“原点”，后来有JavaScript和TypeScript的移植版。<https://ampcode.com/notes/how-to-build-an-agent>
* **Mario Zechner，What I learned building an opinionated and minimal coding agent**（2025年11月）。Pi的作者讲为什么只要四个工具（read、write、edit、bash）和不到一千token的系统提示词：模型本来就被训练成了Coding Agent，不用再花一万token教它。<https://www.mariozechner.at/posts/2025-11-30-pi-coding-agent/>
* **Sid Bharath，Build a Coding Agent from Scratch: The Complete Python Tutorial**。用Python、不用框架，照着Claude Code的样子做一个约三百行的单文件Agent。<https://sidbharath.com/blog/build-a-coding-agent-python-tutorial/>
* **Rodrigo Girão Serrão，Write a coding agent from first principles**。从定义出发：Coding Agent就是配上了工具、能和环境交互的LLM。
* 其他同类教程还有不少，比如Together AI的指南和若干“60行实现一个Agent”的文章，内容大同小异，不一一列出。<https://www.together.ai/blog/how-to-build-a-coding-agent-from-scratch-a-practical-guide-for-developers>

### 概念和定义

* **Simon Willison，“An LLM agent runs tools in a loop to achieve a goal”**（2025年9月18日）。目前最通行的定义，本书第一章直接用它。他同月还写了一篇Designing agentic loops。<https://simonwillison.net/2025/Sep/18/agents/>
* **Lilian Weng，LLM Powered Autonomous Agents**（2023年6月）。早期的经典综述，把Agent拆成规划、记忆、工具三块。很多概念今天仍然在用，但具体做法（比如向量数据库做长期记忆）已经被Coding Agent的实践改写了不少，正好可以拿来做“当年怎么想、现在怎么做”的对照。<https://lilianweng.github.io/posts/2023-06-23-agent/>
* **Chip Huyen，Agents**（2025年1月）。讲工具、规划、失败模式和评估，偏系统视角。

### 官方工程文章

这几篇是“设计思想”那几章的主要依据：

* **Anthropic，Building effective agents**。区分workflow和agent，主张从最简单的方案开始，确有必要才加复杂度。
* **Anthropic，Effective context engineering for AI agents**。把“上下文工程”定义为在推理时挑选、维护最合适的那组token；好的上下文是“最小的一组高信号token”。<https://www.anthropic.com/research/effective-context-engineering-for-ai-agents>
* **Anthropic，Effective harnesses for long-running agents**。长任务跨多个会话时，用进度文件加git历史让新会话接上进度。<https://anthropic.com/engineering/effective-harnesses-for-long-running-agents>
* **Anthropic，Equipping agents for the real world with Agent Skills**。Skills的核心设计是渐进式披露（progressive disclosure）：启动时只放名字和描述，触发时才读`SKILL.md`正文，附带的文件用到才读。<https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills>
* **OpenAI，Unrolling the Codex agent loop**（Michael Bolin，2026年1月）。拆解Codex CLI的循环：一轮里可以有很多次“模型→工具”，直到模型回一条纯文本消息才结束；也讲了上下文压缩。<https://openai.com/index/unrolling-the-codex-agent-loop/>
* **Manus，Context Engineering for AI Agents: Lessons from Building Manus**。把KV缓存命中率当成生产Agent最重要的指标；用不断重写的`todo.md`把目标拉回上下文末尾。这篇和本书最有缘分：KV缓存是第一本第十一章的主角。（原文还没读，目前依据是二手摘要，待核实）

### Claude Code的设计

* Boris Cherny（Claude Code作者）多次提到，团队早期试过用本地向量数据库做RAG，后来换成了grep、glob加读文件的“agentic search”，效果反而好得多。这是“检索”一章最好的切入点。（原话出自Latent Space播客和他的帖子，待核实）
* 2026年3月Claude Code的source map意外泄露之后，出现了大量源码分析，比如“98% of Claude Code Is Not AI”、claudecn.com的源码分析系列、一篇讲那个一千多行主循环的长文。可以参考，但都是外部推断，正文引用时要说明来源性质，不当成官方事实。
* 官方文档（Claude Code的hooks、subagents、skills、MCP、settings）本身就是设计思想的一手材料。

### LLM API

* **OpenAI，Responses vs. Chat Completions**（官方迁移指南）。Chat Completions无状态，每次发完整的消息列表，工具循环由调用方自己写；Responses面向Agent，可以由服务器保存对话状态，还带内置工具。OpenAI建议新项目用Responses，Chat Completions继续支持。<https://developers.openai.com/api/docs/guides/responses-vs-chat-completions>
* **Anthropic Messages API**（官方文档）。同样无状态，但内容是分块的（text、tool_use、tool_result、thinking），system提示词放在顶层，`max_tokens`必填。
* 三套协议的对比文章不少，多出自API网关厂商（Portkey等），可以当索引用，具体参数要回官方文档核对。<https://portkey.ai/blog/open-ai-responses-api-vs-chat-completions-vs-anthropic-anthropic-messages-api>
* 一个现成的对照：Claude Code只说Messages，Codex CLI只说Responses。Pi则自己封装了一层统一的多厂商接口（pi-ai），这层怎么抹平各家差别，正好是第五章的参考。（待核实）
* 本地推理服务：llama.cpp的`llama-server`、Ollama、vLLM都提供OpenAI兼容的接口，但工具调用、流式、思考内容的兼容程度不一，要实测。

### 终端界面

* **pi-tui**（Pi的终端界面库）。主打差分渲染：第一次全部输出，窗口宽度变了就清屏重画，其他时候只从第一处变化的行开始重写；每一帧再用同步输出（CSI 2026）包起来，让终端一次性显示，不闪。第九章基本就是照这个思路写一个Python的玩具版。<https://www.mintlify.com/pt-act/pi-mono/packages/tui>
* **Peter Steinberger，Signature flicker**。讲Claude Code长期存在的闪烁问题，以及2.0.72版的修复，顺带把Pi的差分渲染当成标杆。是观点文章，不是测评，但很适合当第九章“全量重绘为什么闪”的引子。<https://steipete.me/posts/2025/signature-flicker>
* **Claude Code的界面**。用React写终端界面，组件树经过Yoga（flexbox布局引擎）排版再转成ANSI输出。用的是npm上的Ink，还是自己基于react-reconciler写的类Ink渲染器，各家分析说法不一，待核实。社区还有人用Rust和ratatui重写了一个界面，自称内存降了一个数量级（自述数字）。
* **Codex CLI的界面**。核心是Rust，界面一般说是ratatui加crossterm，也有说法是另一层TypeScript外壳，要直接看codex-rs的仓库确认。
* **TUI框架综述**：有几篇比较Ink、Bubble Tea、ratatui、Textual的文章，讲声明式、Elm式、立即模式几种思路的取舍，第十章可以参考。<https://www.youngju.dev/blog/culture/2026-05-14-tui-development-ratatui-bubbletea-ink-textual-terminal-ui-renaissance-deep-dive-2026.en>
* 终端本身的原理（tty、pty、termios、ANSI转义序列）是老知识，资料很多，写第八章时挑一两份经典的就行，比如xterm的控制序列文档。

## 开源教学项目

* **mini-swe-agent**（SWE-bench团队）。Agent类约一百行，只有bash一个工具，连模型的工具调用接口都不用，却能在SWE-bench Verified上拿到七成以上（待核实）。它是“极简”这条路走到头的样子，适合在讲工具调用之前当反例：不用tool call也行，那tool call到底带来了什么？<https://github.com/SWE-agent/mini-swe-agent>
* **Pi**（Mario Zechner，现在由Earendil维护）。TypeScript，四个工具，扩展靠类型化的插件系统，刻意不内置MCP、计划模式和子代理。它代表“小核心、可扩展”的设计取向，和Claude Code的“功能全、开箱即用”正好是两端。
* **shareAI-lab/learn-claude-code**。口号是“Bash is all you need”，按课时一步步做一个类Claude Code的harness，从`s01_agent_loop.py`开始。中文社区很火，结构和本书最接近，写作时要注意区分。
* **alienzhou/zero2agent**。从零做一个类Claude Code或Codex的Agent，附完整的构建日志，仍在更新。

## 书

* **Build an AI Agent (From Scratch)**，Jungjun Hur、Younghee Song，Manning的From Scratch系列（2026年夏，具体日期各处说法不一）。ReAct循环、MCP、agentic RAG、记忆、规划与自我纠错、代码执行Agent、多智能体。它和Raschka那本Build a Large Language Model (From Scratch)同属一个系列，是本书在英文世界最像的一本。不过它是通用Agent，不专门讲Coding Agent。
* **Build Your Own Coding Agent**（Leanpub，纯Python，从零做一个生产级Coding Agent）。题目和本书几乎一样，要找来看看目录。
* **AI Agents in Action**第2版（Micheal Lanham，Manning）、**Building Applications with AI Agents**（Michael Albada，O'Reilly）、**Designing Multi-Agent Systems**（Victor Dibia）。偏框架和应用，不是从零手写。
* **AI Agents with MCP**（Kyle Stratis，O'Reilly）。专讲MCP。
* **Hello-Agents**（Datawhale，中文，开源在线教程）。从智能体的定义和历史讲起，讲ReAct、Plan-and-Solve，手写一个教学用的多智能体框架，后面还有上下文工程、记忆、协议、评估和Agentic RL。它是中文里最系统的一份，但定位是通用智能体，不专注Coding Agent，也不往模型内部讲。<https://github.com/datawhalechina/Hello-Agents>

## 本书的位置

和toyllm2的想法一样，“又一个从零写Agent的教程”本身没有太大必要，上面这些已经够用了。本书要占的是下面几个位置：

1. **从模型内部往外讲。** 读过第一本的人知道，模型只会预测下一个token。那工具调用到底是什么？本书会把它拆到token：工具的JSON Schema怎么被对话模板拼进提示词，模型又是怎么“说出”一段工具调用的。上下文窗口、prompt caching也能落到第一本讲过的KV缓存上。这一点别的教程都给不了，是本书最硬的理由。
2. **专注Coding Agent。** 通用Agent的书讲旅行助手、深度研究；本书只做一件事：一个能在终端里读代码、改代码、跑测试的助手。读者天天在用它，每个设计都能拿自己熟悉的工具对照。
3. **一章一个概念，每章都能跑。** 第一章就有一个能干活的循环，之后每章往上加一样东西：工具调用、权限、压缩、检索、MCP、Skills……加完都能跑，都能和上一章比。
4. **对照真实设计。** 每章结尾看Claude Code、Codex、Pi、mini-swe-agent在同一个问题上是怎么选的，选得不一样又是为什么。和tpu_book“对照Google的选择”是同一个做法。
5. **能“对答案”的部分就对答案。** Agent的输出是随机的，没法像第一本那样逐位比对。但可以退一步：用录好的模型回复回放，让循环本身变成确定的、可测试的；再自建一组小题，用通过率衡量每一章的改动有没有用。

## 待读

写对应章节之前要读完原文：

* [ ] Thorsten Ball的原文，以及Go代码（第一章）
* [ ] Mario Zechner的原文和Pi的源码（第一章、最后几章）
* [ ] mini-swe-agent的源码（第一章、工具调用一章）
* [ ] Anthropic和OpenAI的几篇工程文章（上下文、Harness几章）
* [ ] Manus的原文（上下文一章）
* [ ] Manning那本和Leanpub那本的目录（定位）
* [ ] learn-claude-code和Hello-Agents的目录（避免撞车）
* [ ] MCP规范、Agent Skills规范（对应章节）
* [ ] OpenAI和Anthropic的API文档，尤其是流式事件和工具调用的格式（第二到五章）
* [ ] pi-ai和pi-tui的源码（第五章、第九章）
* [ ] codex-rs的界面代码，确认用的是不是ratatui（第十章）
