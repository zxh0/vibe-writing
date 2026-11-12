# 代码设计

> 工作草稿，随讨论随时改。先定代码的整体结构、验证方法和PoC要回答的问题，再规划每章代码。

## 约束

### 硬件：Google Colab的T4

已定（2026年11月8日）：全书只用Google Colab的T4训练，先不考虑国内平台；模型只设计一个，不分档。

T4的几个特点，决定了后面不少设计：

* 图灵架构（sm75），16GB显存，fp16的峰值约65 TFLOPS。
* 硬件不支持bf16，混合精度只能用fp16，要配`GradScaler`做损失缩放。
* 跑不了FlashAttention（要sm80以上），`F.scaled_dot_product_attention`会自动退到省内存的实现，对我们没有影响。
* 免费版的会话有时长上限，空闲一会儿会断开，GPU也不保证每次都分得到。会话结束，本地磁盘清空。所以数据和checkpoint都放Google Drive，训练必须能断点续训。
* 配的CPU很弱（约2核、12GB内存），数据准备能做，就是慢。

吞吐量还没实测。按小模型利用率低、实际只有5到10 TFLOPS估算，26M的模型每秒大约3万到6万个token，训5亿token要两到五个小时。PoC的P0要先把这个数测出来。

### 作者的Mac：写代码和调试，不训练

作者的机器是2019款Intel MacBook Pro：i7-9750H（6核12线程）、16GB内存、Radeon Pro 555X，不是Apple Silicon。实测一个Llama风格的小模型，batch 8×256，前向加反向加AdamW，torch 2.2.2，fp32：

| 模型 | 配置（d, 层, 头, KV头, FFN, 词表） | CPU | MPS |
|---|---|---|---|
| 4.0M | 256, 4, 4, 2, 704, 4096 | 3,058 tok/s | 6,629 tok/s |
| 11.9M | 384, 6, 6, 2, 1024, 6400 | 940 tok/s | 2,671 tok/s |
| 26.7M | 512, 8, 8, 2, 1408, 8192 | 291 tok/s | 1,236 tok/s |

26M的模型在MPS上训5亿token要四五天，训不了。Mac的用处是：写代码、用很小的配置在CPU上调通、和参考实现核对结果，以及最后用第一本的引擎跑导出的模型。

### 软件

* **Colab：** 用它预装的新版torch，不重装；缺的几个包用`pip`装。
* **Mac：** 一个uv项目，torch钉在2.2.2。Intel Mac上PyTorch从2.3起就没有安装包了（见第一本`code/CLAUDE.md`）。
* **代码只用torch 2.2就有的API**，两边都能跑。所以没有`nn.RMSNorm`（2.4才有），也不依赖`torch.compile`。
* `numpy<2`、`transformers<5`只是Mac上的约束，原因和第一本相同。

### 自包含

和第一本一样：书目录里的东西不能引用父目录或第一本的任何文件。需要第一本的代码，就复制一份过来。两本书之间只通过“模型目录”这一种格式交接（见“和第一本的接口”）。

## 模型

只有一个模型，目标是在T4上训完一遍不超过一次会话：

| 项目 | 数值 |
|---|---|
| 配置 | 词表8192，`d=512`，8层，8头（KV 2头），FFN 1408，权重共享 |
| 参数量 | 约26M |
| 预训练数据 | 约5亿token（Chinchilla，约20 token/参数），FineWeb-Edu的一个子集 |
| 一次完整预训练 | 估计两到五个小时，待P0实测 |
| 一组对照实验 | 只训几千万token，估计十几到几十分钟 |

这个尺寸是暂定的。P0测出来要是比估计慢很多，就往下缩，比如`d=384`、6层，约12M。

对照实验不另设小模型：同一个配置，少训一些token。几千万token已经足够拉开warmup、词表大小这些选择的差别。

### 考虑过但不采用

* **两档模型（Mac上训4M的小号，租卡训26M的正式模型）：** 分档是为了省钱，Colab免费，这个理由就没了。少一档模型，也省掉了给小号找一份简单语料（中文没有现成的TinyStories）的麻烦。
* **租卡（AutoDL等）：** 先不考虑。
* **小云主机（2核2GB，最新PyTorch，只用CPU）：** 在Mac上把CPU限制成2个线程模拟，4M的模型每秒1,417个token、峰值内存0.6GB，26.7M的模型每秒137个token、峰值1.6GB。云主机的“2核”通常只是一个物理核的两个超线程，实际还要再慢一半；2GB内存也放不下26M模型的训练、SmolLM2的梯度核对和分词器训练。最新PyTorch只能带来常数倍的提升。比Mac还慢，不用。

## 数据集

当前方案（2026年11月9日）：全英文，照SmolLM2自己的数据配方缩小来用。HuggingFace把SmolLM2预训练和后训练用的数据大多公开了，正好和第一本的SmolLM2接上：第一本跑SmolLM2，这一本照它的配方训一个缩小版，最后再用第一本的引擎跑。

全书用英文还是中文，还没最终定（见“待定”）。章节提纲目前仍按中文写，定下来再改。

许可证是凭记忆写的，都要到数据集页面上逐个核实。

| 阶段 | 推荐 | 备选 | 许可证（待核实） |
|---|---|---|---|
| 第一章演示 | Tiny Shakespeare（约1MB，nanoGPT的经典例子） | 无 | 公有领域 |
| 第二章亲手清洗 | 原始的FineWeb（没打过分的网页）或英文维基，取一小份 | 无 | ODC-By、CC BY-SA |
| 预训练 | FineWeb-Edu，取一个子集（它有现成的10B token抽样版） | SmolLM语料（FineWeb-Edu去重版、Cosmopedia合成教科书等）；TinyStories | ODC-By |
| SFT | smol-smoltalk（smoltalk专门给1B以下小模型做的精简版） | smoltalk全量、Dolly-15k、UltraChat | Apache 2.0 |
| DPO | UltraFeedback binarized（SmolLM2做DPO用的就是它） | Anthropic HH-RLHF | MIT |
| 强化学习 | 自己生成加法题，不需要数据集 | GSM8K（对26M的模型太难）、Countdown一类合成任务 | 无 |
| 蒸馏 | 老师模型现场生成 | Cosmopedia本身就是大模型生成的合成数据，可以当现成的蒸馏语料来讲 | Apache 2.0 |
| LoRA | 从SFT数据里取一个子集 | 无 | 同SFT |

### 为什么选这一套

* **主线连贯：** 全书讲的是同一个模型家族，数据配方也跟着SmolLM2走。
* **参照多：** nanoGPT和nanochat都用FineWeb-Edu，SmolLM2有公开的技术报告，TinyStories有论文。我们的loss和生成效果有很多现成的东西可以比。
* **第二章有内容：** FineWeb-Edu有原始版和打过分的版本，“质量过滤到底有多大用”可以直接做对照实验，这本来就是FineWeb-Edu论文的核心结论。
* **许可证清楚：** 大多是ODC-By、Apache、MIT这类宽松协议。
* **分词器简单：** 8192的词表对英文够用，嵌入矩阵不会吃掉太多参数。

### 代价

* **要自己拼：** 每个阶段的数据来自不同的地方，格式也不一样，要多写一些下载和转换的代码。
* **中文读者看到的全是英文输出：** 第一章的唐诗也换成了莎士比亚。
* **和Raschka、nanochat撞车更多：** 语言这一层差异没了，卖点全靠和第一本首尾相接、对照实验、先验证再训练这三点。

### 考虑过但不采用

* **MiniMind的语料（中英混合）：** 一个仓库里预训练、SFT、DPO、强化学习的数据全配好，最省事；我们的26M配置和它的minimind2-small也一模一样。不用的原因：数据已经洗好，第二章会变薄；数据集没有统一的许可证，上游有CC-BY-NC-2.0的部分；和SmolLM2这条主线接不上。
* **中文：Fineweb-Edu-Chinese加中文维基（之前的方案）：** 要自己清洗、去重、混合；Fineweb-Edu-Chinese各版本的许可证说法不一致；还要处理繁简转换。如果最后定全书用中文，回到这个方案。
* **中英混合：** 词表要放大到16k左右，嵌入矩阵约占模型的三成；26M的容量被两种语言分走。
* **CCI3.0-HQ（智源）：** 质量好，但500GB太大，许可证没查到。
* **中文TinyStories：** 唯一找到的adam89/TinyStoriesChinese只有48条。

## 目录结构

照第一本：一个uv项目，两棵代码树，共用数据、模型和运行记录。

```
code/
├── pyproject.toml     Mac上的uv项目，torch钉在2.2.2
├── data/              语料：原始、清洗后、分词后，不入库
├── models/            模型目录：SmolLM2-135M（当答案用）、导出的模型，不入库
├── runs/              训练记录：日志、checkpoint，不入库
├── reference/         核对结果用的基准张量，不入库
├── colab/             Colab笔记本：挂载Drive、装依赖、调用book/里的脚本
├── poc/               概念验证：先跑通全流程，冻结后只作参照
│   ├── toytrain/      可复用的包
│   ├── steps/         里程碑脚本，每个能单独跑
│   └── tests/         pytest，重点是和参考实现核对数值
└── book/              随书代码：一章一个目录，ch01/ … ch12/
```

`book/`的规矩和第一本相同：每章从上一章复制过来再改，每章都能单独跑；本章没改的文件和上一章逐字节相同，用一个检查脚本守住。

### 在Colab上怎么跑

代码都是普通的`.py`脚本，笔记本只做三件事：

1. 挂载Google Drive，把`data/`和`runs/`指到Drive上，会话断了也不丢；
2. 拉代码、`pip`装缺的包；
3. 用`!python book/chNN/xxx.py`调脚本。

这样同一份脚本在Mac上用`uv run`也能跑，书里只讲脚本，笔记本是附带的。一章一个笔记本，还是全书一个，待定。

## 模块设计

PoC里的包`toytrain/`，也是随书代码最终会长成的样子。

### `data`：语料

* 下载：用`huggingface_hub`下parquet文件，用`pyarrow`读。不用`datasets`，太重。
* 清洗：规则过滤（长度、乱码、重复字符），每条规则单独统计删掉多少。
* 去重：完全重复用哈希；近似重复手写一个MinHash，不引入新依赖。
* 切分：训练集和验证集一开始就分开，按文档切，不按token切。
* 存储：分词后的token存成`uint16`的扁平二进制文件（词表小于65536），训练时用`numpy.memmap`随机取块。和nanoGPT的做法一样。5亿token约1GB，放在Drive上。

### `tokenizer`：分词器

两条线：

* **手写BPE训练器**：讲原理用，在几十MB的样本上训，纯Python，慢也无所谓。
* **`tokenizers`库训练**：在全部语料上训，产出真正使用的`tokenizer.json`；给整个语料分词也用它。

两者在同一份样本上要训出相同（或几乎相同）的合并规则，这就是检验手写训练器的办法。合并次数相同时谁先谁后，两边的规则可能不同，要在PoC里确认能对到什么程度。

格式必须是字节级BPE（ByteLevel），否则第一本的`StreamDecoder`解不了码。特殊token：`<|endoftext|>`、`<|im_start|>`、`<|im_end|>`，和第一本第十三章用的ChatML一致。

### `config`：模型配置

一个dataclass，和`config.json`互相转换。字段名和SmolLM2的`config.json`一致，`model_type`写`llama`。

### `model`：模型

* 一个`nn.Module`，Llama架构：RMSNorm、RoPE（不交错）、GQA、SwiGLU、权重共享。和第一本的数学一样，写法改成训练用的：多了batch维，不要KV Cache（生成另写），参数是`nn.Parameter`。
* **参数名和HuggingFace的state_dict一字不差**：`model.embed_tokens.weight`、`model.layers.0.self_attn.q_proj.weight`……这样做有两个好处：
  * SmolLM2-135M的权重能直接装进来，和`transformers`对答案；
  * 保存时直接把state_dict写进safetensors，不用改名。
* RMSNorm手写（torch 2.2没有`nn.RMSNorm`），在fp32里算，fp16下不容易溢出。注意力用`F.scaled_dot_product_attention`，第一本已经手写过，这里不再重复。
* 初始化：正态分布，std 0.02；`o_proj`和`down_proj`再按层数缩小（GPT-2的做法）。第四章做对照。
* 自带一个简单的`generate`（带KV Cache），给第六章定期生成和第九章GRPO采样用。

### `train`：训练循环

* 优化器：第五章手写一个AdamW讲原理，和`torch.optim.AdamW`核对结果；之后都用torch自带的。
* 学习率：warmup加cosine，手写，和`transformers`的同名调度器核对结果。
* 梯度累积、梯度裁剪。
* **断点续训是硬要求**：checkpoint存模型、优化器、`GradScaler`、学习率进度、数据位置和随机数状态，定期写到Drive。会话断了，重新连上一条命令接着训。
* 设备和精度：`--device cuda|cpu|mps`。CUDA上用fp16 autocast加`GradScaler`；CPU和MPS上用fp32，只用来调试。
* 日志：每一步写一行JSON到`runs/<名字>/log.jsonl`。画图脚本照第一本用PEP 723声明matplotlib，不进依赖。
* 定期评估：验证集loss，加上固定提示词的生成样例，一起写进日志。

### `export`：导出模型目录

写出`config.json`、`generation_config.json`、`model.safetensors`、`tokenizer.json`，和SmolLM2的目录长得一样。权重共享时不存`lm_head`。权重存fp32，和训练时的混合精度无关。

### 后训练

* `sft`：ChatML模板、只在助手回答上算loss（掩码）、多条对话拼成一个序列。
* `dpo`：参考模型冻结，DPO loss手写。
* `grpo`：加法任务生成器、奖励函数、组内归一化、KL项。用`model.generate`采样。
* `distill`：老师生成数据，然后就是一次SFT。Qwen3-0.6B的fp16权重约1.2GB，T4放得下。
* `lora`：一个包住`nn.Linear`的LoRA层，训练完合并回去再导出。

## 和第一本的接口

两本书只通过模型目录交接：在Colab上导出一个目录，下载到Mac，拷到第一本的`code/models/`下，第一本用`--model`选它。

第一本的引擎（第十四章）对这个目录的要求，读代码确认过：

| 文件 | 要求 |
|---|---|
| `config.json` | `hidden_size`、`num_hidden_layers`、`vocab_size`、`tie_word_embeddings`、`rms_norm_eps`、`num_attention_heads`、`num_key_value_heads`、`rope_theta`、`max_position_embeddings`、`intermediate_size`、`eos_token_id`、`model_type`；`head_dim`和`rope_interleaved`可省 |
| `generation_config.json` | 必须有，引擎直接打开，不检查是否存在 |
| `model.safetensors` | HF命名，每层9个张量；有`lm_head`就必须和词嵌入相同 |
| `tokenizer.json` | `tokenizers`能读；必须是字节级BPE，`StreamDecoder`按GPT-2的字节映射解码 |

一个要注意的地方：第一本第十三章的`chat.py`在不加`--think`时，会替Qwen3塞一段空的`<think></think>`。我们的模型没见过这两个token，直接用会出问题。要么第七章的SFT数据也带上这个格式，要么在第一本加一个开关。待定。

## 验证

延续第一本“和参考实现对答案”的习惯。能逐位对的就逐位对，不能的换成统计或曲线：

| 对象 | 参考 | 标准 |
|---|---|---|
| 手写BPE训练器 | `tokenizers`在同一样本上训出的结果 | 合并规则相同（程度待PoC确认） |
| 模型前向 | `transformers`加载SmolLM2-135M | logits相对误差 < 1e-5（fp32） |
| loss和梯度 | 同上，对同一批数据 | loss和每个参数的梯度，相对误差 < 1e-5（fp32） |
| 手写AdamW | `torch.optim.AdamW` | 若干步后参数`torch.equal`或相对误差 < 1e-6 |
| 学习率调度 | `transformers`的cosine调度器 | 每一步的学习率相同 |
| 导出 | `transformers`加载导出目录 | logits和我们自己的模型一致 |
| 导出 | 第一本的引擎 | 贪婪生成的文字和我们自己的`generate`一字不差 |
| 断点续训 | 不中断地训同样的步数 | 同一设备上，续训后的loss曲线和不中断的一致 |
| 训练本身 | 无法逐位复现 | 固定种子、同一设备上重跑一致；loss曲线和生成样例 |
| DPO loss | 按公式手算的小例子 | 数值相同 |
| GRPO | 无 | 加法正确率随训练上升 |

这些核对都在fp32下做，Mac的CPU上就能跑。`transformers`只作dev依赖，只在核对用的脚本和测试里出现，不进`toytrain/`和章节的主代码。

## 依赖

* 运行：`torch`、`numpy`、`safetensors`、`tokenizers`、`huggingface-hub`、`pyarrow`。Colab上大多已经预装。
* dev：`pytest`、`transformers<5`。

和第一本不同，`tokenizers`是运行依赖：几GB语料的分词得靠它的Rust实现，第一本的引擎本身也用它读`tokenizer.json`。

## PoC：要回答的问题

PoC不是随书代码，它的任务是在写书之前把风险排掉。每个里程碑对应一个问题，回答了才往下走：

| 编号 | 里程碑 | 要回答的问题 | 通过标准 |
|---|---|---|---|
| P0 | Colab环境 | T4上的吞吐量、免费会话能跑多久、Drive读写快不快 | 实测26M模型的tok/s，定下模型尺寸 |
| P1 | 跑通 | 一小份数据，从清洗到训练到生成，全流程能不能在Colab上走通 | 训几千万token，生成的英文句子开始通顺 |
| P2 | 和SmolLM2对答案 | 我们的模型对不对 | loss和全部梯度和`transformers`对上 |
| P3 | 导出 | 第一本的引擎能不能直接跑 | 两边贪婪生成一字不差 |
| P4 | 完整预训练 | 5亿token要几次会话，fp16稳不稳，断点续训靠不靠得住 | 跑完一次完整预训练，中途至少断一次再接上 |
| P5 | SFT | 26M模型能不能学会一问一答 | 在第一本的引擎里聊几句 |
| P6 | DPO和GRPO | 小模型上能不能看到效果 | DPO后好坏回答的概率差拉开；GRPO后加法正确率上升 |
| P7 | 蒸馏 | 在T4上生成老师数据要多久 | 实测Qwen3-0.6B生成一万条的时间 |

P0先做，它决定模型尺寸。P2、P3在Mac上就能做，可以和P0、P1并行。

## 每章代码（初稿）

| 章 | 目录里的新东西 |
|---|---|
| 一 | `char_gpt.py`：一个文件的字符级模型，几分钟训Tiny Shakespeare |
| 二 | `download.py`、`clean.py`、`dedup.py`、`split.py` |
| 三 | `bpe_train.py`（手写）、`train_tokenizer.py`（`tokenizers`）、`compare.py`、`encode.py` |
| 四 | `config.py`、`model.py`、`init_loss.py`、`check_smollm2.py`、`export.py`、`check_export.py`；`code/download_model.sh`下载SmolLM2-135M |
| 五 | `adamw.py`、`schedule.py`、`check_accum.py`、`overfit.py`、`train.py`（含最简单的checkpoint）、`plot_loss.py`；只在Mac上跑300步的对照实验 |
| 六 | `train.py`补上fp16、定期生成，checkpoint补齐`GradScaler`和随机数状态，做到断了接着训和不断的曲线一致 |
| 七 | `chat_template.py`、`sft.py`、`chat.py` |
| 八 | `dpo.py` |
| 九 | `addition.py`、`grpo.py` |
| 十 | `teacher.py`、`distill.py` |
| 十一 | `lora.py`、`merge.py` |
| 十二 | `train.py`改成DDP，用`torchrun`启动；Colab只有一张卡，在CPU上开两个进程演示机制 |

## 待定

1. **老师怎么生成数据：** 自包含的规矩下不能直接调第一本的引擎。用`transformers`生成，还是把第一本的引擎复制一份过来？
2. **`<think>`：** 见“和第一本的接口”。
3. **数据许可证：** “数据集”一节里的许可证都是凭记忆写的，要逐个核实。
4. **全书的语言：** 数据集一节已经按全英文写；章节提纲还按中文写。定了以后，第一、二、三、七章的提纲要跟着改。
5. **笔记本：** 一章一个，还是全书一个。
6. **免费版的限制：** 会话时长和GPU配额经常变，书里写多少、怎么写才不容易过时。
