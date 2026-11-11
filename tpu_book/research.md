# 同类资料调研

调研时间：2026-11。目标是了解目前有哪些讲 TPU 或 AI 加速器设计的资料，它们各自从什么角度切入，以及"假装从零设计 TPU"这本书还能补上什么。

## 一、Google 官方论文：一手资料

这是全书最重要的事实来源。论文按代际排列，正好对应本书的主线。它们写给体系结构同行，重点是设计结果和性能对比，推导过程讲得很少。

- **Jouppi et al.，In-Datacenter Performance Analysis of a Tensor Processing Unit**（ISCA 2017，[arXiv](https://arxiv.org/abs/1704.04760)，[Google Research](https://research.google/pubs/in-datacenter-performance-analysis-of-a-tensor-processing-unit/)）。TPU v1 的论文。核心是一个 256×256、共 65,536 个 8 位乘加单元的矩阵单元，峰值 92 TOPS，配 28 MiB 软件管理的片上存储，与同期的 Haswell CPU 和 K80 GPU 做了对比。本书第一部分基本围绕它展开。
- **Jouppi et al.，A Domain-Specific Supercomputer for Training Deep Neural Networks**（CACM 2020）。讲 TPU v2/v3，从推理芯片走向训练芯片：bfloat16、HBM、向量单元、片间互连和 pod。
- **Jouppi et al.，Ten Lessons From Three Generations Shaped Google's TPUv4i**（ISCA 2021）。推理芯片 TPUv4i 的设计复盘，"十条教训"很适合作为本书各章取舍的对照。
- **Jouppi et al.，TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings**（ISCA 2023，[arXiv](https://arxiv.org/abs/2304.01433)）。光交换（OCS）可重构拓扑，以及处理 embedding 的 SparseCore。
- **Google Cloud TPU 文档**（[TPU7x / Ironwood](https://docs.cloud.google.com/tpu/docs/tpu7x)）。最新一代的官方规格：每芯片两个 chiplet、两个 TensorCore 和四个 SparseCore，192 GB HBM，BF16 峰值约 2,307 TFLOPS，一个 pod 9,216 颗芯片。各代的系统架构页也在这套文档里。

## 二、开源的"小 TPU"项目：最接近本书的动手路线

这类项目和本书的思路最接近：从一个 PE 开始，用 Verilog/SystemVerilog 搭一个小脉动阵列。它们的共同特点是代码可跑、规模很小（2×2 到 16×16），但讲解较少，很少解释"为什么这样设计"，基本停在单个阵列，不涉及多芯片。

- **tiny-tpu-v2/tiny-tpu**（[仓库](https://github.com/tiny-tpu-v2/tiny-tpu)，[架构文档](https://www.mintlify.com/tiny-tpu-v2/tiny-tpu/architecture/systolic-array)，[介绍文章](https://blog.brightcoding.dev/2025/09/23/tiny-tpu-a-minimalist-tensor-processing-unit-inspired-by-google-s-tpu-v1-and-v2)）。参考 TPU v1 和 v2 重新实现的极简 TPU。除了 2×2 脉动阵列，还有向量单元（bias、Leaky ReLU、MSE loss 及其导数），能在硬件上跑完前向、反向和梯度下降。它是少数覆盖训练的开源项目，值得精读。
- **deaneeth/tiny-tpu**（[仓库](https://github.com/deaneeth/tiny-tpu)）。4×4 weight-stationary 阵列，用 cocotb 和 Verilator 验证，还编译成 WebAssembly 做了浏览器里的可视化。可视化的做法可以借鉴。
- **Alanma23/tinytinyTPU**（[仓库](https://github.com/Alanma23/tinytinyTPU-co)）。2×2 阵列，能部署到 Basys3 FPGA，用开源工具链（Yosys/nextpnr）。
- **cameronshinn/tiny-tpu**（[仓库](https://github.com/cameronshinn/tiny-tpu)）。FPGA 上的 TPU 实现，附有报告，综合分支支持到 16×16。
- **RightNow-AI/tiny-tpu**（[镜像](https://gh.r-mj.com/RightNow-AI/tiny-tpu)）。8×8 阵列，INT8/INT32 推理，带 PyTorch 集成。
- **Tiny Tapeout：Mini TPU v2**（[链接](https://tinytapeout.com/chips/ttsky26b/041)）。3×3 阵列，真正流片了。说明"做一颗玩具 TPU 芯片"对个人也是可能的。
- **课程实验**：台湾的一份 Verilog 实验（[HackMD](https://hackmd.io/@Tsai-Wooo/Lab3)）要求写出 PE 和 4×4 阵列，8 位输入、32 位累加；另有一篇 FPGA 上实现脉动阵列的入门笔记（[HackMD](https://hackmd.io/@ampheo/how-to-do-systolic-architecture-on-fpga-with-verilog)）。

## 三、学术界的加速器框架与教材

- **H. T. Kung，Why Systolic Architectures?**（IEEE Computer，1982）。脉动阵列的源头文章。TPU 的矩阵单元直接继承了这个思想，本书讲脉动阵列时应从它讲起。
- **Sze, Chen, Yang, Emer，Efficient Processing of Deep Neural Networks**（Morgan & Claypool，2020，[前身综述 arXiv](https://arxiv.org/abs/1703.09039)）。DNN 加速器设计最系统的教材。给出了数据流的分类（weight / output / input / row stationary）和评价指标，是本书讲"数据流选择"一章的理论依据。作者团队还做了 Eyeriss 芯片。
- **Gemmini（UC Berkeley）**（[论文](https://arxiv.org/abs/1911.09925)，[仓库](https://github.com/ucb-bar/gemmini)）。参数化的脉动阵列生成器，接在 RISC-V 核上，运行时可以在 output-stationary 和 weight-stationary 之间切换，已在 TSMC 16nm 和 Intel 22FFL 上流片。它是"工业级完整度的开源 TPU-like 设计"，适合作为本书设计空间探索的参照。
- **NVDLA**（NVIDIA 开源的深度学习加速器）。与脉动阵列不同的另一种路线，可作对比。
- **Hennessy & Patterson，Computer Architecture: A Quantitative Approach，第 6 版第 7 章 Domain-Specific Architectures**。用 TPU v1 作为案例讲领域专用架构的设计原则，篇幅一章，适合作为本书开篇"为什么要造专用芯片"的参考。
- **TPU-Gen**（[仓库](https://github.com/ACADLab/TPU_Gen)）。用 LLM 从文本描述生成脉动阵列 Verilog 的研究框架（ICLAD 2025），可以作为附录里的一个有趣延伸。

## 四、从软件和系统角度理解 TPU

- **Google DeepMind，How to Scale Your Model**（[全书](https://jax-ml.github.io/scaling-book)，[仓库](https://github.com/jax-ml/scaling-book)）。第 2 章 How to Think About TPUs 从使用者的角度讲 TPU：MXU、VMEM、HBM、ICI 的带宽和算力，用 roofline 判断瓶颈；后续章节讲多芯片并行和通信代价。这是目前讲 TPU 讲得最好的公开材料之一，但视角是"怎么用"，不是"怎么设计"。本书第三、四部分可以大量借用它的数字和成本模型。
- **JAX Pallas TPU 文档**。在真实 TPU 上写 kernel 的接口，能直接看到 TPU 的内存层次对编程的约束，适合作为本书"对照真实硬件"的实验手段。

## 五、中文资料

中文资料以课件和单篇文章为主，偏"介绍式"，很少有动手设计的内容。

- **ZOMI 酱，AISystem 课程**（[仓库](https://github.com/chenzomi12/AISystem)，[TPU 简介课件](https://aisystem-docs.pages.dev/_downloads/b0360534d1dabf4ab8855bb800574891/04TPUIntrol.pdf)，[TPU v3](https://aisystem-docs.pages.dev/_downloads/a4f2ada17d2bf062dd05da31e93f6717/07TPU3.pdf)，[TPU v4](https://aisystem-docs.pages.dev/_downloads/1a90f892304573b6daf14a963acf5d3d/08TPU4.pdf)，[TPU v1 脉动阵列](https://chenzomi12.github.io/_downloads/2e94d42c32e3e6f29400485b2b53e0c6/05TPU1.pdf)）。"AI 芯片与架构"部分有一整套 TPU 课件，按代际讲 v1 脉动阵列、v2/v3 训练卡与 pod、v4 光互连。是中文里最系统的 TPU 资料，形式是 PPT，适合快速建立全景。
- **腾讯云社区：TPU v1 架构解析**（[链接](https://cloud.tencent.com/developer/article/2471719)），以及**智源社区的相关文章**（[链接](https://hub.baai.ac.cn/view/36170)）。单篇解读 TPU v1 的控制单元、矩阵单元和累加器。
- **APX 中文课程页**（[TPU 与 ASIC 简介](https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-1-foundations-ai-compute-infrastructure/introduction-to-tpus-asics)，[Google TPU 架构](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-18-hardware-considerations-llm-training/tpu-architectures-google)，[AI 加速器代码生成](https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-5-heterogeneous-hardware-code-generation/ai-accelerator-code-generation)）。入门级概念介绍，最后一篇涉及矩阵乘法如何映射到脉动阵列。

## 结论：这本书的定位

现有资料大致分成三类，彼此之间缺一座桥：

1. **官方论文和教材**：事实权威、视野完整，但给的是设计结果，推导过程压缩得很厉害，初学者难以体会每个取舍背后的代价。
2. **开源小 TPU 项目**：能跑、能看波形，但规模停在一个小阵列，几乎不讲为什么，也不涉及训练之外的系统问题（片上存储层次、互连、pod）。
3. **系统和使用视角的资料**（如 How to Scale Your Model）：把 TPU 当成黑盒来用，讲得清楚，但不回答"它为什么被设计成这样"。

这本书可以占据的位置是：**面向初学者的中文书，以"假装从零设计"为叙事，把 TPU 从一个乘加单元一路推导到 Ironwood pod**。差异点包括：

- 叙事是设计过程而不是结构介绍。每个部件都由一个具体问题引出，先给出读者自己可能想到的方案，再对照 Google 的选择。
- 一个逐周期的 Python 模拟器贯穿全书，读者随时能测量阵列利用率和瓶颈，相当于 gpu_book 里那套 benchmark 的角色。
- 不停在单个脉动阵列，继续讲训练、存储层次、片间互连和多芯片系统，把开源项目和官方论文之间的空白补上。
- 与 gpu_book 互为对照，用同一套 roofline 语言比较 GPU 和 TPU 在同一个矩阵乘法上的不同解法。
