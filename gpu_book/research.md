# 同类资料调研

调研时间：2026-11。目标是了解目前有没有“以矩阵乘法为主线讲 GPU”的资料，它们各自覆盖到哪里，以及这本书还能补上什么。

## 一、逐步优化 GEMM 的 worklog（博客）

这类资料和本书的思路最接近：从朴素 kernel 出发，每次只做一个优化，并给出性能数字。它们的共同特点是节奏快，默认读者已经会写 CUDA，重点放在“做了什么”，对“为什么硬件是这样”着墨不多。

- **Simon Boehm，How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog**（[原文](https://siboehm.com/articles/22/CUDA-MMM)，[gwern 镜像](https://gwern.net/doc/www/siboehm.com/d4b3e9d6de0655f490fe6369f4d004936248a833.html)）。这是这一类资料里最常被引用的一篇。在 A6000 上从朴素版本（约 309 GFLOPS）开始，依次做访存合并、shared memory 缓存、1D/2D 分块、向量化访存、自动调优、warp tiling，最终达到 cuBLAS 的约 94%。它只做 FP32 SGEMM，Tensor Core 留给了“以后”。文章开头先算 FLOP 数和最小访存量来确定理论上限，这个做法值得借鉴。
- **Outperforming cuBLAS on H100: a Worklog**（[链接](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)）。单文件、无依赖，从分块版本走到 WGMMA 和 TMA，在 N=4096 时比 cuBLAS 快约 7%。作者自己也说明，这只针对特定形状，并不是 cuBLAS 的替代品。
- **Worklog: Optimising GEMM on NVIDIA H100 for cuBLAS-like Performance (WIP)**（[链接](https://hamzaelshafie.bearblog.dev/worklog-optimising-gemm-on-nvidia-h100-for-cublas-like-performance-wip)）。H100 上的同类记录，对 bank conflict 的分析和 padding 修复写得比较细。
- **Aleksa Gordić，Inside NVIDIA GPUs: Anatomy of high performance matmul kernels**（2025-09，原文在 aleksagordic.com/blog/matmul，[读书笔记](https://monishver11.github.io/blog/2025/aleksagordic-gpu-blog-notes)）。分四部分：GPU 架构基础、PTX/SASS、同步的 warp tiling kernel、Hopper 上使用 Tensor Core 和 TMA 的异步 kernel。它是目前覆盖面最接近本书的一篇，但仍是一篇长文，硬件部分讲得很密。
- **Modular，Matrix Multiplication on Blackwell 系列**（[Part 1](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction)，[Part 2](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-2-using-hardware-features-to-optimize-matmul)，后续还有 Part 3、4）。面向 Blackwell，用 Mojo 写。Part 1 讲矩阵乘法为什么重要，适合作为最新架构的参考。

## 二、中文资料

中文资料大多是 GitHub 仓库或单篇博客，以代码为主，讲解偏简略。

- **LessUp，SGEMM Optimization: From Naive to Tensor Core**（[仓库](https://github.com/LessUp/sgemm-optimization)，[中文说明](https://lessup.github.io/sgemm-optimization/README.zh-CN.html)）。从三层循环一直写到 WMMA，附带 roofline 分析和正确性验证，并给出 RTX 3060 Laptop 上各版本的实测数据。
- **georgelyu，CUDA GEMM 优化**（[链接](https://georgelyu.github.io/p/cuda_gemm_opt)）。结合《Programming Massively Parallel Processors》的实践笔记，覆盖朴素实现和 shared memory 分块。
- **aicl-lab，HPC AI 优化实验室：GEMM**（[链接](https://aicl-lab.github.io/hpc-ai-optimization-lab/zh-CN/guide/gemm)）。按 7 步讲优化路径。
- **hands-on-gemm**（[仓库](https://github.com/yangxianpku/hands-on-gemm)）。可编译、可跑 benchmark 的多版本 GEMM。
- **yhwang-hub，Matrix_Multiplication_Performance_Optimization**（[项目页](https://awesome.ecosyste.ms/projects/github.com%2Fyhwang-hub%2FMatrix_Multiplication_Performance_Optimization)）。类似路线，报告达到 cuBLAS 的约 71%。

## 三、书籍与系统性教材

- **Kirk & Hwu，Programming Massively Parallel Processors（PMPP）**（[第 4 版第 5 章](https://www.oreilly.com/library/view/programming-massively-parallel/9780323984638/xhtml/Ch005_93-121_B9780323912310000185.xhtml)，[第 3 版第 4 章](https://www.oreilly.com/library/view/programming-massively-parallel/9780128119877/xhtml/chp004.xhtml)）。CUDA 的经典教材。矩阵乘法只是其中一个例子，用来讲内存层次和分块；全书覆盖卷积、归约、前缀和、稀疏矩阵等大量并行模式，不以矩阵乘法为主线，也不追求逼近 cuBLAS。
- **NVIDIA CUTLASS 文档：Efficient GEMM in CUDA**（[链接](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient%5Fgemm.html)）。工业级 GEMM 的分层结构（threadblock / warp / thread 三级分块、双缓冲、Tensor Core MMA）。是权威参考，但写给已经懂行的人。
- **Google DeepMind，How to Scale Your Model，第 12 章 How to Think About GPUs**（[全书](https://jax-ml.github.io/scaling-book)）。2025-08 新增。从 LLM 规模化的角度讲 H100/B200 的 SM、内存层次、NVLink 拓扑和集合通信的成本模型，适合作为本书多 GPU 部分的参考。
- **NVIDIA Tilus 教程：Matmul (Hopper)**（[链接](https://nvidia.github.io/tilus/latest/tutorials/matmul-hopper/__init__.html)）与 **Colfax 的 CuTe Hopper GEMM 论文**（[PDF](https://research.colfax-intl.com/wp-content/uploads/2023/12/colfax-gemm-kernels-hopper.pdf)）。Hopper 上 TMA + WGMMA 的分步实现。

## 四、CPU 侧

- **matmul（Rust crate）的优化记录**（[链接](https://docs.rs/crate/matmul/latest/source/docs/OPTIMIZATION_JOURNEY.md)）。从朴素代码出发，依次做循环重排、cache 分块、packing、SIMD/FMA 微内核，最终达到 NumPy/OpenBLAS 的约一半。适合作为第 2 章“NumPy 为什么快”的参考。
- Simon Boehm 另有一篇 CPU 版本的矩阵乘法优化文章（Fast Multidimensional Matrix Multiplication on CPU from Scratch），可一并参考。

## 结论：这本书的定位

“以矩阵乘法为主线、一步步逼近 cuBLAS”这条路线已经有不少人走过，而且有质量很高的英文 worklog。但现有资料大致分成两类，中间有一块空白：

1. **worklog 类**：路线清晰、数字扎实，但默认读者会 CUDA、懂硬件，篇幅一篇文章，解释“为什么”的部分很压缩。
2. **教材类**：讲得全面，但矩阵乘法只是众多例子之一，读者看不到一条从慢到快的完整主线。

这本书可以占据的位置是：**面向初学者的中文书，用一个矩阵乘法从 CPU 一路讲到 Tensor Core、算子和多 GPU**。差异点包括：

- 从 CPU 和 NumPy 开始，先把“快”和“慢”的标准建立起来，worklog 通常跳过这一步。
- 硬件知识按需引入，每个概念都由一次具体的性能问题引出，并配图解释。
- 一套 benchmark 贯穿全书，读者随时知道自己离硬件峰值还有多远。
- 不停在 FP32 SGEMM，继续讲 Tensor Core、新架构特性、一般算子和多 GPU，回应“超级大的矩阵”这个出发点。
- 现有资料里的数字（如 Boehm 的各步骤 GFLOPS）可以作为读者自己复现时的参照。
