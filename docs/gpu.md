# GPU学习笔记

> 一个超级大的矩阵乘法，怎样才能算得快？

::: warning 构思和初稿
边学 GPU 边整理成一本书，前几章有了初稿，其余还是大纲，目前没有发布的章节。草稿在仓库的
[gpu_book/chapters/](https://github.com/zxh0/vibe-writing/tree/main/gpu_book/chapters) 下，大纲见
[outline.md](https://github.com/zxh0/vibe-writing/blob/main/gpu_book/outline.md)。
:::

![GPU学习笔记](/gpu/cover.png)

## 这本书想做什么

整本书围绕一个问题展开：**一个超级大的矩阵乘法，怎样才能算得快？**

GPU 擅长的是大规模数据并行和高吞吐。矩阵乘法特别适合 GPU，原因在于它的计算量是 O(n³)，数据量却只有 O(n²)，同一份数据会被反复使用，计算单元可以持续工作。所以这本书真正要回答的问题是：**怎样让计算单元不等数据**。把一个矩阵乘法从最朴素的版本一步步优化到接近 cuBLAS，几乎会用到 GPU 的全部核心概念：并行模型、内存层次、访存合并、shared memory、寄存器分块、Tensor Core。算术强度和 roofline 模型会尽早引入，作为解释每一步优化的统一工具。

一条基准贯穿全书：第一章就建立 benchmark，之后每章都用同一套基准测量，记录当前达到硬件峰值的百分比，让进步看得见。硬件知识按需引入：先写出代码，测量后发现慢，再问为什么慢，此时才引出对应的硬件概念。

## 目录

<BookToc book="gpu" />

## 小工具

写书时顺手做的几个纯前端单页演示，矩阵乘法、循环顺序、GPU 架构、T4 的结构、CUDA 的线程组织，打开就能用。见[小工具](/gpu/tools)页。
