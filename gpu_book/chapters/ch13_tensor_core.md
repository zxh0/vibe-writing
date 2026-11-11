# 第 13 章　Tensor Core

> 本章要回答的问题：专门为矩阵乘法设计的硬件单元，该怎样使用？

## 本章目标

- 理解 Tensor Core 的工作方式
- 用 WMMA API 写出第一个 Tensor Core kernel
- 了解更底层的 mma 指令
- 把第三部分的分块结构迁移到 Tensor Core 上

## 小节

### 13.1 Tensor Core 是什么

- 一条指令完成一个小矩阵乘加
- 与 CUDA Core 的峰值对比
- 各代 GPU 的 Tensor Core 演进（简述）

### 13.2 WMMA API

- fragment 的概念
- 载入、计算、存回
- 第一个 Tensor Core 矩阵乘法 kernel

### 13.3 更底层的接口

- PTX 中的 mma 指令
- ldmatrix
- 这两条指令在 sm_75 上可用，T4 能做这部分实验
- fragment 在线程间的分布

### 13.4 迁移分块结构

- block 分块、warp 分块在 Tensor Core 下的对应
- 共享内存布局与 bank conflict
- 双缓冲（T4 上没有`cp.async`，沿用第 11 章的寄存器中转做法）

### 13.5 与 cuBLAS 对比

- FP16 下与 cuBLAS 的对比（T4 的 Tensor Core 不支持 BF16、TF32）
- 与第三部分 FP32 结果的对比

## 实验

- WMMA 版本的 TFLOPS
- 分块与双缓冲后的 TFLOPS
- cuBLAS FP16 的 TFLOPS

## 配图

- Tensor Core 一次运算的矩阵片段
- fragment 在 warp 内线程间的分布

## 基准里程碑

- 基准表新增：WMMA 版本、优化后的 Tensor Core 版本、cuBLAS 低精度

## 与后续章节的联系

- 手写 Tensor Core kernel 很复杂，第 14 章介绍工业界的做法与新架构特性
