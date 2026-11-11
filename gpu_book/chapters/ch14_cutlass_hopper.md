# 第 14 章　CUTLASS 与新架构

> 本章要回答的问题：工业级的矩阵乘法是怎样组织的？新一代 GPU 又改变了什么？

## 本章目标

- 了解 CUTLASS 和 CuTe 的设计思路
- 了解 Hopper 架构为矩阵乘法新增的硬件特性
- 了解 Blackwell 的主要变化

## 小节

### 14.1 CUTLASS

- 为什么需要模板库
- 分层结构：与第三部分的三级分块一一对应
- 用 CUTLASS 实现一个矩阵乘法

### 14.2 CuTe

- layout 与 tensor 的抽象
- 用 layout 描述数据在线程和内存中的分布

### 14.3 Hopper 的新特性

- 异步拷贝`cp.async`（Ampere 引入）与 TMA：硬件负责的异步数据搬运
- WGMMA：以 warpgroup 为单位的矩阵乘法
- 线程块集群与分布式共享内存
- warp 特化：生产者与消费者

### 14.4 persistent kernel

- 常驻 SM、循环领取计算块
- 与第 11 章的调度顺序结合

### 14.5 Blackwell 的变化（概览）

- 更大的 Tensor Core 运算粒度
- 更低精度的格式支持

## 实验

- 用 CUTLASS 实现矩阵乘法并与 cuBLAS 对比
- 在 H100 上运行 TMA + WGMMA 示例（如有条件）

## 配图

- CUTLASS 分层结构与本书三级分块的对应
- Hopper 上生产者 / 消费者 warp 的流水线

## 基准里程碑

- 基准表新增：CUTLASS 版本、Hopper 版本（如有条件）

## 与后续章节的联系

- 矩阵乘法本身讲到这里结束，第五部分把视野扩展到其他算子和多 GPU
