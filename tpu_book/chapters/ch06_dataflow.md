# 第 6 章　数据流：谁站着不动

> 本章要回答的问题：矩阵乘法有三份数据——权重、输入、部分和。让哪一份留在 PE 里不动，对能耗和带宽影响有多大？

## 本章目标

- 理解 weight / output / input / row stationary 四种数据流
- 会分析每种数据流下各类数据的复用次数和访存量
- 理解 TPU 选择 weight stationary 的理由

## 小节

### 6.1 三份数据

- 以 C = A × B 为例，标出每份数据在循环中被谁使用
- 数据流的本质：循环嵌套的顺序 + 哪个循环在空间上展开

### 6.2 Weight stationary

- 权重预先装进阵列，每个 PE 存一个权重
- 输入从左向右流动，部分和从上向下累加
- 一次装载、用很多次：适合 batch 大的场景

### 6.3 Output stationary

- 每个 PE 负责 C 的一个元素，部分和留在 PE 里
- 权重和输入都流动
- 不需要把部分和传来传去，但需要两路输入

### 6.4 Input stationary 与 row stationary

- Input stationary：与 WS 对称
- Row stationary（Eyeriss）：为卷积优化，把一行卷积放在一个 PE 里
- 数据流的分类方法：参考 Sze 等人的 Efficient Processing of Deep Neural Networks

### 6.5 复用分析

- 每种数据流下，权重、输入、部分和各被读写多少次
- 结合第 2 章的能耗表，估算总能耗
- 结论随 batch 大小和矩阵形状变化

### 6.6 为什么 TPU 选 weight stationary

- 推理时同一份权重被整个 batch 复用
- 权重装载可以和计算重叠（双缓冲，第 8 章）
- 代价：batch 很小时，装一次权重只用几次，效率低

## 实验

- 在模拟器里分别实现 WS 和 OS 阵列
- 统计两种阵列在不同 batch、不同矩阵形状下的 SRAM 读写次数，换算成能耗

## 配图

- 四种数据流的示意
- WS 与 OS 阵列的数据流动对比
- 复用次数对比表、能耗随 batch 变化的曲线

## 对照 Google

- TPU 系列一直采用 weight stationary 的脉动阵列
- Gemmini 可以在运行时切换 OS 和 WS，说明这是一个值得探索的设计维度

## 与后续章节的联系

- 第 7 章讨论阵列尺寸和大矩阵分块
- 第 8 章的 Weight FIFO 和双缓冲，就是为了让 WS 的权重装载不拖后腿
