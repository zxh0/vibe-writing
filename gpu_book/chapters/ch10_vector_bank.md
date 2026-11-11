# 第 10 章　向量化访存与 bank conflict

> 本章要回答的问题：数据搬运的细节里，还藏着哪些浪费？

## 本章目标

- 掌握向量化访存
- 理解 共享内存 的 bank 结构和 bank conflict
- 学会用转置和 padding 消除冲突

## 小节

### 10.1 向量化访存

- 一条指令读取 128 位
- 使用 float4 的条件：对齐
- 对全局内存与共享内存的影响

### 10.2 共享内存 的 bank

- 共享内存 被分成多个 bank
- 地址如何映射到 bank
- 同一 warp 访问同一 bank 的不同地址会怎样

### 10.3 发现 bank conflict

- 在 Nsight Compute 中查看冲突指标
- 分析当前 kernel 的访问模式

### 10.4 消除 bank conflict

- 在载入 共享内存 时转置 A
- padding
- swizzle（简述）

## 实验

- 向量化前后的 GFLOPS
- 消除冲突前后的 bank conflict 指标与 GFLOPS

## 配图

- 共享内存 的 bank 映射
- 有冲突与无冲突的访问模式对比

## 基准里程碑

- 基准表新增：向量化版本、消除 bank conflict 版本

## 与后续章节的联系

- 线程级优化基本到位，第 11 章从 warp 层面重新组织计算
