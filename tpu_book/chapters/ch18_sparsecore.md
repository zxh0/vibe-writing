# 第 18 章　Embedding 与 SparseCore

> 本章要回答的问题：推荐模型里那些巨大的 embedding 表，为什么脉动阵列帮不上忙？要不要为它专门做一块硬件？

## 本章目标

- 理解 embedding 查表负载的特点
- 理解它为什么是访存问题而不是计算问题
- 理解 SparseCore 的定位

## 小节

### 18.1 推荐模型

- 用户、商品等离散特征通过 embedding 表变成向量
- 表的总量可以远超一颗芯片的 HBM，要分布在很多芯片上
- 在 Google 的负载中占有重要地位（v4 论文有统计）

### 18.2 脉动阵列为什么无能为力

- 查表是随机访存，几乎没有计算
- 跨芯片的 all-to-all 通信
- 用 TensorCore 做：大部分时间在等数据

### 18.3 SparseCore

- 专门做 gather/scatter 和小规模归约的小核心
- 靠近 HBM 和互连，处理稀疏、不规则的访问
- 与 TensorCore 并行工作

### 18.4 要不要专门留面积

- 收益：推荐模型大幅加速
- 代价：对其他负载没有用
- 又一次回到 DSA 的基本问题：押对负载

## 实验

- 用模拟器分别在 TensorCore 风格和 SparseCore 风格的硬件上模拟 embedding 查表，比较耗时

## 配图

- embedding 查表的数据流
- TensorCore 与 SparseCore 在芯片上的分工

## 对照 Google

- SparseCore 在 TPU v4 论文中首次详细介绍
- Ironwood 每颗芯片有四个 SparseCore（Cloud TPU 文档）

## 与后续章节的联系

- 第 19 章回顾整条演进路线
