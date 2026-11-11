# 大纲

全书围绕一个问题：**一个超级大的矩阵乘法，怎样才能算得快？** 全书分五部分。第一部分在 CPU 上把问题和衡量标准建立起来；第二部分进入 GPU，写出第一个能跑的 kernel；第三部分把 FP32 矩阵乘法一步步优化到接近 cuBLAS；第四部分用 Tensor Core 和低精度再上一个台阶；第五部分从矩阵乘法推广到一般算子，最后回答矩阵大到一张卡放不下时怎么办。

每章末尾都在同一张基准表里补一行，记录当前实现的 GFLOPS 和它占硬件峰值的百分比。

本文件是全书的总览。各章更细的大纲（目标、小节要点、实验、配图）见 [chapters/](chapters/) 目录，以各章文件为准。

## 第一部分　问题与基线

### [第 1 章　矩阵乘法](chapters/ch01_matmul.md)

- 1.1 定义：C = A × B，每个元素是一行与一列的点积
- 1.2 计算量与数据量：2MNK 次浮点运算，(MK + KN + MN) 个元素
- 1.3 矩阵乘法为什么重要：神经网络里的线性层、注意力，绝大多数计算都是矩阵乘法
- 1.4 行主序与列主序：矩阵在内存里怎么摆放
- 配图：矩阵乘法示意、行主序与列主序的内存布局

### [第 2 章　在 CPU 上算矩阵乘法](chapters/ch02_cpu.md)

- 2.1 Python 三重循环，以及怎样测出 GFLOPS
- 2.2 换成 C：同样的算法快了多少，为什么
- 2.3 循环顺序：ijk 与 ikj，第一次感受到“访存”比“计算”更重要
- 2.4 SIMD：小矩阵快了 5 到 6 倍，大矩阵却没变化，瓶颈转到访存
- 2.5 缓存与分块（blocking）：让数据在快的地方被反复使用
- 2.6 多线程
- 2.7 NumPy 为什么快：BLAS、packing、微内核
- 实验：在本机测出每个版本的 GFLOPS，NumPy 作为 CPU 的天花板
- 伏笔：分块的思想会在第 8 章原样出现在 GPU 上

### [第 3 章　性能的天花板](chapters/ch03_roofline.md)

- 3.1 峰值算力：怎样从硬件参数算出 FLOPS 上限
- 3.2 内存带宽：数据搬运的上限
- 3.3 算术强度：每搬一个字节做多少次运算
- 3.4 Roofline 模型：一张图判断瓶颈在计算还是访存
- 3.5 矩阵乘法与向量加法在 roofline 上的位置：为什么矩阵乘法“天生”适合加速
- 配图：roofline 模型，标出矩阵乘法与向量加法

## 第二部分　走进 GPU

### [第 4 章　为什么是 GPU](chapters/ch04_why_gpu.md)

- 4.1 CPU 的设计取舍：大缓存、乱序执行、分支预测，为降低延迟而生
- 4.2 GPU 的设计取舍：海量简单核心，为提高吞吐而生
- 4.3 GPU 的整体结构：SM、CUDA Core、Tensor Core、L2、HBM
- 4.4 CPU 与 GPU 如何协作：主机、设备、PCIe
- 4.5 用第 3 章的方法算出实验 GPU 的峰值算力和带宽
- 配图：CPU 与 GPU 的芯片面积对比、GPU 整体结构

### [第 5 章　第一个 CUDA 程序](chapters/ch05_first_cuda.md)

- 5.1 实验环境：Colab 的 T4，nvcc，nvidia-smi
- 5.2 主机与设备：cudaMalloc、cudaMemcpy、全局内存，主机与设备之间复制数据的代价
- 5.3 CUDA 编程模型：kernel，线程、block、grid，内置索引变量，自动可扩展性
- 5.4 第一个 kernel：向量加法
- 5.5 朴素矩阵乘法 kernel：一个线程算 C 的一个元素，x 对应行
- 5.6 正确性验证：与 CPU 双精度结果对比，浮点误差与容差
- 5.7 正确地计时：kernel 异步启动、CUDA event、预热
- 5.8 朴素 kernel 有多快：不同规模、不同 block 大小（8×8 比 32×32 快 2.7 倍）
- 5.9 实测 T4 的天花板：峰值算力 7544.3 GFLOPS，显存带宽 277.5 GB/s
- 5.10 cuBLAS：我们的目标，列主序的调用技巧
- 配图：grid、block 与线程的层次，朴素 kernel 的线程映射

### [第 6 章　线程是怎样执行的](chapters/ch06_execution.md)

- 6.1 Warp 与 SIMT
- 6.2 block 如何分配到 SM：实测 block 的分轮执行
- 6.3 访存延迟有多长：用指针追逐实测共享内存、L1、L2、显存的延迟
- 6.4 用并发隐藏延迟：warp 数与带宽、算力，Little 定律
- 6.5 分支分歧（warp divergence）
- 6.6 Occupancy：寄存器、共享内存与能同时驻留的 warp 数，occupancy 不是越高越好
- 6.7 内存层次全景：寄存器、共享内存、L1、L2、全局内存的容量、延迟与作用范围
- 6.8 回到朴素 kernel：排除 occupancy、分支分歧和 block 分配
- 配图：block 到 SM 与 warp 的划分、warp 调度时间线、内存层次金字塔

## 第三部分　把矩阵乘法优化到接近 cuBLAS（FP32）

每一章都遵循同一个节奏：测量 → 用 profiler 找到瓶颈 → 解释对应的硬件机制 → 改代码 → 再测量。

### [第 7 章　访存合并](chapters/ch07_coalescing.md)

- 7.1 第一次使用 Nsight Compute：Speed Of Light，朴素 kernel 的 L1 吞吐 99%
- 7.2 全局内存是怎样被访问的：请求、sector、合并与广播，步长与偏移的微基准
- 7.3 朴素 kernel 的问题：每个请求 16.5 个 sector，block 换成 8×8 后降到 4.5 个
- 7.4 调整线程映射：x 对应列，每个请求 2.5 个 sector，提速 9 到 12 倍
- 配图：一个 warp 的访问涉及几个 sector，两种映射下 warp 访问的位置

### [第 8 章　共享内存分块](chapters/ch08_smem_tiling.md)

- 8.1 数据复用：访存合并后的 kernel 读的数据几乎都在 L1 里，代价是 5 亿多个读请求，而不是显存流量
- 8.2 共享内存：属于一个 block、由程序员管理的片上存储
- 8.3 分块矩阵乘法 kernel：读全局内存的请求降到 1/32，n=4096 时 877.6 GFLOPS，是访存合并的 1.5 倍
- 8.4 两次同步：`__syncthreads()`，去掉任何一次都会算错且不报错
- 8.5 边界处理：搬运时补零
- 8.6 分块大小：8、16、32，分块 8 比不分块还慢
- 8.7 回到 roofline：算术强度 T/4，瓶颈不在显存而在共享内存的读取
- 8.8 证实功耗上限：连续运行时 cuBLAS 的主频降到 750MHz 左右
- 配图：分块在 A、B、C 上的滑动过程，block 内线程与共享内存的关系

### [第 9 章　寄存器分块](chapters/ch09_reg_tiling.md)

- 9.1 新的瓶颈：共享内存的带宽
- 9.2 一个线程算一列结果（1D 分块）
- 9.3 一个线程算一个小块结果（2D 分块）与外积，回扣第 2 章的微内核
- 9.4 寄存器压力与 occupancy 的权衡
- 配图：线程级分块的外积示意

### [第 10 章　向量化访存与 bank conflict](chapters/ch10_vector_bank.md)

- 10.1 一次读 128 位：float4
- 10.2 共享内存的 bank 与 bank conflict
- 10.3 转置存储与 padding
- 配图：bank 的映射与冲突

### [第 11 章　Warp 分块与流水线](chapters/ch11_warp_tiling.md)

- 11.1 三级分块：block、warp、thread
- 11.2 双缓冲：计算当前块时预取下一块（T4 没有`cp.async`，用寄存器中转）
- 11.3 调度顺序与 L2：block swizzle
- 11.4 参数搜索与自动调优
- 11.5 与 cuBLAS 对比：先查明 n=4096 时 cuBLAS 的异常，再看剩下的差距在哪里
- 小结：回顾第三部分的完整基准表
- 配图：三级分块结构、双缓冲时间线

## 第四部分　Tensor Core 与低精度

### [第 12 章　浮点格式与精度](chapters/ch12_float_formats.md)

- 12.1 FP32、TF32、FP16、BF16、FP8 的位布局（T4 硬件只支持 FP16，其余格式用软件模拟舍入）
- 12.2 精度与范围的取舍
- 12.3 混合精度：低精度相乘，高精度累加
- 配图：各种浮点格式的位布局对比

### [第 13 章　Tensor Core](chapters/ch13_tensor_core.md)

- 13.1 Tensor Core 是什么：一条指令完成一个小矩阵乘法
- 13.2 WMMA API：第一个 Tensor Core kernel
- 13.3 更底层的 mma 指令与 ldmatrix
- 13.4 把第三部分的分块结构搬到 Tensor Core 上
- 13.5 与 cuBLAS（FP16）对比
- 配图：Tensor Core 的矩阵片段（fragment）布局

### [第 14 章　CUTLASS 与新架构](chapters/ch14_cutlass_hopper.md)

- 14.1 CUTLASS 与 CuTe：工业级 GEMM 的结构
- 14.2 Hopper：TMA、WGMMA、异步与 warp 特化
- 14.3 Blackwell 的变化（概览）
- 14.4 persistent kernel

## 第五部分　从矩阵乘法到算子，再到多 GPU

### [第 15 章　算子](chapters/ch15_operators.md)

- 15.1 什么是算子：深度学习框架里的基本计算单元
- 15.2 访存受限的算子：逐元素运算、归约、softmax、LayerNorm
- 15.3 算子融合：少搬一次数据就是快
- 15.4 例子：融合的 attention（FlashAttention 的思路）

### [第 16 章　用 Triton 写算子](chapters/ch16_triton.md)

- 16.0 先实测 T4 能否运行 Triton（官方支持从计算能力 8.0 起）
- 16.1 Triton 的编程模型：以 block 为单位思考
- 16.2 用 Triton 重写矩阵乘法
- 16.3 与手写 CUDA、cuBLAS 对比

### [第 17 章　当矩阵放不下一张卡](chapters/ch17_multi_gpu.md)

- 17.1 显存容量的上限
- 17.2 分布式矩阵乘法：按行、按列、按块切分
- 17.3 GPU 之间的通信：NVLink、PCIe、网络，以及 NCCL
- 17.4 集合通信：AllReduce、AllGather、ReduceScatter 及其成本
- 17.5 计算与通信重叠
- 配图：矩阵切分到多卡的几种方式

## 附录

- [附录](chapters/appendix.md)
- A　实验环境搭建：Colab、云 GPU、本书代码仓库的用法
- B　本书基准测试的完整数据
- C　参考资料（见 [research.md](research.md)）

## 待定事项

- **主实验 GPU 选哪一张**：Colab 免费的 T4（Turing）没有 BF16 和 TF32；A100 或 H100 更能代表当前主流，但需要付费租用。第 14 章的 Hopper 内容至少需要 H100。
- **第 2 章是否深入 CPU 优化**：可以只讲到循环顺序和分块，把 SIMD、packing 留作简述，避免喧宾夺主。
- **代码组织**：每章一个目录，还是一个统一的仓库加上按章节编号的 kernel。
