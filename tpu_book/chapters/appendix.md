# 附录

## 附录 A　用 Verilog 写一个脉动阵列

> 目标：把第 4–6 章的 PE 和阵列用硬件描述语言写出来，并用开源工具仿真，与 Python 模拟器的结果逐拍对照。

- A.1 工具链：Verilator、cocotb、GTKWave，在 Mac 上安装
- A.2 一个 PE：int8 乘法、int32 累加、向右向下传递
- A.3 用 generate 循环搭出 N×N 阵列
- A.4 testbench：用 cocotb 从 Python 送入数据、读出结果，与 NumPy 比对
- A.5 看波形：对照第 5 章的时序图
- A.6 延伸：加上 weight FIFO 和简单控制；综合到 FPGA（可选）
- 参考：tiny-tpu-v2、deaneeth/tiny-tpu、tinytinyTPU、Gemmini

## 附录 B　模拟器代码导读

- B.1 整体结构：PE、阵列、存储、控制、多芯片
- B.2 各章对应的代码版本
- B.3 怎样添加一个新的部件或指令
- B.4 已知的简化与不准确之处

## 附录 C　术语表

- 中英对照：MAC、PE、systolic array、MXU、TensorCore、VPU、VMEM、HBM、ICI、OCS、SparseCore、weight stationary 等
- 特别说明：TPU 的 TensorCore 与 NVIDIA 的 Tensor Core 的区别

## 附录 D　参考资料

- 见 [research.md](../research.md)
