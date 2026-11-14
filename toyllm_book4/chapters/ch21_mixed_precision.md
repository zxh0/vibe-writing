# 混合精度

用半精度训练能省一半内存、在GPU上快得多，但梯度会下溢。本章用NumPy模拟一遍。

## fp16和bf16

## 用NumPy模拟半精度训练

## 损失缩放

## 主权重留在fp32

## 代码实现

## 对照PyTorch

看PyTorch的`torch.autocast`和`GradScaler`；第二本在T4上为什么只能用fp16。

## 本章小结
