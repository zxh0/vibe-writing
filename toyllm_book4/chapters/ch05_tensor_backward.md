# 张量的反向传播

标量损失对一个张量求梯度，梯度和张量同形状。本章把第三章的自动求导搬到张量上。

## 梯度的形状

## 向量和雅可比矩阵的乘积

为什么从不真的构造雅可比矩阵。

## Function

### forward：保存反向要用的东西

### backward：算输入的梯度

## 逐元素运算和归约的反向

## 叶子节点和.grad

## 代码实现

## 对答案

## 对照PyTorch

对应PyTorch的`torch.autograd.Function`、`ctx.save_for_backward`和`derivatives.yaml`。

## 本章小结
