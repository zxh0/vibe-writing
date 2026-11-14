# 注意力

注意力、RoPE和SwiGLU的反向。写完本章，一个完整的Block就能求导了。

## softmax的反向

## 因果掩码

## RoPE的反向

旋转的逆就是反方向旋转。

## 多头和GQA

共享的K、V，梯度要加回去。

## SwiGLU的反向

## 一个完整的Block

## 代码实现

## 对答案

和torch逐层比。

## 对照PyTorch

第一本的注意力、位置编码和前馈网络三章；`F.scaled_dot_product_attention`。

## 本章小结
