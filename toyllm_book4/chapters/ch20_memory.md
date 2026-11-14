# 内存

计算图要保存中间结果，训练比推理费内存得多。本章看pytoych的内存花在哪里，怎样省。

## 计算图占了多少内存

## no_grad和detach

## 反向完就释放

顺便看`retain_graph`是做什么的。

## in-place运算和版本计数器

## 梯度重计算

用时间换内存。

## 代码实现

## 对照PyTorch

看PyTorch的saved tensors和`torch.utils.checkpoint`。

## 本章小结
