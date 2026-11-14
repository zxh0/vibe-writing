# 存储与视图

转置`x.T`不复制数据。本章讲张量在内存里到底怎么放，以及视图和梯度的关系。

## 转置不复制数据

## stride

### 用纯Python实现一个迷你的Storage

### 下标怎样换算成偏移

## 视图

转置、切片、expand都只改stride和偏移。

## 连续和不连续

什么时候必须`contiguous()`。

## 回到NumPy

其实NumPy的数组本身就是storage加stride。

## 视图和in-place

## 代码实现

## 对答案

## 对照PyTorch

参考Edward Yang的PyTorch internals；MiniTorch的张量模块。

## 本章小结
