# 自己动手写深度学习框架

<BookSubtitle book="framework" />

::: warning 写作计划
书名待定，目前还是写作计划，各章只有框架，没有发布的章节。大纲在仓库的
[toyllm_book4/](https://github.com/zxh0/vibe-writing/tree/main/toyllm_book4) 下。
:::

## 这本书想做什么

天天写 `loss.backward()` 和 `optimizer.step()`，大致也知道背后是反向传播：链式法则，从输出往回一层一层乘。但「大致知道」和「亲手写过」差得很远。广播过的张量，梯度怎么缩回原来的形状？`x.T` 为什么不复制数据，它的梯度又从哪来？`with torch.no_grad()` 到底省了什么？所以这本书还是 toyllm 系列的老办法：只用 NumPy，从零写一个玩具版的 PyTorch，叫 pytoych，一点一点来，每一步都看得见、跑得通。

它和别的「从零写 autograd」材料最大的不同，是处处和真的 PyTorch 对答案：API 照抄 PyTorch，每个算子的梯度都和 torch 比对，同一份训练脚本把 `import torch` 换成 `import pytoych as torch` 就能跑。

这是 toyllm 系列的第四本，补的是「框架」这一层。第一本用 PyTorch 的张量和前向手写推理引擎，第二本用 PyTorch 的反向和优化器训练小模型，这一本把它们用到的那个黑盒拆开。终点和第一本接上：用 pytoych 训练一个和第一本同样结构的小模型，导出成第一本引擎认识的模型目录，让它在第一本的引擎里生成文字。

## 目录

<BookToc book="framework" />
