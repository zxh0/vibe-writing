# 调研

> 工作草稿。2026年11月14日做的第一轮调研，大部分条目只看了首页、目录和二手介绍，标了“待核实”的数字和细节，写正文前要回到原文确认。

## 一句话结论

“从零写一个深度学习框架”这件事，材料已经很多，而且有几份做得相当好：Karpathy的micrograd和视频把标量自动求导讲透了，斋藤康毅的DeZero有中译本，MiniTorch、CMU的needle、哈佛的TinyTorch都是完整的课程。缺的不是“又一个从零写autograd的教程”，而是三样东西：一份**处处和真PyTorch对答案**的（API一致、梯度逐个比对，读者写完能直接拿真的torch脚本来跑）；一份**每章对照PyTorch真实设计**、讲清楚它为什么长成这样的；以及一份**和推理、训练两本书接上**的，最后训出来的模型能交给第一本的引擎去跑。

## 标量自动求导：入门的起点

这一类都只处理标量，代码极短，讲清楚了计算图和链式法则，但离“框架”还差张量这一大步。

* **Andrej Karpathy，micrograd**，以及配套视频**The spelled-out intro to neural networks and backpropagation: building micrograd**（Neural Networks: Zero to Hero系列第一讲，2022年）。一个`Value`类加一个两层的神经网络库，合起来一两百行（具体行数待核实）。这是这个话题事实上的“原点”，几乎所有后来的教程都从它出发。本书第三章的标量版会和它很像，要想清楚在它之外多讲什么。<https://github.com/karpathy/micrograd>
* **Christopher Olah，Calculus on Computational Graphs: Backpropagation**（2015年）。用图讲清楚前向模式和反向模式为什么一个适合“一个输入多个输出”，一个适合“多个输入一个输出”。本书第二章讲求导的几种办法时可以借它的图。<https://colah.github.io/posts/2015-08-Backprop/>
* **Baydin et al.，Automatic Differentiation in Machine Learning: a Survey**（arXiv:1502.05767）。自动微分的标准综述：数值微分、符号微分、自动微分的区别，前向和反向模式，运算符重载和源码变换两种实现方式。第二章的理论依据。

## 张量级的反向传播：最容易出错的一步

从标量走到张量，难点是广播、矩阵乘法和形状变换的梯度。这一类材料专门讲这些。

* **Karpathy，Building makemore Part 4: Becoming a Backprop Ninja**（2022年10月，约两小时）。不用`loss.backward()`，对一个带BatchNorm的两层MLP逐个张量手推梯度，再和PyTorch的结果比对。其中广播的反向“要在被复制的维度上求和”讲得最清楚。本书第七、八章的直接参考，它的“写完就和PyTorch对一下”也正是本书的做法。<https://github.com/karpathy/nn-zero-to-hero>
* **Karpathy，llm.c**（2024年）。用纯C训练GPT-2，CPU的fp32版本约一千行，每一层都手写了前向和反向，激活和梯度都和PyTorch参考实现比对。它是“不用框架，逐层手写反向”走到头的样子，本书第四部分写注意力和归一化的反向时，可以拿它的C代码当第二个参照。<https://github.com/karpathy/llm.c>
* **Terence Parr、Jeremy Howard，The Matrix Calculus You Need for Deep Learning**（arXiv:1802.01528）。把矩阵求导写成一份给工程师看的手册，第五章推矩阵乘法梯度时可以引用。

## 从零写一个框架：最直接的同类

这几份和本书的题目几乎一样。写作前要逐个读完目录，避免撞车，也要说清楚本书多了什么。

* **斋藤康毅，《深度学习入门2：自制框架》**（郑明智译，人民邮电出版社，2023年；日文原版是《ゼロから作るDeep Learning ③ フレームワーク編》，2020年）。从零写一个叫DeZero的框架，分60个步骤、五个阶段：自动求导、让代码写起来更自然、高阶导数、神经网络、GPU和模型保存。DeZero的API参照的是Chainer，不是PyTorch。**这是中文里和本书撞得最厉害的一本**，而且口碑很好（“鱼书”系列）。本书和它的区别要在定位里说清楚，见文末。（中译本书名里的序号各处写法不一，待核实）
* **Sasha Rush等，MiniTorch**（Cornell Tech机器学习工程课，约2020年起）。纯Python重新实现Torch的API，分五个模块：标量自动求导、张量（含stride和广播）、并行和GPU（用Numba）、卷积网络。学生填TODO、跑单元测试，最后的库能运行Torch代码。它的“张量”模块把storage、stride、index讲得很扎实，本书第九章可以参考。它是课程作业的形式，讲解在文档里，篇幅不大。<https://minitorch.github.io/>
* **CMU 10-414/714 Deep Learning Systems**（陈天奇、Zico Kolter）。五次作业从零写一个叫needle的框架：自动微分和基本算子、模块和优化器、NDArray后端（CPU和CUDA）、卷积和循环网络，最后一份是Transformer（作业的具体分法各年不同，待核实）。它把框架和后端分成两层，Tensor管计算图，NDArray管存储和计算，这个分层值得借鉴。课程视频和讲义都公开。<https://dlsyscourse.org/>
* **Vijay Janapa Reddi等，TinyTorch**（哈佛，2025年12月发布，论文arXiv:2601.19107）。哈佛Machine Learning Systems教材的配套课程，20个模块，用纯Python重写PyTorch的核心：张量、自动求导、优化器、CNN、Transformer，后面还有一组讲性能优化的模块。只要4GB内存，不需要GPU。**这是英文里和本书最像的一份**，而且很新，写作时要细读它的模块划分。<https://mlsysbook.ai/tinytorch>
* **fast.ai，Practical Deep Learning for Coders Part 2（2022年）**。课程名就是Deep Learning from the Foundations，从Python循环写矩阵乘法开始，一直讲到反向传播和混合精度训练，过程中写出一个叫miniai的框架。它是用PyTorch的张量往上搭，不重写张量本身。另外fastbook的第17章A Neural Net from the Foundations是同一思路的压缩版。<https://github.com/fastai/course22p2>
* **tinygrad和teenygrad**（George Hotz，tiny corp）。tinygrad是一个真在用的框架，核心思想是惰性求值加少量基本算子，再编译成各个后端的kernel。teenygrad是它的教学精简版，不到一千行就能训练MNIST，和tinygrad共用大部分前端代码。它代表了和PyTorch不同的另一条路（惰性、编译），本书最后一章讲“从动态图到编译”时可以拿来对照。<https://github.com/tinygrad/tinygrad>
* **JAX，Autodidax: JAX core from scratch**，以及新写的**Autodidax2**。从零实现JAX的核心：用解释器栈实现前向模式、反向模式、`vmap`和`jit`。它代表“函数变换”这条路，和PyTorch的“张量上挂计算图”很不一样，适合在第二章或最后一章做对照。<https://docs.jax.dev/en/latest/autodidax.html>
* **其他书和教程**：Andrew Trask的Grokking Deep Learning第13章写了一个小框架；Harrison Kinsley的Neural Networks from Scratch in Python（nnfs）只写前向和手推反向，不做自动求导；Leanpub上的PyTorch From Ground Up、Gumroad上的babygrad系列，以及几篇“1300行实现PyTorch”的博客，内容大同小异，不一一细读。

## PyTorch本身的设计

本书每章结尾“对照PyTorch”的依据。

* **Paszke et al.，PyTorch: An Imperative Style, High-Performance Deep Learning Library**（NeurIPS 2019，arXiv:1912.01703）。PyTorch的设计原则：以Python为先、动态图、引用计数及时释放显存、为什么不追求极致的速度而追求易用。本书第一章讲“框架要做什么”时可以从它出发。
* **Paszke et al.，Automatic Differentiation in PyTorch**（NIPS 2017 Autodiff Workshop）。更早的一篇短文，专讲PyTorch的自动求导：运算符重载、记录操作的“磁带”、in-place操作和版本计数器。
* **Edward Z. Yang，PyTorch internals**（2019年5月，同名演讲的长文版）。讲Tensor、Storage、stride和视图，讲算子怎么经过dispatcher分派到CPU、CUDA和Autograd。第九章的stride和最后一章的分派机制都以它为主要参考。<https://blog.ezyang.com/2019/05/pytorch-internals/>
* **PyTorch博客，Overview of PyTorch Autograd Engine**（2021年）及同系列的How Computational Graphs are Constructed in PyTorch、How Computational Graphs are Executed in PyTorch。讲`grad_fn`节点组成的反向图、引擎怎样调度、为什么只算向量和雅可比的乘积而不构造整个雅可比矩阵，以及每个算子的导数写在`tools/autograd/derivatives.yaml`里。第三、五章的对照依据。<https://pytorch.org/blog/overview-of-pytorch-autograd-engine/>
* **PyTorch官方文档**：Autograd mechanics（保存的张量、`no_grad`、in-place的正确性检查）、Extending torch.autograd（自定义`Function`）、`torch.autograd.gradcheck`、Broadcasting semantics、Tensor Views。写对应章节时逐篇读。
* **Ansel et al.，PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation**（ASPLOS 2024）。TorchDynamo和TorchInductor，最后一章讲“从动态图到编译”的依据。

## 优化器、初始化、归一化：各章引用的论文

这些是第三、四部分各章要引用的原始论文，按章节分组，写到时再读原文。

* 初始化：Glorot和Bengio，Understanding the difficulty of training deep feedforward neural networks（AISTATS 2010）；He et al.，Delving Deep into Rectifiers（arXiv:1502.01852）。
* 优化器：Kingma和Ba，Adam（arXiv:1412.6980）；Loshchilov和Hutter，Decoupled Weight Decay Regularization（arXiv:1711.05101，即AdamW）。
* 归一化：Ba et al.，Layer Normalization（arXiv:1607.06450）；Zhang和Sennrich，Root Mean Square Layer Normalization（arXiv:1910.07467）。
* 正则化：Srivastava et al.，Dropout（JMLR 2014）。
* 省显存：Chen et al.，Training Deep Nets with Sublinear Memory Cost（arXiv:1604.06174，梯度重计算）；Micikevicius et al.，Mixed Precision Training（arXiv:1710.03740，损失缩放）。

## 中文资料

中文里成体系的“自己写框架”主要就是DeZero的中译本，其余以课件和系列博客为主。

* **斋藤康毅，《深度学习入门：基于Python的理论与实现》**（陆宇杰译，人民邮电出版社，2018年）。“鱼书”第一本，手写前向和反向，不做自动求导。很多中文读者的入门书，本书可以假定读者读过或读过同类的书。
* **ZOMI酱，AISystem**（AI系统课程）。“AI框架”部分有一整套自动微分和计算图的课件：数值、符号、自动微分的区别，前向和反向模式，运算符重载和源码变换，动态图和静态图。形式是PPT，适合快速建立全景，不动手写。<https://github.com/chenzomi12/AISystem>
* **openmlsys，《机器学习系统：设计和实现》**（开源中文教材）。有编程接口、计算图、编译器前端（含自动微分）、编译器后端等章节，偏系统设计的概览。<https://openmlsys.github.io/>
* **《动手学深度学习》**（d2l）的自动微分一节，讲怎么用而不是怎么实现。
* **CSDN、知乎、掘金上的“从零实现深度学习框架”系列**。有一个基于纯Python和NumPy、仿PyTorch的系列（作者和仓库名待查），还有不少DeZero和needle的读书笔记、作业题解。质量参差，不逐一列出。

## 本书的位置

和toyllm系列第二、三本的想法一样，“又一个从零写autograd的教程”本身没有太大必要，micrograd、DeZero、MiniTorch、TinyTorch已经够用了。本书要占的是下面几个位置：

1. **处处对答案。** 这是toyllm系列第一本的做法：每章都和`transformers`逐位比对。框架同样能做到：pytoych的API刻意照抄PyTorch，同一份脚本把`import torch`换成`import pytoych as torch`就能跑；每个算子的前向和梯度都和真的torch比，误差标准说清楚。DeZero参照的是Chainer，MiniTorch和TinyTorch用单元测试，都没有把“和PyTorch逐个比对”当成全书的主线。
2. **每章对照PyTorch的真实设计。** 写完一个部件，就去看PyTorch在同一个问题上是怎么做的：`grad_fn`和`derivatives.yaml`、stride和视图、版本计数器、`no_grad`、AMP、`torch.compile`。读者读完不只是“会写一个玩具”，而是看真PyTorch的源码和文档时能对上号。这和tpu_book“对照Google的选择”、toyllm第三本“对照Claude Code”是同一个做法。
3. **接上toyllm系列。** 第一本讲推理，用的是PyTorch的前向；第二本讲训练，用的是PyTorch的反向和优化器；这一本把那个黑盒拆开。终点是用pytoych训练一个和第一本同样结构的小模型（RMSNorm、RoPE、SwiGLU、GQA），导出成第一本引擎认识的模型目录，让它在第一本的引擎里生成文字。别的教程停在MNIST或者一个独立的GPT，接不到任何地方。
4. **讲到语言模型的每一层。** micrograd、DeZero停在MLP和简单的CNN、RNN。本书要把第一本出现过的每一层的反向都亲手写一遍：嵌入、RMSNorm、RoPE、注意力（含因果掩码和GQA）、SwiGLU、交叉熵。
5. **中文，一章一个概念。** 读者是读过《人人能懂的人工神经网络》或者toyllm系列第一、二本、想知道`loss.backward()`里到底发生了什么的人。

## 待读

写对应章节之前要读完：

* [ ] micrograd的源码和视频（第三章）
* [ ] Backprop Ninja的notebook（第五到八章）
* [ ] DeZero的目录和代码（定位，避免撞车）
* [ ] MiniTorch的张量模块文档（第九章）
* [ ] needle的作业说明，尤其是NDArray后端那一份（第四、九章）
* [ ] TinyTorch的模块列表和论文（定位）
* [ ] Edward Yang的PyTorch internals（第九章、最后一章）
* [ ] PyTorch autograd的三篇博客和Autograd mechanics文档（第三、五、二十章）
* [ ] llm.c的`train_gpt2.c`（第四部分）
* [ ] PyTorch 2的论文和teenygrad的源码（最后一章）
