# Agent循环（Agent Loop）

先写一个几十行的Agent，让它真的修好一个bug，再看它还缺什么。

## 我们每天看到的是什么

用Coding Agent时屏幕上发生了什么：模型在说话、在读文件、在跑命令。

## 什么是Agent

### 一个通行的定义

把Simon Willison的定义“An LLM agent runs tools in a loop to achieve a goal”逐词拆开。

### 循环在哪里停

模型不再要求调用工具，就是这一轮结束了。

## 最小的Agent

### 一个工具：bash

只给模型一个执行命令的能力。

### 一个循环

调模型、执行命令、把结果交回去，直到模型说做完了。

### 让它修一个bug

在一个小项目里埋一个bug，看它怎么一步步找到并修好。

## 模型和Harness

模型负责想，模型之外的一切都是Harness。本书写的就是Harness。

## 路线图

这个最小的Agent缺什么，后面每章补一样。

## 代码实现

## 代码效果

## 对照

极简到头的例子：mini-swe-agent只有bash一个工具，照样能在SWE-bench上拿高分。

## 本章小结
