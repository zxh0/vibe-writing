# 附录B：引擎公式汇总

这个附录把推理引擎用到的公式，按数据流动的方向从头到尾排一遍，每一步都标上SmolLM2里的形状和数字。它对应第一章那张总览图，也就是本书的路线图。公式的来历和推导，请翻回注明的章节。

下表是用到的符号，以及它们在SmolLM2里的值。第十三章换上的Qwen3-0.6B，除了多一步QK归一化，公式完全一样，只是数不同，也列在旁边。注意 $d_k$ 不一定等于 $d / h$ ：SmolLM2的`config.json`里没有`head_dim`，按 $576 / 9 = 64$ 算；Qwen3写明了128，16个头加起来是2048维，比 $d = 1024$ 还宽，所以它的Q投影是 $[1024 \times 2048]$ ，输出投影是 $[2048 \times 1024]$ （第十三章）：

| 符号               | 含义                                          | SmolLM2-135M | Qwen3-0.6B |
| ------------------ | --------------------------------------------- | -----------: | ---------: |
| $T$                | 这一次送进模型的词元数                        |   随输入变化 | 随输入变化 |
| $V$                | 词表大小，`vocab_size`                        |        49152 |     151936 |
| $d$                | 词嵌入维度，`hidden_size`                     |          576 |       1024 |
| $h$                | Q头数，`num_attention_heads`                  |            9 |         16 |
| $n_\mathrm{kv}$    | KV头数，`num_key_value_heads`                 |            3 |          8 |
| $g$                | 每组几个Q头共用一组K和V， $h / n_\mathrm{kv}$ |            3 |          2 |
| $d_k$              | 每个头的维度，`head_dim`，没有就是 $d / h$    |           64 |        128 |
| $d_\mathrm{ff}$    | FFN中间层的宽度，`intermediate_size`          |         1536 |       3072 |
| $n_\mathrm{layer}$ | Decoder块的个数，`num_hidden_layers`          |           30 |         28 |
| $\epsilon$         | RMSNorm里防止除零的小数，`rms_norm_eps`       |        1e-05 |      1e-06 |
| $b$                | RoPE的底数，`rope_theta`                      |       100000 |    1000000 |

下面写的形状，都是一条序列的情况。第十四章让张量前面多了一维批次，比如主干从 $(T, 576)$ 变成 $(B, T, 576)$ ，B条序列一起算；这时只要在每个形状的最前面加上B，公式一个都不用改，因为每一步要么只看最后一维，要么只看最后两维。



## 一、入口：词嵌入（第四章）

词嵌入矩阵 $\mathbf{E}$ 是 $[V \times d]$ 的，按词元ID取出对应的行，得到 $[T \times d]$ 的 $\mathbf{X}$ ，形状从 $(T,)$ 变成 $(T, 576)$ 。这一步只是查表，没有计算：

$$
\mathbf{X} = \mathbf{E}[\mathrm{ids}]
$$



## 二、归一化：RMSNorm（第五章）

归一化对每个词元的 $d$ 个数单独做，和别的词元无关，形状不变，还是 $(T, 576)$ 。 $\boldsymbol{\gamma}$ 是训练出来的 $d$ 维权重，每个Decoder块两个，出口一个，一共 $2 n_\mathrm{layer} + 1 = 61$ 个。用FP16、BF16算时，要先升到FP32把均方根算完，再降回原来的精度，否则几百个数的平方加起来，不是溢出就是丢精度（第十四章）。注意 $\epsilon$ 在根号**里面**：

$$
\mathrm{RMSNorm}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}
$$



## 三、注意力（第六章、第七章）

注意力是整个Decoder块里唯一让词元之间互相看的部件，也是步骤最多的一个。它的输入是归一化之后的 $(T, 576)$ ，输出还是 $(T, 576)$ ，中间依次是：投影出Q、K、V，拆成多个头，给Q和K转上RoPE（Qwen3在这之前还要做一次QK归一化），每个头各算一遍注意力，最后把各个头拼回去，乘上输出投影。下面就按这个顺序写。



### 投影

三个权重矩阵里， $\mathbf{W}_Q$ 是 $[576 \times 576]$ ， $\mathbf{W}_K$ 和 $\mathbf{W}_V$ 都是 $[576 \times 192]$ ，权重文件里存的是它们的转置。乘完之后， $\mathbf{Q}$ 是 $(T, 576)$ ， $\mathbf{K}$ 、 $\mathbf{V}$ 是 $(T, 192)$ ：

$$
\mathbf{Q} = \mathbf{X} \times \mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{X} \times \mathbf{W}_K, \quad
\mathbf{V} = \mathbf{X} \times \mathbf{W}_V
$$



### 拆头

拆头就是沿着列切，每个头 $d_k = 64$ 维。 $\mathbf{Q}$ 从 $(T, 576)$ 变成 $(9, T, 64)$ ， $\mathbf{K}$ 和 $\mathbf{V}$ 从 $(T, 192)$ 变成 $(3, T, 64)$ ：

$$
[\mathbf{Q}_1, \ldots, \mathbf{Q}_h] = \mathbf{Q}, \quad
[\mathbf{K}_1, \ldots, \mathbf{K}_{h/g}] = \mathbf{K}, \quad
[\mathbf{V}_1, \ldots, \mathbf{V}_{h/g}] = \mathbf{V}
$$



### QK归一化（只有Qwen3等模型有，第十三章）

拆完头之后、RoPE之前，Q和K的每个头各自过一次RMSNorm，归一化的是每个头的 $d_k$ 维。SmolLM2没有这一步，Qwen3有：

$$
\mathbf{q} \leftarrow \mathrm{RMSNorm}(\mathbf{q}), \quad \mathbf{k} \leftarrow \mathrm{RMSNorm}(\mathbf{k})
$$



### RoPE

位置编码把每个头的 $d_k$ 维两两配对，配成 $d_k/2$ 个平面，按half-split配对，也就是第 $j$ 维和第 $j + d_k/2$ 维一组。第 $i$ 个平面的转速是 $\theta_i$ （ $i = 1, \ldots, d_k/2$ ），在第 $m$ 个位置上转 $m\theta_i$ 。实际计算时不乘旋转矩阵，而是用两次哈达玛积：

$$
\begin{aligned}
\theta_i &= b^{-2(i-1)/d_k} \\
\mathrm{RoPE}(\mathbf{x}, m) &= \mathbf{x} \odot \cos(m\boldsymbol{\theta}) + \mathrm{rotate\_half}(\mathbf{x}) \odot \sin(m\boldsymbol{\theta})
\end{aligned}
$$

其中 $\mathrm{rotate\_half}([\mathbf{a}, \mathbf{b}]) = [-\mathbf{b}, \mathbf{a}]$ ， $\mathbf{a}$ 和 $\mathbf{b}$ 是前后两半； $\cos(m\boldsymbol{\theta})$ 是把32个角度的余弦抄两遍拼成的64维向量， $\sin$ 同理。只转Q和K，不转V，形状不变。转完之后，第 $m$ 个位置的Q和第 $n$ 个位置的K的点积只和 $n - m$ 有关：

$$
(\mathbf{R}_m \mathbf{q})^\top (\mathbf{R}_n \mathbf{k}) = \mathbf{q}^\top \mathbf{R}_{n-m} \mathbf{k}
$$



### 每个头的注意力

第 $i$ 个Q头用第 $\lceil i/g \rceil$ 组K和V。因果掩码把未来的位置换成负无穷，Softmax按行做，分数表是 $(9, T, T)$ ，输出是 $(9, T, 64)$ 。下面第一行是整个头的写法，第二行是掩码，第三行把其中一行展开，是第 $a$ 个词元的输出：

$$
\begin{aligned}
\mathbf{O}_i &= \mathrm{softmax}\left(\mathrm{mask}\left(\frac{\mathbf{Q}_i \times \mathbf{K}^\top_{\lceil i/g \rceil}}{\sqrt{d_k}}\right)\right) \mathbf{V}_{\lceil i/g \rceil} \\
\mathrm{mask}(s_{ab}) &=
\begin{cases}
s_{ab}, & b \le a \\
-\infty, & b > a
\end{cases} \\
\mathbf{o}_a &= \sum_{b=1}^{a}
\frac{\exp(\mathbf{q}_a \cdot \mathbf{k}_b / \sqrt{d_k})}{\sum_{c=1}^{a} \exp(\mathbf{q}_a \cdot \mathbf{k}_c / \sqrt{d_k})}
\mathbf{v}_b
\end{aligned}
$$



### 合并与输出投影

各个头的输出左右拼回去，形状从 $(9, T, 64)$ 变回 $(T, 576)$ ，再乘上输出投影 $\mathbf{W}_O$ 。 $\mathbf{W}_O$ 是 $[576 \times 576]$ ，所以乘完还是 $(T, 576)$ ：

$$
\mathbf{O} = [\mathbf{O}_1, \ldots, \mathbf{O}_h], \quad \mathbf{Y} = \mathbf{O} \times \mathbf{W}_O
$$



## 四、前馈网络：SwiGLU（第八章）

前馈网络用三个矩阵： $\mathbf{W}_\mathrm{gate}$ 和 $\mathbf{W}_\mathrm{up}$ 都是 $[576 \times 1536]$ ， $\mathbf{W}_\mathrm{down}$ 是 $[1536 \times 576]$ ，形状从 $(T, 576)$ 升到 $(T, 1536)$ ，再降回 $(T, 576)$ 。 $d_\mathrm{ff} = \frac{8}{3} d = 1536$ ，让三个矩阵的参数和2017年两个4倍宽的矩阵一样多。下面第一行是SiLU，它只套在门那一路上：

$$
\begin{aligned}
\mathrm{SiLU}(z) &= z \cdot \sigma(z) = \frac{z}{1 + e^{-z}} \\
\mathrm{FFN}(\mathbf{x}) &= \big(\mathrm{SiLU}(\mathbf{x} \times \mathbf{W}_\mathrm{gate}) \odot (\mathbf{x} \times \mathbf{W}_\mathrm{up})\big) \times \mathbf{W}_\mathrm{down}
\end{aligned}
$$



## 五、Decoder块：残差连接（第五章、第九章）

采用pre-norm，每个子层的结果加回到它自己的输入上。两个RMSNorm各有各的 $\boldsymbol{\gamma}$ ，整个块重复 $n_\mathrm{layer} = 30$ 次，形状始终是 $(T, 576)$ ：

$$
\begin{aligned}
\mathbf{H}' &= \mathbf{H} + \mathrm{Attn}(\mathrm{RMSNorm}(\mathbf{H})) \\
\mathbf{H}_\mathrm{out} &= \mathbf{H}' + \mathrm{FFN}(\mathrm{RMSNorm}(\mathbf{H}'))
\end{aligned}
$$

把这个递推一层层展开，残差流上的每一处，都是词嵌入加上前面各个子层的贡献：

$$
\mathbf{h}_{l+1} = \mathbf{h}_1 + f_1(\mathbf{h}_1) + f_2(\mathbf{h}_2) + \cdots + f_l(\mathbf{h}_l)
$$



## 六、出口：线性投影（第四章、第五章）

等30层走完，先过出口那个RMSNorm，再取最后一个位置，乘上词嵌入矩阵的转置（权重共享），形状从 $(576,)$ 变成 $(49152,)$ 。第 $j$ 个分数，就是这个向量和第 $j$ 个词元的词嵌入的点积：

$$
\mathbf{z} = \mathrm{RMSNorm}(\mathbf{h}_T) \times \mathbf{E}^\top
$$



## 七、采样（第十章）

出口给出的是49152个分数，还不是概率，更不是下一个词元。采样要做的，就是从这些分数里挑出一个词元ID，接到输入后面，再跑下一圈。下面先讲Softmax怎么把分数变成概率，再讲抽之前改分数的三个过滤器，最后讲怎么抽出一个词元。



### Softmax

把分数变成概率靠Softmax。下面后一种写法是安全Softmax，先减去最大值再取指数，结果和前一种完全一样，只是不会溢出：

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}} = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^{V} e^{z_j - \max(\mathbf{z})}}
$$



### 三个过滤器

三个过滤器按温度、top-k、top-p的顺序排，分数进，分数出，下面三行依次是它们。这里 $T_\mathrm{temp}$ 是温度（为了和序列长度 $T$ 区分，加了下标）； $z_{(k)}$ 是第 $k$ 大的分数； $S$ 是把概率从小到大排好之后，从最小的开始累加、累加和不超过 $1 - p$ 的那些词元，概率最大的那个永远不在 $S$ 里：

$$
\begin{aligned}
z_i &\leftarrow z_i / T_\mathrm{temp} \\
z_i &\leftarrow
\begin{cases}
z_i, & z_i \ge z_{(k)} \\
-\infty, & z_i < z_{(k)}
\end{cases} \\
z_i &\leftarrow
\begin{cases}
z_i, & i \notin S \\
-\infty, & i \in S
\end{cases}
\end{aligned}
$$



### 抽取

最后从过滤完的分数里抽一个：温度为0时是贪婪采样，直接取分数最大的；否则按Softmax之后的概率随机抽。抽到EOS（SmolLM2里是0号词元）就停：

$$
\mathrm{id} =
\begin{cases}
\arg\max_i z_i, & \text{贪婪采样，} T_\mathrm{temp} = 0 \\
\text{按 } \mathrm{softmax}(\mathbf{z}) \text{ 随机抽一个}, & \text{随机采样}
\end{cases}
$$



## 八、KV缓存（第十一章）

缓存里已经存了 $L$ 个位置时，新来的 $T$ 个词元在第 $L$ 到第 $L + T - 1$ 个位置，RoPE从第 $L$ 个位置起算；它们的K和V（RoPE之后的）追加到缓存末尾，分数表变成 $(9, T, L + T)$ 。decode时 $T = 1$ ，不需要掩码：

$$
\mathbf{o} = \mathrm{softmax}\left(\frac{\mathbf{q} \times \mathbf{K}_\mathrm{cache}^\top}{\sqrt{d_k}}\right) \mathbf{V}_\mathrm{cache}
$$

缓存占的内存，和词元数成正比。按FP32算，SmolLM2每个词元46,080字节，8192个位置是377,487,360字节。换成FP16或BF16就减半；批处理时B条序列各有一份，再乘上B（第十四章）：

$$
\text{缓存字节数} = 2 \times n_\mathrm{kv} \times d_k \times n_\mathrm{layer} \times \text{每个数的字节数} \times \text{词元数}
$$



## 九、在线Softmax（第十二章）

把一行分数 $\mathbf{x}$ 和对应的 $\mathbf{v}$ 切成两块，每块各自记下最大值 $m_k$ 、分母 $\ell_k$ 和分子 $y_k$ ，这是下面第一行；合并时换到同一个基准 $m = \max(m_1, m_2)$ ，这是第二行。合并的结果和整行一次算完严格相等，不是近似。切成多少块都一样，每来一块就合并一次：

$$
\begin{aligned}
m_k &= \max_{i \in \text{第}k\text{块}} x_i, \quad
\ell_k = \sum_{i \in \text{第}k\text{块}} e^{x_i - m_k}, \quad
y_k = \sum_{i \in \text{第}k\text{块}} e^{x_i - m_k} v_i \\
a &= \frac{y_1 e^{m_1 - m} + y_2 e^{m_2 - m}}{\ell_1 e^{m_1 - m} + \ell_2 e^{m_2 - m}}
\end{aligned}
$$



## 十、参数量（第二章、第六章、第八章）

出口的线性投影和词嵌入共用一个矩阵，所以下表不另算。权重文件用BF16存，每个参数2字节；读进内存升成FP32，每个参数4字节，合计大约538MB：

| 部分              | 公式                                       |     SmolLM2 |
| ----------------- | ------------------------------------------ | ----------: |
| 词嵌入            | $V \times d$                               |  28,311,552 |
| 每块的注意力      | $d \times (2 h d_k + 2 n_\mathrm{kv} d_k)$ |     884,736 |
| 每块的FFN         | $3 \times d \times d_\mathrm{ff}$          |   2,654,208 |
| 每块的两个RMSNorm | $2 \times d$                               |       1,152 |
| 出口的RMSNorm     | $d$                                        |         576 |
| 合计              | 词嵌入 + $n_\mathrm{layer}$ × 每块 + 出口  | 134,515,008 |
