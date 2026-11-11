# 模型（Model）

上一章把语料变成了一串整数：九亿多个词元，存在`train.bin`里。这一章要搭的，是吃下这串整数、学着预测下一个词元的模型。

模型的架构，第一本已经从头到尾写过一遍。我们的模型和SmolLM2是同一个架构，也就是Llama的架构，只是小得多。所以这一章不会再从矩阵乘法讲起，而是只讲一个问题：一个只会推理的模型，要改哪些地方，才能拿来训练？改动其实不多，一共四处：多了一个batch维，去掉了KV缓存，权重变成了可以学习的参数，还要自己给权重一个初始值。其中最后一处值得多说几句，初始值给得不好，训练一开始就会出问题。

架构改完，还要定下模型的尺寸，并把训练好的模型存成第一本的引擎能读的格式。最后，在真正开始训练之前，我们要先确认这个模型写对了：把SmolLM2-135M的权重装进来，算出的loss和梯度，要和`transformers`库对上。第一本每写完一章，都要和`transformers`对一次答案，这一本也保持这个习惯。

这一章的代码在`code/book/ch04/`下面，一共六个文件。`config.py`记录模型的配置；`model.py`是模型本身；`init_loss.py`比较几种初始化方法；`check_smollm2.py`装入SmolLM2的权重，和`transformers`对答案；`export.py`把模型存成一个目录，再读回来；`check_export.py`检查`transformers`能不能读这个目录。本章新加了两个依赖：读写权重文件用的`safetensors`，以及只在核对时用的`transformers`。对答案要用到SmolLM2-135M的权重，先在`code/`目录下运行`./download_model.sh`把它下载下来，一共二百七十多MB。



## 从第一本带过来什么

先把第一本里已经讲过的部件点一遍名。它们的数学一点都不改，这一章只是换一种写法，具体细节可以翻回第一本对应的章节去看。（TODO：是否要提一下整体架构：decoder-only Transformer）

* **词嵌入**（第一本第四章）：一张表，每个词元占一行。词元编号进来，按编号取出那一行，就是这个词元的向量。
* **RMSNorm**（第一本第五章）：每个词元的向量，除以它自己的均方根，再乘一个可学习的权重。和LayerNorm相比，少了减均值和偏置两步。
* **注意力**（第一本第六章）：整个模型里，只有这一步能让词元之间互相看见。我们用的是分组查询注意力，也就是GQA（Grouped-Query Attention），几个查询头共用一组键和值。
* **RoPE**（第一本第七章）：旋转位置编码。查询和键按各自所在的位置旋转一个角度，两个词元之间的距离，就体现在旋转角度的差里。
* **SwiGLU**（第一本第八章）：前馈网络。两路并行升维，其中一路过SiLU激活函数当开关，两路逐元素相乘，再降回原来的维度。
* **残差连接**（第一本第九章）：注意力和前馈网络的输出，都不直接替换输入，而是加回到输入上。

这几样东西拼起来，就是Llama架构的一层：先RMSNorm，再注意力，加回残差；再RMSNorm，再前馈网络，加回残差。若干层叠起来，最后再过一次RMSNorm，乘上输出层的矩阵，就得到了每个位置上，词表里每个词元的分数，也就是logits。

还有一点也和第一本一样：我们用的是half-split的RoPE，第 $i$ 维和第 $i+32$ 维配成一对，而不是相邻的两维配成一对。这是`transformers`的写法，也是SmolLM2的`config.json`里`rope_interleaved`为`false`的意思。第一本说过，两种配对都能用，用错了也不报错，只有和`transformers`对答案才分得出来。



## 训练要改什么

第一本的模型只做推理，它的写法是为推理服务的。拿来训练，有四个地方要改。



### batch维

推理时，模型一次处理一条序列，张量的形状是`(T, d)`， $T$ 是序列长度， $d$ 是隐藏层维度。第一本第十四章为了在GPU上一次处理好几条序列，已经让张量前面可以多出几维，不过那只是一个附加的功能。

训练时，一次处理很多条序列是常态。每一步，我们从语料里截下 $B$ 段，每段 $T$ 个词元，叠成一个`(B, T)`的整数张量送进模型。这 $B$ 段叫一个**批次**（Batch），张量最前面那一维叫**batch维**。一次算一个批次，一是GPU的并行能力能用满，二是 $B$ 段文本的loss取平均，算出来的梯度更稳。所以训练用的模型，所有中间结果都是三维的`(B, T, d)`，拆成多头之后是四维的`(B, 头数, T, 64)`。



### 不要KV缓存

第一本第十一章花了一整章讲KV缓存：生成时，每次只多一个新词元，前面那些词元的键和值不会变，存起来，下次直接用。

训练用不着它。训练时，一整段序列是一次性给定的，模型要同时预测每个位置的下一个词元：看到第1个词元，预测第2个；看到前2个，预测第3个……一直到看到前 $T-1$ 个，预测第 $T$ 个。这 $T$ 个预测，在一次前向计算里就全部做完了，靠的是因果掩码：第 $t$ 个位置只能看见它自己和它前面的词元，看不见后面的。这和第一本里处理提示词的那一步完全一样，只不过训练时每一步都是这样的整段计算，不存在“一次只来一个新词元”的情况。

所以，训练用的注意力反而比推理用的简单。下面是我们的`Attention.forward`：

```python
    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        h, kv, hd = self.n_heads, self.n_kv_heads, self.head_dim   # 8, 2, 64
        # 拆头，把头挪到第1维：(B, T, 512) → (B, 8, T, 64)
        q = self.q_proj(x).view(B, T, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, kv, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, kv, hd).transpose(1, 2)

        # RoPE：训练时序列总是从位置0开始，取表的前T行就行。
        q = q * cos[:T] + rotate_half(q) * sin[:T]
        k = k * cos[:T] + rotate_half(k) * sin[:T]

        # GQA：每个kv头连续复制4份，对齐8个q头，(B, 2, T, 64) → (B, 8, T, 64)。
        k = k.repeat_interleave(h // kv, dim=1)
        v = v.repeat_interleave(h // kv, dim=1)

        # 整段序列一起算，因果掩码交给is_causal，一次算完，不需要KV Cache。
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, -1)        # (B, T, 512)
        return self.o_proj(out)
```

和第一本比，少了缓存，少了“从第几个位置开始”的那个偏移量，注意力的最后一段也直接交给了PyTorch自带的`F.scaled_dot_product_attention`。这个函数第一本第十四章用过，它在GPU上会自动挑最快的实现，比如第十二章讲的FlashAttention。注意力的分数表、掩码、softmax，第一本已经手写过，这里就不再重复了。

不过，训练好的模型还是要能生成文字的，后面几章要定期看看模型写得怎么样了。为此，模型里还有一个简单的`generate`方法：每生成一个新词元，就把整段序列从头重算一遍。没有KV缓存，很慢，但训练中途看一眼生成效果，每次只生成几十个词元，慢一点无所谓。正式的推理，交给第一本的引擎。



### 从张量到参数

第一本的权重，是从文件里读出来的一堆普通张量，放在一个字典里，算的时候拿出来乘。推理时这样就够了，权重从头到尾不会变。

训练不一样。训练要不断地改权重，这就需要两样东西：一是每个权重都要有梯度，二是优化器要能找到所有的权重。PyTorch为此准备了`nn.Parameter`和`nn.Module`。一个张量包成`nn.Parameter`，PyTorch就知道它是要学习的，反向传播时会替它算梯度；`nn.Parameter`挂在`nn.Module`上，`model.parameters()`就能把它们全部列出来，交给优化器。我们直接用现成的`nn.Linear`和`nn.Embedding`，它们里面已经包好了`nn.Parameter`。

我们的模型由几个`nn.Module`嵌套而成：`LLM`里面有一个`Backbone`和一个`lm_head`，`Backbone`里有词嵌入、8个`DecoderLayer`和最后的RMSNorm，每个`DecoderLayer`里又有注意力、前馈网络和两个RMSNorm。这些模块的属性名，我们故意起得和`transformers`里的一模一样。`nn.Module`会按嵌套关系，自动给每个参数起一个带路径的名字，于是我们的参数名就和HuggingFace的完全相同：

```
model.embed_tokens.weight
model.layers.0.input_layernorm.weight
model.layers.0.self_attn.q_proj.weight
...
model.layers.7.mlp.down_proj.weight
model.norm.weight
lm_head.weight
```

这样做有两个好处。一是SmolLM2的权重文件，可以原样装进我们的模型，不用改任何名字，后面对答案就靠它。二是我们自己的模型存盘时，参数名本来就是HuggingFace的格式，第一本的引擎和`transformers`都能直接读。

还有RoPE的cos表和sin表，它们也要跟着模型走：模型搬到GPU上，这两张表也得搬过去。但它们不是参数，不需要学习，也不用存盘。PyTorch管这种东西叫**buffer**，用`register_buffer`登记，再加上`persistent=False`，它就会跟着模型搬家，但不会出现在存盘的参数里。



### 训练模式和推理模式

每个`nn.Module`都有两种模式：`model.train()`是训练模式，`model.eval()`是推理模式。有些层在两种模式下的行为不一样，最典型的是**dropout**：训练时，随机把一部分数置成零，逼着模型不要依赖某几个特定的数，起到防止过拟合的作用；推理时，dropout什么都不做。

早期的GPT-2，在注意力和残差上都加了dropout。现在的大模型一般不加，SmolLM2和Qwen3的`config.json`里，`attention_dropout`都是0。原因是预训练的数据量远远大于模型能记住的量，每份数据往往只看一遍，模型根本来不及过拟合，dropout只会白白拖慢学习。我们也不加。这样一来，我们的模型在两种模式下的行为完全一样。不过养成好习惯，生成文字之前还是调一下`model.eval()`，训练之前调一下`model.train()`。



## 参数初始化

推理时，权重是别人训好的，从文件里读进来就行。训练时，一切从零开始，我们得自己给每个权重一个初始值。这个初始值有讲究：全是零不行，所有的神经元都一样，反向传播算出来的梯度也一样，它们就永远一样，学不出不同的东西；随便给个大数也不行，下面会看到它有多糟。



### 第一步的loss应该是多少

怎么判断初始值给得好不好？有一个非常简单的检查办法：看训练第一步的loss。模型什么都还没学的时候，最合理的表现是什么？是对词表里的8192个词元一视同仁，每个都给 $1/8192$ 的概率。这时，不管正确答案是哪个词元，交叉熵loss都是：

$$
-\ln \frac{1}{8192} = \ln 8192 \approx 9.011
$$

第一章的字符级模型，用的就是同样的道理：65个字符乱猜，loss是 $\ln 65 \approx 4.174$ ，而随机初始化的模型，第一步的loss是4.342，很接近。初始化得当的模型，第一步的loss应该就在 $\ln 8192$ 附近。如果远远高于这个数，说明模型一开始就很有把握地猜错了，训练得先花不少力气把这份错误的自信磨掉，严重时还会直接训崩。



### 本书的初始化

我们照GPT-2的做法来初始化，`LLM`的`reset_parameters`方法是这样写的：

```python
        std = self.config.initializer_range
        small = std / math.sqrt(2 * self.config.num_hidden_layers)
        for name, p in self.named_parameters():
            if name.endswith("norm.weight"):
                nn.init.ones_(p)
            elif name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=small)
            else:
                nn.init.normal_(p, mean=0.0, std=std)
```

规则有三条。RMSNorm的权重全部是1，一开始什么都不缩放。线性层和词嵌入，用均值为0、标准差为0.02的正态分布随机生成。每一层里，最后写回残差的那两个投影，也就是注意力的`o_proj`和前馈网络的`down_proj`，标准差再除以 $\sqrt{2 \times \text{层数}}$ ，我们的模型是8层，就是除以4，标准差变成0.005。

最后这一条是为了残差连接。每一层都往残差里加两次东西，一次是注意力的输出，一次是前馈网络的输出，8层一共加16次。每次加的都是随机的数，方差会越加越大，层数越多，最后一层的输出就越大。把这两个投影的初始值调小，每次加进去的东西小一些，加完16次，总的方差也就控制住了。



### 三种初始化比一比

脚本`init_loss.py`拿验证集开头的8段文本，每段256个词元，比较三种初始化。第一种是本书的做法。第二种是全部用标准差0.02，不对`o_proj`和`down_proj`特殊处理。第三种是什么都不做，用PyTorch自带的默认初始化：`nn.Linear`是一种均匀分布，`nn.Embedding`是标准正态分布，也就是标准差为1。结果如下：

```
均匀猜测的loss：ln(8192) = 9.011

初始化          第一步loss  logits标准差  残差流标准差      猜成自己
本书                 9.090         0.453         0.111         64.7%
不缩小残差           9.113         0.450         0.494          0.1%
PyTorch默认        478.759        23.216         1.064        100.0%
```

先看第一步的loss。本书的做法是9.090，比 $\ln 8192$ 只高了一点点，正是我们想要的。PyTorch默认的初始化，loss高达478.8，比乱猜还差了五十多倍。问题出在权重共享上，下一节会讲：词嵌入和输出层是同一个矩阵，它的标准差是1，乘出来的logits标准差有23，最高分和最低分差了上百，softmax之后，模型几乎把全部概率押在了一个词元上，押错了，loss就爆了。用这样的初始值去训练，前几百步都要花在把这些巨大的权重缩回去上。

再看“残差流标准差”这一列，它是最后一层输出的标准差。所谓**残差流**（Residual Stream），就是那条贯穿所有层、每层都往上加东西的主干。本书的做法是0.111；不缩小残差的投影，变成了0.494，大了四倍多。这时第一步的loss还看不出多少差别，差别要到训练的过程中才会显现出来：残差流越大，最后那个RMSNorm要缩得越狠，各层的梯度也就越不均衡。这个对照，留到第六章去做。

最后一列很有意思，它统计的是：在所有位置上，得分最高的那个词元，恰好就是当前位置上的那个词元自己的比例。本书的初始化是64.7%，也就是说，一个什么都没学过的模型，有将近三分之二的位置，都在猜“下一个词还是我”。这也是权重共享带来的，下一节解释。



## 权重共享

模型的入口是词嵌入，一个`(8192, 512)`的矩阵，把词元编号换成向量。模型的出口是输出层，在`transformers`里叫`lm_head`，一个`(512, 8192)`的线性层，把最后的向量换成词表里每个词元的分数。这两个矩阵的形状正好互为转置。**权重共享**（Weight Tying）就是让它们用同一个矩阵：出口直接拿入口的那张表来用，算分数时，就是拿最后的向量，和每个词元的词嵌入算点积，越像的得分越高。第一本第四章已经详细讲过，SmolLM2就是这么做的，这里只补充训练时要注意的地方。

在PyTorch里，实现权重共享只需要一行：

```python
            self.lm_head.weight = self.model.embed_tokens.weight
```

两个名字指向同一个`nn.Parameter`。反向传播时，入口和出口各自算出一份梯度，PyTorch会把它们加在一起，存在这个共享的参数上。

对小模型来说，权重共享省下的参数相当可观。我们的词嵌入有 $8192 \times 512 = 4{,}194{,}304$ 个参数，不共享的话，输出层还要再来这么多，模型就从26.7M变成30.9M，多出来的这16%，全花在了输出层上。SmolLM2-135M也共享，词嵌入占它全部参数的21%。第一本提过，Qwen2.5-7B、Mistral-7B这样的大模型就不共享了，词嵌入在几十亿参数里只占很小一部分，省下来的那点，不值得为此让入口和出口绑在一起。

现在可以解释上一节那个64.7%了。初始化之后，每层往残差里加的东西都很小，最后那个向量里，当前词元自己的词嵌入还占着不小的分量。输出层拿这个向量和每个词元的词嵌入算点积，和它自己的词嵌入算出来的分数，就常常是最高的。不缩小残差的投影，各层加进来的随机噪声更大，把这个分量淹没了，比例就掉到了0.1%。PyTorch默认的初始化正好相反，词嵌入的标准差是1，各层加进来的东西相比之下微不足道，100%的位置都在猜自己。这件事对训练没有什么影响，训练开始几十步之后，它就会消失。但它会在随机初始化的模型生成的文字里露个脸，后面“代码效果”一节会看到。



## 定尺寸

模型的配置写在`config.py`的一个dataclass里。每个字段的名字，都照抄SmolLM2的`config.json`，这样两边能互相转换：SmolLM2的配置能读进来，我们的配置存出去，第一本的引擎和`transformers`都认识。默认值就是本书的模型：

```python
@dataclass
class Config:
    vocab_size: int = 8192             # 第三章训练的分词器
    hidden_size: int = 512             # 每个词元用多少个数表示，即d
    num_hidden_layers: int = 8
    num_attention_heads: int = 8       # 每个头64维
    num_key_value_heads: int = 2       # GQA：每4个q头共用一组k、v
    intermediate_size: int = 1408      # SwiGLU中间那一层的宽度
    max_position_embeddings: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True   # 输出层和词嵌入共用一个矩阵
    initializer_range: float = 0.02    # 初始化的标准差
    bos_token_id: int = 0
    eos_token_id: int = 0              # <|endoftext|>
```



### 尺寸由什么决定

模型多大，不是凭感觉定的，而是由三样东西倒推出来的：

* **硬件和时间**：我们只有Colab上的一张T4，免费的会话最长只有几个小时，断了还得接着训。第一章估算过，T4实际能用上的算力，大约每秒5到10万亿次浮点运算。
* **计算量**：第一章说过，训练的计算量大约是 $6ND$ ， $N$ 是参数量， $D$ 是训练的词元数。
* **数据量**：按Chinchilla法则， $D$ 大约取 $N$ 的20倍最划算。

把后两样合起来，计算量就是 $6N \times 20N = 120N^2$ ，参数量翻一倍，数据也要跟着翻一倍，计算量就变成了四倍。26.7M的模型，算下来是 $8.6 \times 10^{16}$ 次运算，在T4上要两到五个小时，刚好是一次会话能完成的量。参数量再大一倍，就要八到二十个小时，一次会话装不下了。这个估计还没有在T4上实测过，第六章开训之前会先测一下速度，如果比估计的慢很多，就把模型往下缩。



### 本书的配置

参数总量定在二三千万，剩下的问题就是怎么分配。我们的几个选择，大多是向SmolLM2看齐：

* **每个头64维**：和SmolLM2、Llama都一样。隐藏层维度512，就是8个头。
* **2个键值头**：8个查询头分成两组，每组4个共用一组键和值。SmolLM2是9个查询头配3个键值头，每组3个。GQA本来是为了让推理时的KV缓存小一些，对训练没有什么影响，我们保留它，是为了和第一本的引擎一致。
* **前馈网络的中间层1408**：SwiGLU有三个矩阵，比普通的前馈网络多一个。为了让参数量和普通的4倍宽度持平，中间层通常取隐藏层维度的 $8/3$ 倍，也就是1365，再向上取到64的倍数，就是1408。
* **8层**：每层282万个参数，8层是2255万。宽度和深度怎么取舍，没有定论；同样的参数量，SmolLM2选择了又窄又深，30层，每层只有576维。我们的模型太小，层数太多，每层就太窄了，8层是一个折中。
* **最长1024个位置**：训练时每段序列最长1024个词元。用我们的分词器数，FineWeb-Edu的文档平均一千二百多个词元，1024个位置能装下大部分文档的一大段。
* **RoPE的 $\theta$ 取10000**：这是原始论文的取值。SmolLM2取100000，那是为了把上下文拉长到8192个位置，转得慢一些，远处的位置才不会转回到原点附近。我们只有1024个位置，用不着。

运行`model.py`，可以看到每一部分各有多少参数：

```
词嵌入（和输出层共用）         4,194,304
每层：注意力 q、k、v、o          655,360
每层：前馈 gate、up、down      2,162,688
每层：两个RMSNorm                  1,024
每层合计                       2,819,072
8层合计                       22,552,576
最后的RMSNorm                        512
全部                          26,747,392
```

一共26,747,392个参数，和第一章逐项算出来的完全一样。每层里，前馈网络占了四分之三还多，注意力只占不到四分之一，这是GQA的功劳：键和值各只有128维，是查询的四分之一。



## 存成第一本能读的格式

训练好的模型，要交给第一本的引擎去跑。两本书之间，只靠一种东西交接：一个和SmolLM2长得一样的模型目录。`export.py`里的`save`函数，往这个目录里写四个文件：

* `config.json`：模型的配置，由`Config.save`写出，除了上面那些字段，还要补上`model_type`、`hidden_act`这几个固定的字段，`transformers`靠它们认出这是一个Llama模型。
* `generation_config.json`：起始符和结束符的编号。第一本的引擎会直接打开这个文件，缺了它会报错。
* `model.safetensors`：权重。
* `tokenizer.json`：第三章训练的分词器，原样复制过来。

存权重的那一段是这样的：

```python
    state = model.state_dict()
    if cfg.tie_word_embeddings:
        # lm_head.weight和model.embed_tokens.weight是同一块内存。safetensors不肯
        # 存两个共享内存的张量，SmolLM2的文件里也只有后者，所以去掉前者。
        del state["lm_head.weight"]
    # 不管训练时用什么精度，存盘都是fp32；contiguous是safetensors的要求。
    state = {k: v.detach().float().contiguous().cpu() for k, v in state.items()}
    save_file(state, out_dir / "model.safetensors", metadata={"format": "pt"})
```

函数`state_dict()`返回模型全部参数的一个字典，键就是前面那些HuggingFace格式的参数名。权重共享时，`lm_head.weight`和`model.embed_tokens.weight`都在这个字典里，但它们是同一块内存。safetensors格式不允许存两个共享内存的张量，SmolLM2的文件里也只存了词嵌入，所以我们把`lm_head.weight`去掉。读的时候反过来：按`config.json`建好模型，再把文件里的权重装进去，缺的只有`lm_head.weight`，而它本来就是词嵌入，什么都不用做。

权重一律存成fp32。第六章在T4上训练时，会用fp16的混合精度来算，但那只是计算时的精度，模型本身的权重一直是fp32的。SmolLM2存的是bf16，第一本的引擎读的时候会转成需要的精度，存fp32对它来说没有区别，文件大一倍而已：我们的模型存下来是107MB。



## 先对答案

模型写完了，还没有训练，怎么知道它写对了？最直接的办法，是找一个已经训好的、同样架构的模型，把它的权重装进来，看算出来的结果对不对。SmolLM2-135M就是现成的答案：它的架构和我们的完全一样，只是尺寸不同；我们的参数名和HuggingFace的一字不差，它的权重可以原样装进来。

只要把配置换成SmolLM2的，我们的`LLM`类就能建出一个SmolLM2：576维、30层、9个查询头、3个键值头。脚本`check_smollm2.py`把SmolLM2的权重装进这个模型，再用`transformers`加载同一份权重，两边喂同一批数据，比三样东西：

1. **logits**：前向计算对不对。
2. **loss**：交叉熵算得对不对，第 $t$ 个位置预测第 $t+1$ 个词元的这一位错开得对不对。
3. **每个参数的梯度**：反向传播对不对。

第三样是推理时不用比的。我们的梯度是PyTorch自动算的，按理说不会错，但模型搭错了，梯度也就跟着错了。最典型的是权重共享：如果`lm_head`和词嵌入没有真正共用一个参数，而是在建模型时复制了一份，前向计算的结果一模一样，可训练时它们各自更新，越走越远。只比logits，是发现不了这种问题的。

比较的时候，两边的接口不太一样。我们的`forward`要求把输入和答案分开给：输入是前 $T-1$ 个词元，答案是后 $T-1$ 个，正好错开一位。`transformers`则是把整段序列当成`labels`传进去，由它在内部错开。两种写法算的是同一个东西。结果如下：

```
SmolLM2-135M：d=576，30层，9头（KV 3头），FFN 1536，词表49,152
输入         (2, 33)，两条序列各33个词元
logits       相对误差 6.9e-07
loss         我们 2.501235，transformers 2.501235
梯度         272个参数，最大相对误差 1.0e-05
             出现在model.layers.25.input_layernorm.weight
```

两边的logits，相对误差不到百万分之一，loss到小数点后六位都一样。梯度一共比了272个参数，30层每层9个，再加上词嵌入和最后的RMSNorm，最大的相对误差是十万分之一，出现在一个RMSNorm的权重上。这么小的误差，来自浮点数加法的顺序不同：RMSNorm的权重的梯度，要把所有位置上的贡献加起来，加的顺序不一样，最后几位就会有出入。这些都在fp32下、在CPU上算，作者的Mac上用了半分钟。

顺便说一句，两段英文喂给SmolLM2，loss是2.5，相当于每个位置平均只在十几个候选里犹豫，这是一个真正学会了英语的模型。我们的模型，起点是9.0。



## 代码实现

本章的代码分三组。`config.py`和`model.py`是模型本身，后面每一章都要用；`export.py`负责把模型存成目录、再读回来，后面导出模型时也要用；`init_loss.py`、`check_smollm2.py`、`check_export.py`三个脚本，都是为了确认模型写对了，只在这一章运行。



### 模型的前向计算

模型的`forward`方法，把前面讲的这些部件串了起来。它接收一个`(B, T)`的词元编号张量，先查词嵌入，再依次过8层，最后过RMSNorm和输出层：

```python
    def forward(self, ids, targets=None):
        """ids (B, T) → logits (B, T, 词表大小)；给了targets就顺便算loss。

        targets和ids一样形状，第t个位置放的是第t+1个词元，
        也就是每个位置“应该预测出来的那一个”。
        """
        T = ids.shape[1]
        assert T <= self.config.max_position_embeddings, "序列比RoPE表还长"
        x = self.model.embed_tokens(ids)
        for layer in self.model.layers:
            x = layer(x, self.cos, self.sin)
        logits = self.lm_head(self.model.norm(x))
        if targets is None:
            return logits, None
        # 交叉熵：B×T个位置各算一次-log p(正确答案)，取平均。在fp32里算。
        loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        return logits, loss
```

给了`targets`，就顺便把loss算出来。交叉熵是PyTorch现成的`F.cross_entropy`，它要求logits是二维的，每行一个位置，所以要先把 $B$ 和 $T$ 两维合成一维。下一章写训练循环时，会仔细讲交叉熵到底在算什么。



### RMSNorm

我们用的PyTorch 2.2还没有现成的`nn.RMSNorm`，只好自己写。好在它只有三行，和第一本一样，先升到fp32再算：

```python
    def forward(self, x):
        # 先升到fp32：fp16下，512个数的平方加起来，精度和范围都不够。
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(x.dtype)
```

第一本用fp32推理，这一升一降什么都不做。到了第六章在T4上用fp16训练，它就派上用场了：fp16能表示的最大的数只有65504，512个数的平方加起来，很容易就超了。



## 代码效果

各个脚本的主要结果，前面已经报告过了，这里补上导出和读回的结果。`export.py`建一个随机初始化的模型，存到`code/models/toyllm2-init/`，再读回来，从“Once upon a time”开始往下生成20个词元，贪婪生成和随机采样各一次：

```
config.json                      572 字节
generation_config.json            45 字节
model.safetensors        106,997,648 字节
tokenizer.json               551,609 字节

贪婪生成20个词元（折行是为了排版加的）：
  'Once upon a time time necessary necessary necessary necessary necessary
  necessary necessary necessary necessary necessary necessary necessary
  necessary necessary necessary necessary necessary necessary necessary'

随机采样生成20个词元（折行是为了排版加的）：
  'Once upon a timeones acquWhy jointstitution lawsPl Is himself sections
  correctagn aircraftburAC conqu commercialigned always desire'
```

生成的文字是一个Python字符串，太长了，脚本按空格把它折成了几行，文字一个字没改。随机采样生成的，是一串毫无意义的词元，和第一章字符级模型训练之前生成的乱码是一个性质，只不过这里的每个词元都是一个完整的单词片段。贪婪生成的那一段更有意思：第一个词元把“time”又重复了一遍，接下来就一直是“necessary”。这正是前面讲的“猜成自己”：模型看到“time”，得分最高的就是“time”自己；看到“necessary”，得分最高的还是“necessary”，于是就再也出不来了。

脚本`check_export.py`用`transformers`加载这个目录，和我们自己读回来的模型比较同一段输入的logits，以及贪婪生成的20个词元：

```
logits       相对误差 0.0e+00
贪婪生成     一模一样：True
```

两边的logits完全相同，贪婪生成的20个词元也一模一样。作者还把这个目录拷到第一本的`code/models/`下面，用第一本第十四章的引擎跑了一遍：

```bash
uv run python book/ch14/engine.py "Once upon a time" --model toyllm2-init
```

引擎顺利读进了这个模型，生成的也是“time”后面跟着一串“necessary”，和上面一字不差。从文件格式上说，两本书已经接通了。剩下的，就是让这个模型学会说话。



## 本章小结

这一章搭好了我们要训练的模型。它的数学和第一本完全一样，写法改成了训练用的：张量多了batch维；因为整段序列一次算完，不需要KV缓存；权重包成`nn.Parameter`挂在`nn.Module`上，参数名和HuggingFace的一字不差；不用dropout。

初始化看第一步的loss：应该在 $\ln 8192 \approx 9.01$ 附近。我们用标准差0.02的正态分布，写回残差的两个投影再按层数缩小，第一步的loss是9.09；什么都不做的话，因为词嵌入和输出层共用一个标准差为1的矩阵，loss高达479。

模型定在26.7M参数：词表8192，512维，8层，8个查询头配2个键值头，前馈网络中间层1408，词嵌入和输出层共享。这个尺寸是从T4上一次会话能训完倒推出来的。

训练之前，我们把SmolLM2-135M的权重装进了自己的模型，loss到小数点后六位和`transformers`相同，272个参数的梯度最大相对误差是十万分之一。模型存成和SmolLM2一样的目录之后，`transformers`和第一本的引擎都能读，生成的结果一模一样。模型有了，数据也有了，下一章，我们写一个训练循环，让loss真正降下来。
