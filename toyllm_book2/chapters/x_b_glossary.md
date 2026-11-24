# 附录B：术语表

这个附录收的是正文里定义过的术语，按它们在书里第一次出现的先后排列，和章节的顺序一致。最后一列是讲它的章节：先在前面提了一句、到后面才详细讲的，两章都写上。有简称的，简称写在英文后面。第一本已经讲过、这本书在第四章又速览了一遍的架构术语，也一并收了进来；另外有一些词，正文里没有专门给出定义，但用得很多，比如KL散度、warmup、fp16，也收在这里。

| 中文 | 英文 | 说明 | 章节 |
| ---- | ---- | ---- | ---- |
| 词元 | Token | 模型处理文字的单位 | 第一章、第三章 |
| 损失 | Loss | 衡量模型预测得有多差的一个数，正文直接叫它loss；训练就是让它降下来 | 第一章、第五章 |
| 反向传播 | Backpropagation | 从loss往回算，得到每个参数该往哪个方向调 | 第一章 |
| 优化器 | Optimizer | 按算出来的方向，把每个参数调一小步 | 第一章、第五章 |
| 批次 | Batch | 一步里一起训练的若干条序列，正文直接叫它batch | 第一章、第五章 |
| 基座模型 | Base Model | 只做过预训练的模型，只会续写 | 第一章、第六章 |
| 监督微调 | Supervised Fine-Tuning，简称SFT | 用对话数据接着训，教模型一问一答 | 第一章、第七章 |
| 偏好对齐 | Preference Alignment | 给模型看成对的回答，让它往好的那边靠 | 第一章、第八章 |
| 基于人类反馈的强化学习 | Reinforcement Learning from Human Feedback，简称RLHF | 先训奖励模型，再用强化学习让模型追高分 | 第一章、第八章 |
| 直接偏好优化 | Direct Preference Optimization，简称DPO | 绕过奖励模型，直接在成对的回答上训练 | 第一章、第八章 |
| 强化学习 | Reinforcement Learning | 不给标准答案，只在模型答完之后告诉它好不好 | 第一章、第八章 |
| 蒸馏 | Distillation | 让大模型当老师，把它会的东西教给小模型 | 第一章 |
| LoRA | Low-Rank Adaptation | 冻住原来的权重，只训练旁边加上的一小部分参数 | 第一章 |
| Chinchilla法则 | Chinchilla Scaling Law | 算力固定时，训练的词元数取参数量的20倍左右最划算 | 第一章、第六章 |
| 合成数据 | Synthetic Data | 让能力强的大模型按指定的主题和风格批量写出来的数据 | 第二章 |
| 数据配比 | Data Mixture | 网页、代码、书籍等不同来源的数据怎么搭配 | 第二章 |
| 快照 | Snapshot | Common Crawl的一次抓取，每个快照有二十多亿个网页 | 第二章 |
| 正文提取 | Text Extraction | 从HTML里抽出正文，扔掉菜单、页脚这些 | 第二章 |
| 语言识别 | Language Identification | 判断一篇文档是哪种语言 | 第二章 |
| 启发式过滤 | Heuristic Filtering | 按长度、重复、符号比例这类简单规则扔掉低质量的文档 | 第二章 |
| 质量分类器 | Quality Classifier | 给一篇文档打分的小模型，FineWeb-Edu靠它挑出有教育价值的网页 | 第二章 |
| 去重 | Deduplication | 去掉重复的文档，分精确去重和近似去重 | 第二章 |
| 个人身份信息 | Personally Identifiable Information，简称PII | 邮件地址、IP地址这类能认出某个人的信息 | 第二章 |
| PII脱敏 | PII Removal | 把个人身份信息换成固定的占位内容 | 第二章 |
| 数据污染 | Data Contamination | 基准测试的题目和答案混进了训练数据 | 第二章 |
| 去污染 | Decontamination | 把训练数据和测试集比对，去掉有重叠的文档 | 第二章 |
| 行组 | Row Group | Parquet文件内部可以单独读出来的一块 | 第二章 |
| 切片 | Shingling | 把文档按词切开，每连续 $n$ 个词算一个片段 | 第二章 |
| Jaccard相似度 | Jaccard Similarity | 两个集合的交集大小除以并集大小 | 第二章 |
| 最小哈希 | MinHash | 两个集合的最小哈希值相等的概率，正好是它们的Jaccard相似度 | 第二章 |
| 签名 | Signature | 一篇文档在 $k$ 个哈希函数下的 $k$ 个最小哈希 | 第二章 |
| 局部敏感哈希 | Locality-Sensitive Hashing，简称LSH | 把签名分组建哈希表，不用两两比较就能找出候选的重复对 | 第二章 |
| 验证集 | Validation Set | 训练时不让模型看，用来检验效果、帮我们做选择 | 第二章 |
| 训练集 | Training Set | 用来训练的那部分数据 | 第二章 |
| 测试集 | Test Set | 一直锁着，所有选择都做完了才用一次 | 第二章 |
| 分词器 | Tokenizer | 把文本切成词元、换成编号 | 第三章 |
| BPE | Byte Pair Encoding | 一次次合并最常见的相邻片段，得到词表和合并规则 | 第三章 |
| 预切分 | Pre-tokenization | 先按规则把文本切成片段，合并只在片段内部进行 | 第三章 |
| 特殊词元 | Special Token | 事先指定、编码时永远不会被拆开的词元 | 第三章、第七章 |
| 每字节比特数 | Bits Per Byte | 平均每个字节的文本要用多少比特来编码，换了分词器也能比 | 第三章、第六章 |
| 词嵌入 | Embedding | 每个词元一行的表，按编号取出这个词元的向量 | 第四章 |
| RMSNorm | Root Mean Square Normalization | 向量除以自己的均方根，再逐元素乘一个可学习的权重 | 第四章 |
| 因果掩码 | Causal Mask | 对角线以上全是 $-\infty$ ，每个词元只看得见它前面的词元 | 第四章 |
| 多头注意力 | Multi-Head Attention | 切成几个头，各算各的注意力，再拼回来 | 第四章 |
| 分组查询注意力 | Grouped-Query Attention，简称GQA | 几个查询头共用一组键和值 | 第四章 |
| 旋转位置编码 | Rotary Position Embedding，简称RoPE | 按词元所在的位置，旋转查询和键 | 第四章 |
| 前馈网络 | Feed-Forward Network，简称FFN | 对每个词元单独计算，我们用的是SwiGLU | 第四章 |
| 残差连接 | Residual Connection | 子层的输出加回到输入上，而不是替换它 | 第四章 |
| 残差流 | Residual Stream | 贯穿所有层、每层都往上加东西的那条主干 | 第四章 |
| 权重共享 | Weight Tying | 输出层直接用词嵌入那张表 | 第四章、第六章 |
| 交叉熵 | Cross Entropy | 语言模型的loss，正确答案的概率取对数再取负 | 第五章 |
| 困惑度 | Perplexity | loss取指数，大约是模型平均在几个候选里犹豫 | 第五章、第六章 |
| 最大似然 | Maximum Likelihood | 调整参数，让训练语料的概率最大 | 第五章 |
| 熵 | Entropy | 衡量一个分布的不确定性，分布越平，熵越大 | 第五章 |
| KL散度 | Kullback-Leibler Divergence | 交叉熵比熵多出来的部分，衡量两个分布差得有多远 | 第五章、第八章 |
| 梯度 | Gradient | 每个参数调大一点，loss会怎么变 | 第五章 |
| 梯度范数 | Gradient Norm | 所有梯度平方和的平方根，衡量这一步的梯度有多“猛” | 第五章、第六章 |
| 学习率 | Learning Rate | 控制每一步迈多大 | 第五章 |
| 随机梯度下降 | Stochastic Gradient Descent，简称SGD | 在随机取的一小批数据上算梯度，沿反方向走一步 | 第五章 |
| AdamW | AdamW | 给每个参数单独定步长的优化器，权重衰减直接作用在参数上 | 第五章 |
| 权重衰减 | Weight Decay | 每一步把参数本身缩小一点点 | 第五章 |
| 发散 | Divergence | loss一路往上涨，最后变成无穷大或NaN | 第五章、第六章 |
| warmup | Warmup | 训练开始的一小段，学习率从0线性升到最大 | 第五章 |
| cosine衰减 | Cosine Decay | 之后学习率沿余弦曲线的半个周期慢慢降下来 | 第五章 |
| 梯度累积 | Gradient Accumulation | 一个大batch拆成几个小batch，梯度加起来再走一步 | 第五章 |
| 梯度裁剪 | Gradient Clipping | 梯度范数超过上限时，把整个梯度按比例缩小 | 第五章、第六章 |
| checkpoint | Checkpoint | 定期存下来的训练状态，断了可以从这里接着训 | 第五章、第六章 |
| 吞吐量 | Throughput | 每秒能训多少个词元 | 第六章 |
| fp16 | Half Precision | 5位指数、10位尾数的16位浮点格式，最大只能表示65504 | 第六章 |
| bf16 | BFloat16 | 8位指数、7位尾数的16位浮点格式，范围和fp32一样，T4不支持 | 第六章 |
| 混合精度 | Mixed Precision | 参数和优化器状态留在fp32，只有前向和反向的计算用16位 | 第六章 |
| 上溢 | Overflow | 数太大，变成无穷大 | 第六章 |
| 下溢 | Underflow | 数太小，变成0；fp16训练里麻烦的是它 | 第六章 |
| 损失缩放 | Loss Scaling | 反向传播前把loss放大，更新前再把梯度缩回去，小梯度就不会下溢 | 第六章 |
| 尖峰 | Loss Spike | loss突然往上跳一下，几步之后又回来 | 第六章 |
| 平台 | Plateau | loss很长一段时间不降 | 第六章 |
| 探针 | Probe | 出几道题让模型续写，看它学到了什么 | 第六章 |
| 对话模板 | Chat Template | 把一段多轮对话拼成一串词元，标出谁在说话 | 第七章 |
| 对话模型 | Chat Model | 在基座模型上用对话数据接着训出来的模型，也叫指令模型（Instruct Model） | 第七章 |
| 后训练 | Post-training | 预训练之后的各步：SFT、偏好对齐、强化学习、蒸馏 | 第七章 |
| 掩码 | Mask | 每个词元一个0或1，1表示它是助手说的，要算进loss | 第七章 |
| 灾难性遗忘 | Catastrophic Forgetting | 在新数据上训练，旧任务上的表现变差 | 第七章 |
| 离线策略 | Off-policy | 训练用的回答不是正在训练的模型自己写的 | 第八章 |
| 在线策略 | On-policy | 让模型自己写回答，再看写得好不好 | 第八章 |
| 奖励模型 | Reward Model | 给一个问题和一个回答打分 | 第八章 |
| Bradley-Terry模型 | Bradley-Terry Model | 两个回答的分数差经过sigmoid，就是好回答胜出的概率 | 第八章 |
| PPO | Proximal Policy Optimization | RLHF里更新模型最常用的强化学习算法 | 第八章 |
| 策略 | Policy | 强化学习里对模型的叫法，给出每个回答的概率 | 第八章 |
| 奖励黑客 | Reward Hacking | 模型钻奖励模型的漏洞，分数很高，人看了并不满意 | 第八章 |
| 参考模型 | Reference Model | 出发时的模型，冻住不动，用来约束模型别走太远 | 第八章 |
| 价值模型 | Value Model | 估计一个回答写到一半时，最后能得多少分 | 第八章 |
| 隐式奖励 | Implicit Reward | $\beta$ 乘以策略模型和参考模型概率之比的对数 | 第八章 |
| 差距 | Margin | 好回答的隐式奖励减去坏回答的隐式奖励 | 第八章 |
| 长度偏差 | Length Bias | 偏好数据里好回答往往更长，模型学着把回答越写越长 | 第八章 |
