# 附录A：论文列表

正文里提到的论文，按第一次出现的先后排列，最后一列是它出现在哪一章。每一篇都附上arXiv编号，在arXiv网站上搜索编号就能找到原文。年份取的是论文第一次发布在arXiv上的时间，和正式发表的年份可能差一年。

| 年份 | 论文 | 说了什么 | 章节 |
| ---- | ---- | -------- | ---- |
| 2022 | 《Training Compute-Optimal Large Language Models》（arXiv:2203.15556） | Chinchilla法则：算力固定时，训练的词元数取参数量的20倍左右最划算 | 第一章、第四章、第六章 |
| 2020 | 《Scaling Laws for Neural Language Models》（arXiv:2001.08361） | 训练的计算量 $C \approx 6ND$ | 第一章、第六章 |
| 2024 | 《The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale》（arXiv:2406.17557） | FineWeb和FineWeb-Edu，本书预训练用的语料 | 第二章 |
| 2019 | 《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》（arXiv:1910.10683） | T5，以及整理C4语料用的过滤规则 | 第二章 |
| 2021 | 《Scaling Language Models: Methods, Analysis & Insights from Training Gopher》（arXiv:2112.11446） | Gopher，以及它的过滤规则 | 第二章 |
| 2014 | 《Adam: A Method for Stochastic Optimization》（arXiv:1412.6980） | Adam优化器 | 第五章 |
| 2017 | 《Decoupled Weight Decay Regularization》（arXiv:1711.05101） | AdamW：权重衰减直接作用在参数上，不经过梯度 | 第五章 |
| 2020 | 《Language Models are Few-Shot Learners》（arXiv:2005.14165） | GPT-3；模型越大，学习率越小 | 第五章 |
| 2023 | 《LIMA: Less Is More for Alignment》（arXiv:2305.11206） | 一千条精选的问答就能做好SFT，质量比数量重要 | 第七章 |
| 2023 | 《UltraFeedback: Boosting Language Models with Scaled AI Feedback》（arXiv:2310.01377） | 本书DPO用的偏好数据 | 第八章 |
| 2022 | 《Training language models to follow instructions with human feedback》（arXiv:2203.02155） | InstructGPT，RLHF的三步：SFT、奖励模型、PPO | 第八章 |
| 2017 | 《Proximal Policy Optimization Algorithms》（arXiv:1707.06347） | PPO | 第八章 |
| 2023 | 《Direct Preference Optimization: Your Language Model is Secretly a Reward Model》（arXiv:2305.18290） | DPO：绕过奖励模型，直接在成对的回答上训练 | 第八章 |
| 2025 | 《DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning》（arXiv:2501.12948） | 用可验证的奖励做强化学习，练出思考模型；先冷启动再强化学习 | 第九章 |
| 2024 | 《DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models》（arXiv:2402.03300） | GRPO，以及它用的KL估计 | 第九章 |
| 2025 | 《Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?》（arXiv:2504.13837） | 强化学习让pass@1涨，可 $k$ 很大时，pass@k反而不如强化学习之前 | 第九章 |

## 其他资料

不是论文，或者不在arXiv上，但正文提到过的：

- Broder在1997年的论文《On the Resemblance and Containment of Documents》：最小哈希（第二章）
- Karpathy的nanoGPT：Tiny Shakespeare这个例子就出自这里（第一章）
- 同样从头训练小模型的开源项目和书：MiniMind、nanochat，以及Raschka的《Build a Large Language Model (From Scratch)》（前言）
