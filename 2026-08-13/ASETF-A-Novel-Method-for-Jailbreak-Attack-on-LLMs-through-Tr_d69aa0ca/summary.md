---
title: "ASETF-A-Novel-Method-for-Jailbreak-Attack-on-LLMs-through-Tr"
source: https://aclanthology.org/2024.emnlp-main.157.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:12:04"
---

# 论文速读：ASETF-A-Novel-Method-for-Jailbreak-Attack-on-LLMs-through-Tr

## 一句话总结
提出ASETF（Adversarial Suffix Embedding Translation Framework），通过在目标LLM的连续嵌入空间中联合优化交叉熵与MMD损失获取对抗后缀向量，再经微调的嵌入翻译模型将其还原为流畅可读的自然语言文本，在大幅降低计算开销的同时实现了对多款开源及黑盒LLM的高效越狱攻击。

## 研究问题与动机
1. **防御方法滞后于攻击演化**：现有安全对齐与规则防御多依赖人工设计的劫持模板，难以跟上新型攻击变体的涌现速度。
2. **离散空间优化计算开销极高**：基于梯度的离散后缀搜索（如GCG）需在完整词表上迭代数百次，每次调用数百次LLM，总调用量超10万次。
3. **对抗文本可读性差易被过滤**：传统梯度优化生成的后缀无实际语义，困惑度（PPL）极高，极易被perplexity过滤器或人工审查拦截。
4. **现有自动化工具未充分利用模型内部信息**：如MasterKey、GPTFuzzer等方法多依赖外部模板或变异策略，缺乏对目标模型嵌入空间梯度的直接利用。

## 核心贡献（创新点）
1. **连续嵌入优化与文本翻译解耦**：将离散后缀搜索转移至连续嵌入空间进行梯度优化，再通过独立翻译模型映射回自然语言。（与GCG等离散迭代方法的本质区别：跳过海量词表遍历，单次优化仅需几十次梯度更新，计算效率呈数量级提升。）
2. **引入MMD分布对齐损失约束优化向量**：在交叉熵损失基础上加入Maximum Mean Discrepancy损失，强制对抗向量$\phi$贴近目标模型的词汇嵌入分布。（与纯梯度攻击的本质区别：防止优化向量漂移到分布外区域，保障后续可被翻译模型稳定还原。）
3. **上下文感知的自监督嵌入重建框架**：利用Wikipedia语料构建“上下文句+后缀句”平行对，微调GPT-J学习跨模型嵌入空间的线性映射与文本重构。（与AutoDan等直接搜索/遗传算法的本质区别：不依赖离散搜索树，而是通过学习连续空间的语义映射一次性生成高质量后缀。）
4. **多目标联合训练的通用/可迁移后缀生成**：同步融合多个目标LLM的嵌入层进行联合微调，输出单一后缀即可攻击多款模型甚至黑盒API。（与单模型定向攻击的本质区别：打破模型边界限制，实现从白盒梯度优化到黑盒API攻击的端到端迁移。）

## 方法详解
1. **连续对抗后缀嵌入优化（Section 3.1）**：
   - 给定有害指令$x_{\text{harm}}$与目标响应$R$，随机采样$n$个词嵌入作为初始向量$\phi \in \mathbb{R}^{L \times d}$，拼接至指令嵌入$E_p$后输入目标LLM。
   - 联合优化目标：$\nabla_\phi L = \nabla_\phi L_{ce} + \nabla_\phi L_{mmd}$。其中$L_{ce}$为标准自回归交叉熵，驱动模型输出匹配$R$；$L_{mmd}$采用高斯核（$\sigma=1$）度量$\phi$与目标模型词嵌入分布$X$（采样$m=100$个词）的分布距离，约束优化向量不偏离合法词嵌入流形。
   - 多目标扩展时，损失函数叠加所有$K$个目标模型的概率分布：$L_{ce} = -\sum_{j=1}^{K}\sum_{i=1}^{M}\sum_{t=1}^{n} \log P_j(r_{i_t}|r_{i_{1:t-1}}, E_i; \theta)$。
2. **单目标嵌入翻译框架（Section 3.2.1）**：
   - 训练数据从Wikipedia英文语料抽取连续句对$\{c_1, c_2\}$，$c_1$作上下文（替换指令），$c_2$作后缀（替换对抗向量）。
   - 架构：翻译模型（GPT-J）嵌入层编码$c_1$，目标模型嵌入层编码$c_2$，两者经全连接对齐层$W_{ad}$合并为$E_C \oplus E_S W_{ad}$，输入翻译LLM进行原文重建。
   - 增强策略：训练时在$E_S$上添加随机高斯噪声$\epsilon$，提升翻译模型对推理阶段非精确嵌入的鲁棒性。
   - 目标函数：$J(\theta) = \frac{1}{n|D|}\sum_{(c_1,c_2)\in D}\sum_{i=1}^{n} L(s_i, o_i;\theta)$，$L$为交叉熵。
3. **多目标/通用后缀翻译（Section 3.2.2）**：
   - 同步接入$m$个目标LLM的嵌入层，联合最小化各目标的重构损失，使翻译模型学会跨模型共享的嵌入映射策略，最终生成的文本后缀在不同目标模型中均能保持攻击有效性。

## 实验与结果
- **数据集与模型**：攻击指令来自Advbench（500+条有害指令）；训练语料为Wikipedia英文；目标模型含Llama2-7b-chat、Vicuna-7b-v1.5、Mistral-7b、Alpaca-7b (Safe-RLHF)；迁移测试覆盖Vicuna-13b、Llama2-13b、ChatGLM3-6b及黑盒ChatGPT/Gemini；翻译基础模型为GPT-J-6b。
- **评估指标**：$ASR_{prefix}$（负向词列表匹配）、$ASR_{gpt}$（GPT-3.5-turbo分类器判定）、Perplexity（PPL）、Self-BLEU、耗时。
- **主要结果**：
  - **Ad-hoc攻击**：ASETF在Llama2上$ASR_{prefix}=0.91$、$ASR_{gpt}=0.74$，PPL降至32.59（基线GCG高达1513.09），耗时仅104.53秒（GCG约233秒），Self-BLEU达0.399（多样性最佳）。在Vicuna、Mistral、Alpaca上均取得最优ASR与最低PPL。
  - **抗改写防御**：经ChatGPT paraphrase后，ASETF在Llama2上的$ASR_{gpt}$仍保持0.37，显著高于GCG的0.21与AutoDan系列的0.19~0.21。
  - **通用后缀**：同时优化25条指令生成的通用后缀，ASETF在Llama2上$ASR_{gpt}=0.67$，耗时427.52秒，较GCG（965.75秒）提速近一倍。
  -
