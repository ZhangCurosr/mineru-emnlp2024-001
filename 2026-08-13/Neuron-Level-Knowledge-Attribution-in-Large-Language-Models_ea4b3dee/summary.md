---
title: "Neuron-Level-Knowledge-Attribution-in-Large-Language-Models"
source: https://aclanthology.org/2024.emnlp-main.191.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:12:50"
field: "大语言模型机制可解释性"
keywords: ["neuron-level attribution", "knowledge localization", "mechanistic interpretability", "value neurons", "query neurons", "log probability increase", "LLM interpretability"]
innovations: ["提出log probability increase作为静态神经元级归因评分，同时考虑神经元特征和输入上下文", "区分并定位value neurons和query neurons，揭示知识存储的信息流模式"]
benchmarks: ["TriviaQA", "GPT2-large", "Llama-7B"]
---

# 论文速读：Neuron-Level-Knowledge-Attribution-in-Large-Language-Models

## 一句话总结
本文提出了一种静态的神经元级知识归因方法，基于对数概率增量（log probability increase）识别直接贡献最终预测的"value neurons"，并通过内积计算识别激活这些value neurons的"query neurons"。实验表明该方法在三种评估指标上均优于七种已有静态基线方法。

## 研究问题与动机
- **计算复杂度限制**：现有归因方法（因果追踪、积分梯度等）需要多次前向/反向传播，难以扩展到LLM中数百万级别的神经元，而这些神经元已被证明是知识存储的基本单元。
- **缺乏系统性对比**：已有的神经元分析方法（如Geva et al., 2022）缺少与其他方法的公平对比，导致"哪种方法最有效"仍不明确。
- **模块覆盖不全面**：现有方法通常只关注attention模块或FFN模块之一，缺乏对两者重要性的定量比较。
- **无法识别上游依赖**：多数静态方法只能找到直接贡献预测的"value neurons"，而无法定位激活这些神经元的关键"query neurons"。

## 核心贡献（创新点）
- **提出log probability increase重要性评分**：通过$Imp(v) = \log p(w|v+h^{l-1}) - \log p(w|h^{l-1})$量化神经元对目标token概率的影响，同时考虑神经元本身特征和输入上下文，且满足可加性便于模块组合分析。
- **区分"query neurons"与"value neurons"**：针对已识别的value neurons，提出基于子键（subkey）与残差流中各神经元内积的方法定位激活它们的query neurons。
- **统一评估attention和FFN层**：在同一框架下定量比较两种模块对知识存储的贡献。
- **揭示知识存储的层级流向规律**：发现浅层/中层FFN神经元提取特征→激活深层attention value neurons→进一步激活深层FFN value neurons的信息流模式。
- **精确定位少量关键神经元即可显著影响预测**：仅干预300个top神经元（200个attention+100个FFN）即可使GPT2/Llama-7B的正确token概率下降约99%。

## 方法详解
- **Transformer基础公式**：$h_i^l = h_i^{l-1} + A_i^l + F_i^l$，其中attention输出为多头加权求和，FFN输出可分解为FFN神经元的加权和。
- **FFN神经元定义**：第k个FFN神经元为$fc2_k^l$（$W_{fc2}$的第k列），其系数$m_{i,k}^l = \sigma(fc1_k^l \cdot (h_i^{l-1} + A_i^l))$由子键$fc1_k^l$与残差输出的内积决定。
- **Attention神经元定义**：将position value-output vector $W_{j,l}^o(W_{j,l}^v h_p^{l-1})$视为基本单元，$W_{j,l}^o$的第k列为第k个attention子value，$W_{j,l}^v$的第k行为对应子key。
- **Log probability increase公式**：
  - Attention层：$Imp(v^l) = \log p(w|v^l + h^{l-1}) - \log p(w|h^{l-1})$
  - FFN层：$Imp(v^l) = \log p(w|v^l + A^l + h^{l-1}) - \log p(w|A^l + h^{l-1})$
- **Query neuron识别**：计算value neuron子key与残差流中各神经元/subvector的内积，内积越大表示该神经元越有助于激活value neuron。

## 实验与结果
- **数据集**：从TrivialQA提取6种知识类型（language, capital, country, color, number, month）的问答对，共1,350句（GPT2-large）和3,141句（Llama-7B）。
- **模型**：GPT2-large（36层，20头，每头64神经元，FFN 5,120神经元）和Llama-7B（32层，32头，每头128神经元，FFN 11,008神经元）。
- **评估指标**：干预top-K神经元后，正确token的MRR、概率（prob）、对数概率（logp）的下降幅度。
- **主要结果**（Table 2）：
  - **GPT2-large**：原模型prob=7.1%，method a干预后降至3.4%（降幅最大）；MRR从0.361降至0.201（最优）。
  - **Llama-7B**：原模型prob=21.5%，method a干预后降至9.2%；MRR从0.551降至0.312（最优）。
  - 最强提升：仅干预10个FFN神经元，Llama-7B正确token概率下降超过50%。
- **层分布发现**：所有top10重要层均在深层（Layer 17-33）；probability increase倾向于最深层，log probability increase能同时覆盖中深层和深层。
- **神经元级干预效果**（Appendix C）：干预top200 attention + top100 FFN神经元，GPT2的MRR/prob下降96.3%/99.2%，Llama-7B下降96.9%/99.6%；随机干预同样数量仅下降0.22%/0.14%。
- **Query神经元验证**：干预top1000 shallow query FFN神经元，GPT2和Llama-7B的MRR/prob分别下降92%/95%和87%/95%（Table 7）。

## 相关工作脉络
- **因果追踪方法**（Meng et al., 2022; Vig et al., 2020）：通过扰动中间激活值估计贡献，计算开销大，难以扩展到神经元级。本文方法与因果方法形成对比，强调静态方法的高效性。
- **积分梯度**（Sundararajan et al., 2017）：需要大量前向/反向传播，计算成本高。本文方法仅需单次前向传播。
- **FFN作为key-value memory**（Geva et al., 2020, 2022）：首次将FFN分解为神经元级分析，但使用$|m|\times|v|$或$|m|\times 1/rank(w)$作为评分，未考虑输入上下文x的影响。本文方法改进为log probability increase。
- **注意力权重可解释性质疑**（Serrano & Smith, 2019; Jain & Wallace, 2019）：指出attention weights本身不能直接解释模型行为。本文避免依赖attention weights作为归因依据。
- **Mechanistic Interpretability**（Olah, 2022; Nanda et al., 2023a）：逆向工程输入到预测的电路。本文方法为这一方向提供了可扩展的神经元级归因工具。
- **Causal mediation analysis**（Stolfo et al., 2023）：用于算术推理的机制分析，同样面临计算复杂度问题。本文提出静态替代方案。

## 局限性与未来方向
- 仅研究六种特定知识类型，其他类型（如日期、人名、专业术语）有待探索。
- 实验仅在GPT2-large和Llama-7B上进行，跨模型比较（如更大规模的Llama-13B/70B）不足。
- 仅使用静态方法，未与因果中介分析、梯度基方法等进行系统对比。
- Query neurons的可解释性较弱（投影到词汇空间后语义模糊），值得进一步研究。
- 最深层neuron的重要性可能因log概率曲线的饱和效应而被低估，需改进评分函数。
- 潜在风险：方法可能被用于放大毒性/偏见神经元，但也可用于减少幻觉和偏差。

## 研究启发与可借鉴点
- **Log probability increase作为归因评分**：同时考虑神经元自身特性和输入上下文，且满足可加性，可推广到其他归因任务。
- **Query/Value分离框架**：将神经元分为"直接贡献"和"上游激活"两类，为理解信息流提供新视角，可应用于电路发现。
- **小样本高效干预**：仅300个神经元即可影响99%的预测概率，为后续的神经元级知识编辑（knowledge editing）提供精确靶点。
- **知识类型的层分布规律**：语义相似的知识（如语言/首都/国家）集中在相同层/头，语义不同的知识分散在不同层，可用于指导知识组织的结构化分析。
- **评估指标设计**：通过"干预后概率下降幅度"评估归因质量，比单纯相关性指标更直接反映因果效应，可作为归因方法的通用评测范式。

## 关键术语表
- **Value neurons**：直接包含目标知识信息、对最终预测概率有贡献的神经元（FFN的fc2向量或attention的output向量）。
- **Query neurons**：不直接包含知识信息，但通过内积激活value neurons的上游神经元（FFN的fc1行向量或attention的value矩阵列向量）。
- **Log probability increase**：本文提出的重要性评分，衡量加入某神经元后目标token对数概率的变化量。
- **bs-value (before-softmax value)**：向量与未嵌入矩阵$E_u$的內积$e_w \cdot x$，softmax前的logit值，与token概率一一对应。
- **Coefficient score (m)**：FFN神经元的前置系数，由子键与残差流内积经非线性函数σ计算得到。
- **FFN subkey / subvalue**：$W_{fc1}$的行向量（subkey）和$W_{fc2}$的列向量（subvalue），分别用于计算系数和构成输出。
- **Intervention**：将识别出的关键神经元的参数置零，观察模型输出的变化以验证归因结果的有效性。
- **Mechanistic interpretability**：通过逆向工程揭示模型内部计算机制的研究方向，本文方法为其提供神经元级归因工具。

## 可复现要素
- **数据集**：TriviaQA（公开），论文从其中提取6种知识类型的问答对。
- **代码**：已开源，https://github.com/zepingyu0512/neuron-attribution
- **模型**：GPT2-large和Llama-7B（均为公开模型）。
- **关键超参**：top-K取值（FFN干预用top10/top100/top200，attention干预用top200，query干预用top1000）。
- **论文未提及**：训练细节（本方法为静态分析无需训练）、具体硬件配置、训练随机种子。
