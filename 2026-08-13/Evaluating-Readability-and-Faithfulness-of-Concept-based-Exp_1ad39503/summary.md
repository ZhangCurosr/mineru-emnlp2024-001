---
title: "Evaluating-Readability-and-Faithfulness-of-Concept-based-Exp"
source: https://aclanthology.org/2024.emnlp-main.36.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:17:39"
field: "可解释人工智能（XAI）"
keywords: ["concept-based explanation", "faithfulness", "readability", "explainable AI", "meta-evaluation", "measurement theory", "XAI evaluation"]
innovations: ["将概念基解释统一形式化为虚拟神经元，基于扰动优化问题量化忠实度", "提出基于模式相干性的自动可读性度量（UCI/UMass/EmbDist/EmbCos），与人工评估高度相关", "首次引入测量理论进行元评估，系统检验评估指标的可靠性与有效性"]
benchmarks: ["Pile", "Pythia-70M", "GPT-2 Small"]
---

# 论文速读：Evaluating Readability and Faithfulness of Concept-based Explanations

## 一句话总结
本文针对概念基解释（Concept-based Explanations）缺乏统一评估方法的问题，提出了基于虚拟神经元形式化的统一框架，分别通过扰动量化**忠实度（Faithfulness）**、通过模式相干性量化**可读性（Readability）**，并引入测量理论进行元评估以筛选可靠有效的度量指标。

## 研究问题与动机
- **缺乏统一形式化（C1）**：现有概念基解释方法（如 TCAV、Sparse Autoencoder、Neuron）对高维隐藏空间中概念的表达方式各异，缺少统一的 formalization，导致无法公平比较不同方法的评估指标。
- **可读性评估困难（C2）**：概念具有跨样本的非局部性，且概念数量庞大时人工评估成本过高；现有 LLM-based 方法受限于输入长度，覆盖不全面。
- **评估度量本身缺乏可靠性与有效性验证（C3）**：已有自动评估指标（如 LLM-Score、UCI、UMass）的可靠性（Random error 敏感性）和有效性（是否真正测量了目标构念）尚未系统检验，测量理论框架未被引入该领域。

## 核心贡献（创新点）
1. **统一概念形式化**：将各类概念基解释方法统一为"虚拟神经元"（Virtual Neuron），以激活函数 $a(h): \mathbb{R}^m \rightarrow \mathbb{R}$ 统一定义概念，覆盖监督/无监督、事后/设计内、文本/图像等多种设定。
2. **基于扰动的忠实度量化**：将概念扰动转化为高维空间中的优化问题，推导出线性激活函数下 ε-addition 和 Ablation 的闭式解（GRAD/ABL），并结合 Loss/Div/Class 三类输出差异形成 8 种忠实度度量。
3. **基于相干性的可读性自动评估**：提出 IN/OUT 两侧的模式相干性度量（UCI、UMass、Embedding Distance、Embedding Cosine Similarity），通过提取最大激活样本中的 token 对计算语义相似度，可自动、可扩展地替代人工评估。
4. **基于测量理论的元评估框架**：首次将经典测量理论引入 XAI 评估，从可靠性（重测相关、子集一致性、评分者间信度）和有效性（同时效度、收敛效度、区分效度）两个维度系统评估各度量，过滤出低可靠性指标（LLM-Score、GRAD-Loss、IN-UCI、IN-UMass），验证剩余指标有效性。

## 方法详解
- **概念形式化**：将概念表示为虚拟神经元，激活函数 $a(h): \mathbb{R}^m \rightarrow \mathbb{R}$，正值表示激活。包含语义表达式（可由人或 LLM 提供）。
- **忠实度度量** $\gamma(a, \xi, \delta)$：通过扰动隐藏表示 $h$ 后观察输出 $g(h)$ 的变化来量化，$\xi$ 为扰动函数，$\delta$ 为输出差异函数，公式(1)对样本求平均。扰动策略：
  - **ε-addition（GRAD）**：沿概念方向微扰以最大化激活，线性情形下 $\xi_e(h,a) = \lim_{\epsilon \to 0} h + \epsilon v$（公式4）
  - **Ablation（ABL）**：最小化扰动距离使激活为零，线性情形下 $\xi_a(h,a) = h - \frac{v^T h}{v^T v}v$（公式5）
  - 输出差异：Loss（训练损失差）、Div（KL散度）、Class（预测类/真实类logit差）
- **可读性度量**：基于主题相干性思想，提取最大激活 token 对并计算语义相似度：
  - **UCI**（公式9）：$\log \frac{P(x^i, x^j) + \epsilon}{P(x^i)P(x^j)}$
  - **UMass**（公式10）：$\log \frac{P(x^i, x^j) + \epsilon}{P(x^j)}$
  - **EmbDist**（公式11）：负欧氏距离 $-\|e(x^i)-e(x^j)\|_2$
  - **EmbCos**（公式12）：余弦相似度 $\frac{e(x^i)\cdot e(x^j)}{\|e(x^i)\|\|e(x^j)\|}$
  - 前缀 IN/OUT 表示输入侧/输出侧，如 IN-EmbCos、OUT-EmbDist
- **元评估**：
  - 可靠性：重测相关（Pearson 相关，阈值 ≥0.9）、子集一致性（Cronbach's α）、评分者间信度（Kendall's τ）
  - 有效性：同时效度（与人工评估的 Kendall/Pearson/Spearman 相关）、收敛效度（同构念不同度量间相关）、区分效度（不同构念度量间相关），采用 MTMM 矩阵方法

## 实验与结果
- **数据集**：Pile（825 GiB，涵盖22个子数据集的多语言文本与代码）
- **模型**：Pythia-70M（主实验）、GPT-2 Small（一致性验证）
- **基线方法**：Neuron（Bills et al., 2023）、Sparse Autoencoder（Cunningham et al., 2023）、TCAV（Kim et al., 2018）
- **实验设置**：每个度量在10个 batch（每 batch 256句×128 token，共327,680 token）上测试；从每个无监督基线随机采样100个概念
- **可靠性筛选结果**：
  - 重测可靠性：除 **LLM-Score** 外所有提出度量均为确定性方法，通过阈值0.9
  - 子集一致性：**GRAD-Loss**（忠实度）、**IN-UCI** 和 **IN-UMass**（可读性）低于0.9阈值，被过滤
  - 人工评估评分者间信度：三位专家在输入侧平均 Kendall's τ=0.77，输出侧=0.75（Table 3）
- **有效性结果**：
  - 同时效度：**IN-EmbCos** 预测输入可读性最强（Kendall=0.56, Pearson=0.68, Spearman=0.70）；**OUT-EmbCos** 预测输出可读性最强（Kendall=0.67, Pearson=0.75, Spearman=0.80）（Table 4）
  - 区分效度：忠实度度量与可读度量的相关系数为0.0–0.3，符合预期；输入/输出可读性相关<0.15
  - 收敛效度：同类度量间呈中度相关（忠实度平均~0.5）
- **方法对比**（Fig. 4）：**Sparse Autoencoder** 在所有度量上优于 Neuron 方法；但作为无监督方法，其平均质量仍低于有监督的 **TCAV**。人类评分在基线间的区分度小于自动度量。
- **最强结果**：OUT-EmbCos 在输出可读性同时效度上 Spearman 相关达 **0.80**；IN-EmbCos 在输入可读性同时效度上 Spearman 相关达 **0.70**，显著优于 LLM-Score。

## 相关工作脉络
1. **TCAV (Kim et al., 2018)**：监督式概念激活向量方法，本文将其激活函数纳入统一框架；TCAV 使用线性分类面作为概念表示，本文的 GRAD-PClass 即源于此。
2. **Concept Bottleneck Models (Koh et al., 2020)**：设计内可解释方法，通过预先定义概念模块实现可解释性；本文统一形式化可涵盖此类方法。
3. **Neuron Explainer (Bills et al., 2023)**：无监督神经元解释方法，使用 LLM 生成语义描述并基于激活方差计算可读性分数（LLM-Score）；本文证明其在可靠性上存在缺陷。
4. **Sparse Autoencoders (Cunningham et al., 2023)**：基于稀疏字典学习的解缠方法；本文实验表明其在所有度量上优于 Neuron，但弱于有监督 TCAV。
5. **NetDissect (Bau et al., 2017)**：基于像素级标注的视觉概念分析方法；本文形式化可将其激活函数形式纳入。
6. **ProtoPNet (Chen et al., 2019a)**：基于原型的可解释图像分类网络；非线性激活函数使梯度扰动不再适用，本文优化框架可处理此类非线性和非平滑激活。

## 局限性与未来方向
- 框架未涵盖概念基解释的全部评估维度，如**鲁棒性（Robustness）**和**稳定性（Stability）**需后续研究补充。
- 相干性度量并非可读性的终极解决方案，其他方面如**意义性（Meaningfulness）**也值得探索，未来可研究如何自动量化这些维度。
- 受限于 GPU 资源和预算，主实验仅使用 Pythia-70M 的中间层（第3层），未来需在更大模型上扩展验证。
- 人工评估样本量有限（200个概念×3位评分者），可能影响同时效度的估计精度。

## 研究启发与可借鉴点
1. **测量理论用于 XAI 评估的范式**：将经典心理测量学中的可靠性（重测、子集一致性、评分者间）和有效性（同时、收敛、区分）体系引入 AI 解释性评估，提供了一个系统化、可复用的元评估框架，可迁移至其他解释类型（如 attribution methods）和任务（如生成任务）。
2. **自动可读性评估替代人工评估**：OUT-EmbCos 与人工评估的高相关性（Spearman 0.80）表明深度嵌入相似度可有效替代昂贵的人工评估，适用于大规模概念筛选场景。
3. **MTMM 矩阵的跨维度验证**：通过多特质多方法矩阵同时检验可靠性与多种有效性，提供了一种全面的度量筛选流程，避免单一指标评估的片面性。
4. **优化视角的扰动设计**：将概念扰动形式化为带约束的优化问题并推导闭式解，使不同激活函数类型的概念均可公平评估，该方法论可推广至其他黑盒模型的扰动评估。
5. **IN/OUT 双侧可读性分析**：区分输入侧和输出侧的可读性（两者低相关<0.15），揭示了现有研究仅关注输入侧的不足，为后续研究提供了新的分析维度。

## 关键术语表
**Concept-based Explanation（概念基解释）**：通过识别模型学习到的高层次模式（概念）来解释模型内部行为的 XAI 方法，相比 attribution 方法提供更全局、更易理解的解释。
**Faithfulness（忠实度）**：衡量概念解释在多大程度上反映了模型内部机制的真实程度，通过扰动概念后输出变化来量化。
**Readability（可读性）**：衡量提取的概念在多大程度上能被人类理解，通过最大激活模式的语义相干性近似。
**Virtual Neuron（虚拟神经元）**：本文提出的统一概念形式化，将概念表示为定义在隐藏表示上的激活函数 $a(h): \mathbb{R}^m \rightarrow \mathbb{R}$。
**Meta-evaluation（元评估）**：对评估指标本身进行可靠性与有效性的评估，基于经典测量理论框架。
**Test-retest Reliability（重测可靠性）**：同一度量在不同时间对相同对象两次测量的结果一致性，以 Pearson 相关衡量。
**Subset Consistency（子集一致性）**：度量在不同数据子集上结果的一致性，以 Cronbach's α 衡量。
**Coherence-based Measure（相干性度量）**：基于 token 对语义相似度或共现概率自动计算概念可读性的指标，包括 UCI、UMass、EmbDist、EmbCos。

## 可复现要素
- **数据集**：Pile（公开，Hugging Face Hub 可下载），825 GiB 多语言文本与代码
- **模型**：Pythia-70M、GPT-2 Small（均公开于 Hugging Face Hub）
- **代码/权重**：论文未明确说明代码开源状态，但所有数据集和 backbone 模型均可从 Hugging Face Hub 获取
- **关键超参**：
  - Sparse Autoencoder：Adam optimizer，learning rate=1e-3，dictionary size=8×hidden dimension，L1 loss coefficient=0.5，训练11 billion activation vectors/epoch
  - TCAV：LLM有害QA作为正样本，随机文本作为负样本，训练线性分类器
  - 每个度量测试10个batch，每batch 256句×128 token
  - 可读性评估取 top-10 高激活 token
  - 人工评估：3位评分者，每概念限时5分钟，评分范围1-5
