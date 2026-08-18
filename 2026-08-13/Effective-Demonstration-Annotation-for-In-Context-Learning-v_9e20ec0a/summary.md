---
title: "Effective-Demonstration-Annotation-for-In-Context-Learning-v"
source: https://aclanthology.org/2024.emnlp-main.74.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:33:58"
field: "大语言模型提示工程"
keywords: ["In-Context Learning", "Selective Annotation", "Determinantal Point Process", "Active Learning", "Few-shot Learning", "Demonstration Selection"]
innovations: ["提出LM-DPP框架，通过条件DPP联合建模不确定性与多样性进行ICL演示集选择", "设计可解释的DPP核分解策略实现贪心高效推断", "验证方法跨模型规模（6B至175B）与预算的泛化性"]
benchmarks: ["GLUE", "Super-NaturalInstructions", "Hellaswag", "COPA", "XSUM", "Natural Questions"]
---

# 论文速读：Effective-Demonstration-Annotation-for-In-Context-Learning-via-LM-DPP

## 一句话总结
本文从主动学习视角重新审视ICL的标注策略，提出LM-DPP方法，通过条件行列式点过程（DPP）联合建模大语言模型的不确定性与样本多样性，在极端低资源场景（仅16条标注）下显著优于现有选择方法。

## 研究问题与动机
- **问题**：现有ICL研究高度依赖大规模标注支持集，但在实际场景中难以获取，需解决"如何用极少量标注数据构建高效演示集"的问题。
- **现有方法不足**：Vote-k等代表性选择方法忽视不确定性评估及上下文示例间的关联性；Perplexity基线仅关注低不确定性，缺乏多样性约束。
- **核心洞察**：高语义相似度、低不确定性与高多样性构成有效且高效的标注策略（Margatina et al., 2023）。
- **设定**：聚焦"极端低资源ICL"场景，即可用标注示例仅限几十条，目标识别能最大化利用LLM能力的特定演示子集。

## 核心贡献（创新点）
1. **提出LM-DPP框架**：将ICL重新建模为选择性标注问题，首次通过条件DPP联合优化不确定性与多样性；区别于仅依赖语义相似度（Vote-k）或单一困惑度排序的方法。
2. **设计可解释的DPP核矩阵分解**：将DPP核分解为$L = \text{Diag}(r) \cdot \phi \cdot \text{Diag}(r)$，其中$r_i$为困惑度倒数（不确定性度量），$\phi_i$为归一化特征向量（多样性度量）；区别于vanilla DPP仅建模多样性的传统方式。
3. **实验验证跨模型泛化性**：在GPT-J、LlaMA-2-7B及GPT-3.5-Turbo（175B）上均取得显著提升；相较Fast Vote-k平均提升3.25%，证明方法不依赖特定模型规模。
4. **揭示金标标签的价值边界**：通过UN-LM-DPP实验发现，Gold标注对ICL性能有正向贡献，但在QNLI等高金标比例任务中伪标签可部分补偿标注成本。

## 方法详解
**整体流程**（三步）：
1. **不确定性建模**：采用SPELL方法，使用LLM困惑度评估候选实例$x$的得分$r(x) = 1/\text{PPL}(x) = \exp\left(\frac{1}{t}\sum_{i=1}^{t}\log G_\theta(\tilde{x}_i|\tilde{x}_{<i})\right)$，困惑度越低表示模型越"熟悉"该样本。
2. **条件DPP联合建模**：构建核矩阵$L' = \text{Diag}(\exp(\alpha r)) \cdot \phi \cdot \text{Diag}(\exp(\alpha r))$，其中$\alpha = \lambda/(2(1-\lambda))$；对数似然函数为$\log\det(L_S)' = \lambda\sum_{i\in S}r_i + (1-\lambda)\log\det(L_S)$，通过超参$\lambda$调控不确定性与多样性权衡。
3. **贪心MAP推断**：使用Cholesky分解增量更新，每次选择使边际增益$\log\det(L_{S\cup\{j\}}) - \log\det(L_S)$最大的样本，复杂度从$O(K^3)$降至$O(NK^2)$；测试时采用Sentence-BERT余弦相似度检索Top-K相似示例，并按相似度递增排序（利用LLM的recency bias）。

## 实验与结果
- **数据集**：9个NLU任务（RTE、MNLI、MRPC、QNLI、SST-5、TREC、DBpedia、Hellaswag、COPA）+ 2个生成任务（NQ、XSUM）。
- **基线**：Random、Perplexity、K-means、Vote-k、Fast Vote-k。
- **主要结果**（GPT-J，预算|L|=100）：
  - NLU平均准确率：**65.83%**（LM-DPP）vs 64.68%（Fast Vote-k），绝对提升**1.15%**。
  - 在MRPC（67.10）、QNLI（53.26）、TREC（68.92）上取得最优；标准差仅2.0，显著稳定。
- **小预算场景**（|L|=16）：LM-DPP在RTE、Hellaswag、QNLI上仍显著优于Fast Vote-k，证明极端低资源下有效性。
- **大模型迁移**（GPT-3.5-Turbo 175B）：较Random在TREC上+5.6%，MNLI +1.8%，平均超越Fast Vote-k **3.25%**。
- **生成任务**（LlaMA-2-7B，|L|=100）：XSUM ROUGE-L提升**2.18%**，但FactCC事实一致性略有下降（追求多样性导致主题偏离）。

## 相关工作脉络
1. **Vote-k (Su et al., 2022)**：图投票机制选择多样且代表性的演示集，但忽略不确定性评估及示例间关联——LM-DPP通过条件DPP同时建模两者。
2. **SPELL (Gonen et al., 2022)**：发现低困惑度与ICL性能正相关，仅按困惑度排序选样——LM-DPP在此基础上引入多样性约束避免冗余。
3. **CEIL (Ye et al., 2023)**：训练可学习的条件DPP检索器，但需大量标注数据微调——LM-DPP无需训练，直接应用于预训练LLM。
4. **代表性演示选择 (Zhao et al., 2023)**：使用两阶段DPP进行 corpus-level ICL，仍需gold labels评分——LM-DPP探索无标注条件下的替代策略。
5. **知识蒸馏提示选择 (Li & Qiu, 2023)**：基于实例级相似度检索——LM-DPP在支持集构建阶段即考虑多样性与不确定性平衡。

## 局限性与未来方向
- **不确定性度量单一**：仅使用困惑度作为不确定性代理，未探索熵或其他AL不确定性指标对ICL的影响。
- **检索方法受限**：仅验证相似度Top-K检索，未测试Coverage-based Retrieval或互信息检索等变体。
- **检索器单一**：仅使用Sentence-BERT架构，未验证其他编码器（如BERT、Contriever）的泛化性。
- **语言局限**：所有数据集为英语，跨语言泛化性未验证。
- **未来方向**：探索无标注ICL（annotation-free）、扩展至更多NLP任务、结合强事实核查机制缓解生成任务的事实不一致问题。

## 研究启发与可借鉴点
1. **条件DPP核分解策略**：$L = \text{Diag}(r) \cdot \phi \cdot \text{Diag}(r)$的可解释分解可迁移至其他资源选择任务（如RAG文档检索、主动学习采样）。
2. **贪心MAP推断的Cholesky增量更新**：$O(NK^2)$复杂度实现大规模候选集高效筛选，可直接复用于其他DPP应用。
3. **超参$\lambda$的任务敏感性分析**：发现不同任务对不确定性-多样性权衡敏感度不同（如QNLI仅1.95%波动，DBpedia超10%），提示未来工作需进行任务自适应调参。
4. **小模型替代评分**：GPT-2（117M）作为GPT-J的困惑度估算器可实现10倍加速且精度损失仅0.07%，为大规模候选池场景提供实用方案。

## 关键术语表
- **In-Context Learning (ICL)**：无需参数更新，通过在提示中提供输入-输出示例让LLM学习并泛化到新样本的范式。
- **Determinantal Point Process (DPP)**：建模负相关性（多样性）的概率模型，通过行列式核矩阵衡量子集多样性。
- **Perplexity-based Uncertainty**：利用LLM对样本的困惑度倒数作为不确定性度量，低困惑度表示模型"熟悉"该样本。
- **Selective Annotation**：从大量未标注数据中有选择地标注最有价值的样本，以降低人工标注成本。
- **MAP Inference**：寻找最可能子集的极大后验估计，DPP的MAP推断为NP-hard，常用贪心算法近似。
- **Recency Bias**：LLM倾向于更重视提示序列末尾示例的认知偏差，用于指导演示排序策略。
- **FactCC**：基于BERT的事实一致性评估指标，衡量生成文本与源文本的事实吻合度。
- **UN-LM-DPP**：无监督变体，使用伪标签构建训练集并结合少量金标数据进行Icl验证。

## 可复现要素
- **数据集**：SST-5、RTE、MNLI、MRPC、QNLI、TREC、DBpedia、Hellaswag、COPA、CosmosQA、XSUM、NQ（均为公开数据集）。
- **代码/权重**：论文未提供开源代码链接；使用GPT-J-6B、LlaMA-2-7B、GPT-3.5-Turbo API。
- **关键超参**：
  - 权衡系数$\lambda \in \{0.0, 0.2, ..., 1.0\}$，最优值通常在0.5-0.6。
  - 标注预算$|{\mathcal{L}}| \in \{16, 100, 300, 800\}$。
  - 检索器：paraphrase-mpnet-base-v2（Sentence-BERT）。
  - 最大token长度：GPT-J为2048，LlaMA-2为4096。
- **实验环境**：单张Tesla V100 GPU（32GB），CPU生成标注集约6秒。
