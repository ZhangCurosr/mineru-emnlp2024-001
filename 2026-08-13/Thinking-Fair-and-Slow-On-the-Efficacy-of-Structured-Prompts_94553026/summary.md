---
title: "Thinking-Fair-and-Slow-On-the-Efficacy-of-Structured-Prompts"
source: https://aclanthology.org/2024.emnlp-main.13.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:33:05"
field: "大语言模型的公平性与安全"
keywords: ["去偏见", "大语言模型", "提示工程", "System 2 思维", "公平性", "黑盒去偏"]
innovations: ["提出基于 System 2 思维的三类结构化提示去偏框架（PP/SR/IP）", "发明 Implication Prompting 三步推理策略，让模型自行推断并修正隐含偏见", "证明黑盒提示去偏可达到与白盒方法相当效果且不影响下游任务性能"]
benchmarks: ["StereoSet", "Regard", "RealToxicityPrompts", "TruthfulQA", "BoolQ"]
---

# 论文速读：Thinking-Fair-and-Slow-On-the-Efficacy-of-Structured-Prompts

## 一句话总结
本文探索了在不依赖模型内部参数和 logits 访问的情况下，通过结构化提示（Prompting）策略对大语言模型进行去偏见的可行性；提出基于 System 2 思维的三类提示框架（Prefix Prompting、Self-Refinement、Implication Prompting），实验表明 Implication Prompting 在多个基准上显著优于其他方法，且能保持下游任务性能。

## 研究问题与动机
1. **核心问题**：现有去偏见技术（如微调、logit 校准、特征空间去偏）需要访问模型内部参数、训练数据或解码策略，对依赖闭源 API 的终端用户不可行。
2. **现实约束**：即使拥有开源模型，高质量去偏数据的构建成本高昂；且对已精心预训练的 LLM 修改权重或解码策略可能损害其他下游任务性能。
3. **研究空白**：尽管复杂提示策略已被广泛应用于推理增强，但系统性地研究其对公平文本生成的作用尚属空白；既有工作局限于关键词或简单前缀提示，未深入探索迭代、多步、反馈型去偏提示。
4. **动机**：探索终端用户能否仅通过"提示"实现对 LLM 输出的自适应去偏见，而无需修改模型本身。

## 核心贡献（创新点）
1. **首次系统评估系统性提示去偏框架**：提出以 Kahneman System 2 思维为隐喻的三类提示策略（PP/SR/IP），覆盖单步与多步、指令与角色扮演两种变体，填补公平文本生成中提示工程研究的空白。
2. **Implication Prompting (IP) 的三步推理框架**：创新性地让 LLM 先生成初始文本，再自行推断其中的隐含偏见（Implication），最后结合推理进行去偏重构；该策略在 StereoSet、Regard、Toxicity 三大基准上均取得最佳性能。
3. **端到端用户友好的去偏范式证明可行性**：实验表明 IP 等方法无需访问模型内部，即可达到与白盒方法（如 SelfDebias、CAFIE）相当的去偏效果，同时保持 TruthfulQA、BoolQ 等下游任务性能不下降甚至提升。

## 方法详解
借鉴人类"快思考-慢思考"决策理论（Kahneman, 2011），将提示策略分为三类：

**3.1 Prefix Prompting (PP)**
- 在用户提示 $C$ 前拼接去偏前缀 $I_{\text{debias}}$，得到 $C_{\text{debias}} = \text{concat}(I_{\text{debias}}, C)$。
- 两种变体：**Instruction PP**（直接指令"生成无偏见文本"）和 **Role PP**（角色设定"你是一个无偏见的人"）。

**3.2 Self-Refinement (SR) — k 步迭代**
- Step I：用 PP 策略生成初始输出 $S_0$。
- Step II：将 $S_0$ 与 SR 前缀 $I_{\text{SR}}$ 拼接，引导模型参考自身输出进行去偏生成，迭代 $k$ 次：
  $$C_{\text{SR}} = \text{concat}(I_{\text{SR}}, S_{i-1}, C), \quad S_i = M(C_{\text{SR}}, V)$$
- 最终输出 $S_{\text{debiased}} = S_k$。

**3.3 Implication Prompting (IP) — 三步推理**
- **Step I**：由用户提示 $C$ 生成初始输出 $S$（可能含偏见）。
- **Step II**：构造 $C_{\text{Impl}} = \text{concat}(S, I_{\text{Impl}})$，要求模型推断 $S$ 中隐含的偏见（生成 Implication $S_{\text{Impl}}$）。提供三种 $I_{\text{Impl}}$：Instruction-based、Zero-Shot CoT-based、Fewshot-based。
- **Step III**：构造最终提示 $C_{\text{IP}} = \text{concat}(S, S_{\text{Impl}}, I_{\text{IP}}, C)$，引导模型结合原始输出与偏见推理生成去偏文本 $S_{\text{debiased}}$。

**核心思想**：越复杂的 System 2 提示（需要逻辑推理、自我反思）越能有效引导模型在潜在空间中搜索去偏路径，而非仅依赖表面关键词匹配。

## 实验与结果
**模型**：GPT-J (6B)、Mistral-v0.1 (7B)、Llama-2 (7B/13B)、MPT-Instruct (7B)。

**评测基准与指标**：
- **StereoSet**：SS（偏见倾向分，理想=50%）、LM（语言建模能力，理想=100%）、ICAT（偏见与建模能力的权衡，理想=100%）。
- **Regard**：针对性别、种族、性取向三个属性的 regard 分差（理想=0，负值表示有偏见）。
- **Toxicity**：RealToxicityPrompts 数据集上的平均毒性分数。
- **下游任务**：TruthfulQA、BoolQ。

**主要结果**：
- **IP 整体最优**：平均较其他方法 SS 降低 4.05%、ICAT 提升 6.80%，Regard 提升 26.85%，Toxicity 降低 6.98%。
- **Role vs Instruction**：Role-based 提示在所有基准上均优于 Instruction-based（ICAT 高 1.7%，Toxicity 低 4.5%）。
- **SR 迭代收益有限**：SR(k=1) 较 PP 有显著提升（SS 降低 6.85%，ICAT 提升 11.65%），但 k=2 及以上仅带来边际改善（<0.5%）。
- **下游任务无损**：IP 和 SR 方法在 TruthfulQA 和 BoolQ 上表现与基线相当甚至更优（TruthfulQA 最高提升约 9%）。
- **Implication 生成模型大小可伸缩**：使用 TinyLlama (1.1B) 或 Llama-2 (13B) 生成 Implication，与使用原模型生成性能相近，支持低延迟场景。

## 相关工作脉络
1. **White-box 去偏方法**（SelfDebias, CAFIE, SentenceDebias）：依赖模型 logits 或内部特征修改，需要 API 权限或重新训练；本文定位为"黑盒用户友好"替代方案。
2. **关键词/简单提示去偏**（Borchers et al., 2022）：仅用关键词替换或简单指令，去偏效果有限；本文推进到多步推理型提示。
3. **CoT/Few-shot 提示**（Wei et al., 2022; Kojima et al., 2022）：用于提升推理能力，但未系统研究其对公平性的影响；本文将其引入去偏场景并证明有效性。
4. **Self-Refinement**（Madaan et al., 2023）：自反馈迭代优化；本文将其应用于去偏并发现 k=1 即达饱和。
5. **Fairness-guided Prompting**（Ma et al., 2023）：需要大量计算搜索最优 prompt；本文方法零成本、即插即用。
6. **Prompt-based Debiasing 边界工作**：Brown et al. (2020a) 证明自然语言指令可引导 GPT-3 减少性别偏见，但缺乏系统性比较；本文首次全面评估不同提示复杂度与模型/数据集的交互效应。

## 局限性与未来方向
1. **隐喻局限**："System 2 思考"仅为启发式框架，LLM 本质仍是模式匹配而非真正推理，结论不应过度解读为"模型具备反思能力"。
2. **社会偏见的简化风险**：将复杂社会偏见压缩为固定提示可能无法覆盖所有偏见形态；提示一致性假设在不同模型版本间可能失效。
3. **计算资源约束**：未测试 70B 级模型（如 Llama-2-70B、Mixtral-45B MoE），泛化性存疑。
4. **未探索的高级提示**：Tree-of-Thought、Self-Consistency、Directional Stimulus Prompting 等前沿方法未被纳入比较。
5. **任务泛化性未知**：System 2 提示仅在模型 latent space 包含相关去偏信息时有效，对超出训练数据分布的任务可能失效。
6. **未来方向**：设计更具挑战性的去偏任务、探索动态 prompt 搜索策略、扩展至多模态场景。

## 研究启发与可借鉴点
1. **System 2 提示的通用价值**：将"慢思考"提示框架引入 fairness 领域，启发了后续研究将 CoT/反思机制用于其他伦理属性（如可解释性、隐私保护）。
2. **k=1 迭代的性价比**：SR 方法证明单次迭代即可捕获大部分增益，为实际部署节省计算开销；后续工作可参考此设计原则平衡效果与延迟。
3. **Role-based 提示优于 Instruction-based**：角色设定比直接指令更有效，这一发现适用于所有面向公平性/安全性的 prompt engineering 工作。
4. **Implication 生成的可伸缩性**：用小模型生成 Implication 不影响去偏效果，为低资源场景提供可行方案；可迁移至 RAG/Agent 系统中按需调用轻量模块。
5. **端到端可复现的去偏流水线**：本文方法无需修改模型权重，可直接集成到现有 LLM 服务管道中，为工业界落地提供低风险路径。

## 关键术语表
**System 2 Prompting**：借鉴 Kahneman 双系统理论，指需要逻辑推理、自我反思的慢速提示策略，区别于直觉式的 System 1（简单前缀指令）。

**Implication Prompting (IP)**：三步提示框架，先让模型生成输出，再推断其中的隐含偏见，最后结合推理重构去偏文本。

**Stereotype Score (SS)**：StereoSet 指标，衡量模型倾向生成刻板印象句子的概率，理想值为 50%。

**ICAT (Idealized Context Association Test)**：StereoSet 综合指标，结合 SS 和 LM 得分反映去偏与语言建模能力的权衡，理想值为 100%。

**Regard Score**：衡量模型对不同人口统计学群体的社会评价差异，理想值为 0（无偏见）。

**Self-Refinement (SR)**：让模型参考自身先前生成的有偏输出，迭代修正为去偏文本的多步提示策略。

**Prefix Prompting (PP)**：在用户提示前拼接去偏指令或角色设定的最基础提示策略。

**Token Vocabulary (V)**：模型生成的离散符号集合，LLM 在此空间上采样生成文本序列。

## 可复现要素
- **数据集**：StereoSet（公开，https://stereoset.stanford.edu/）、Regard（公开，https://regard.cs.cmu.edu/）、RealToxicityPrompts（公开，https://github.com/google-research-datasets/real-toxicity-prompts）、TruthfulQA（公开）、BoolQ（公开）。
- **代码**：论文未明确声明 GitHub 仓库，附录包含完整 prompt 模板与实验细节。
- **模型权重**：GPT-J、Mistral、Llama-2、MPT-Instruct 均可从 Hugging Face 获取。
- **关键超参**：temperature=1.0，StereoSet 上使用 repetition penalty=1.3，默认解码策略为 beam search；SR 迭代次数 k=1 为主要配置。
