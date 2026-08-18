---
title: "Thinking-Fair-and-Slow-On-the-Efficacy-of-Structured-Prompts"
source: https://aclanthology.org/2024.emnlp-main.13.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:53:42"
field: "语言模型公平性与可控生成"
keywords: ["LLM去偏", "Prompt工程", "公平性", "结构化提示", "System 2思维", "StereoSet", "Regard"]
innovations: ["提出基于System 2思维的三步迭代prompt去偏框架（PP/SR/IP）", "首次系统验证Implication Prompting在无需模型内部访问下的有效去偏能力", "发现Role-based prompt显著优于Instruction-based且向下兼容多模型多基准"]
benchmarks: ["StereoSet", "Regard", "RealToxicityPrompts", "TruthfulQA", "BoolQ"]
---

# 论文速读：Thinking-Fair-and-Slow-On-the-Efficacy-of-Structured-Prompts

## 一句话总结
本文探索通过结构化 prompt 工程（无需访问模型内部参数）对 LLM 进行后处理去偏，提出基于 Kahneman System 2 思维的三种迭代提示框架（Prefix Prompting、Self-Refinement、Implication Prompting），其中 Implication Prompting 在 StereoSet、Regard、Toxicity 等基准上显著优于其他方法，且不损害下游任务性能。

## 研究问题与动机
1. **现有去偏方法依赖模型内部访问权限**：如 Sent-Debias（投影去偏子空间）、SelfDebias/CAFIE（修改 logit/解码策略）、CDA 训练微调等方法均需访问模型权重或输出分布，对于仅开放 API 的闭源模型不可用。
2. **用户侧灵活适配需求缺失**：用户可能希望根据具体场景动态调整去偏程度，而重新训练或 fine-tuning 成本高且可能损害其他下游任务表现。
3. **Prompt 去偏的系统性研究空白**：虽有少量关键词级 prompt 工程研究（Borchers et al., 2022），但缺乏对复杂推理型 prompt（如 CoT、自反思）在去偏任务上的全面分析与对比。
4. **如何以"慢思考"（System 2）引导 LLM 生成公平文本**：借鉴人类决策理论，研究是否能通过结构化、反思式 prompt 诱导模型进行逻辑推理与偏见修正。

## 核心贡献（创新点）
1. **首次系统性地构建基于 System 2 思维框架的迭代 prompt 去偏体系**：将现有 prompt 方法按反馈粒度与复杂度分类为 Prefix Prompting、Self-Refinement、Implication Prompting 三大类别，并按单步/多步、指令/角色两个维度展开实验。
2. **提出 Implication Prompting (IP) 作为最有效的去偏策略**：通过让模型先生成对自身输出中偏见的推理说明（implication），再结合该推理进行去偏文本生成，实现端到端无需修改模型权重的后处理去偏。
3. **发现 Role-based prompt 显著优于 Instruction-based prompt**： Across 所有模型与基准，Role prefix 在 ICAT 上平均提升约 5%，在 Regard 上平均提升近 40%，毒性降低 4.34%。
4. **验证去偏 prompt 方法在保持下游任务性能的前提下可实现与白盒方法相当的去偏效果**：在 GPT2-small 上与 SelfDebias、CAFIE、SentenceDebias 等方法持平或超越，在更大模型（GPT-J 6B）上 SR 和 IP 也达到可比水平。

## 方法详解
本文灵感来自 Kahneman (2011) 的快速/慢速思维框架：System 1 为直觉式快速反应（直接 prefix），System 2 为缓慢、努力、逻辑性的思考（SR 和 IP）。

### 3.1 Prefix Prompting (PP)
在用户 prompt 前拼接去偏前缀，分两类：
- **Instruction PP**："The following text is unbiased and does not discriminate..."
- **Role PP**："You are an unbiased person who does not discriminate..."

### 3.2 Self-Refinement (SR)
将单步 PP 扩展为 k 步迭代。以 k=1 为例：
- **Step 1**：用 PP prompt 得到初始输出 $S_0$（可能含偏见）
- **Step 2**：将 $S_0$ 与新的 SR prompt 拼接，生成更公平的文本 $S_1$

通用形式见 Algorithm 1（k 步迭代，每次将上一步输出作为参考）。

### 3.3 Implication Prompting (IP)
三阶段流程（Algorithm 2）：
- **Step I**：用原始 prompt C 生成初始输出 S
- **Step II**：用 S 拼接 implication 生成指令 $I_{Impl}$，让模型推理 S 中存在哪些偏见（支持三种 variant：Instruction-based、Zero-Shot CoT-based、Few-shot-based）
- **Step III**：将原始输出 S、生成的 implication $S_{Impl}$、去偏指令 $I_{IP}$ 与原始 prompt C 拼接，生成最终去偏文本

## 实验与结果
**模型**：GPT-J (6B)、Mistral (7B)、Llama-2 (13B)、MPT-Instruct (7B)，均在单张 32GB V100 上运行。

**评估基准**：
- **StereoSet**：SS（刻板印象分数，理想 50%）、LM（语言建模能力，理想 100%）、ICAT（综合指标，理想 100%）
- **Regard**：测量性别/种族/性取向维度的社会感知偏差（理想 0）
- **Toxicity**：使用 RealToxicityPrompts 和仇恨言论检测器

**主要结果**（跨模型平均）：
| 指标 | IP 相对其他方法提升 |
|------|---------------------|
| StereoSet SS | -4.05%（更低=更公平） |
| StereoSet ICAT | +6.80% |
| Regard | +26.85% |
| Toxicity | -6.98%（毒性降低） |

**关键发现**：
- Role-based 在 Regard 上平均比 Instruction-based 提升约 39.47%
- SR (k=1) 比 PP 在 StereoSet ICAT 上提升 11.65%，Regard 提升 21.64%
- SR (k=2) 仅比 k=1 提升约 0.23%，更多迭代无明显收益
- IP 在 TruthfulQA 上较 base 提升约 9%，在 BoolQ 上 SR 方法表现最佳（+1%）
- 去偏 prompt 对下游任务（TruthfulQA、BoolQ）基本无损害，部分方法甚至有提升

## 相关工作脉络
1. **Sent-Debias (Liang et al., 2020)**：通过消除句表示中的偏见子空间投影去偏，属于白盒方法，需访问模型内部特征——本文方法完全基于 prompt 无需修改模型。
2. **SelfDebias (Schick et al., 2021) / CAFIE (Banerjee et al., 2023)**：利用模型输出概率和 counterfactuals 重新校准解码策略，同样需要 logits 访问权限——本文不依赖任何模型内部信息。
3. **Counterfactual Data Augmentation (CDA, Zmigrod et al., 2019)**：通过增广训练数据微调模型去偏，需要大量公平数据——本文方法无需重新训练。
4. **Borchers et al. (2022)**：探索关键词级 prompt 工程缓解招聘广告中的性别偏见，结论为 prompt 去偏效果有限；本文证明结构化、推理型 prompt 可实现显著改善。
5. **Chain-of-Thought Prompting (Wei et al., 2022; Kojima et al., 2022)**：用于提升推理能力，但鲜有工作将其系统应用于公平性任务——本文是首次将 CoT 思路引入去偏。
6. **Ma et al. (2023)**：提出 prompt-search 框架用于预测公平性，但计算开销巨大，泛化性有限——本文提供即插即用的轻量化 prompt 方案。

## 局限性与未来方向
1. **不能过度拟人化 LLM 的"思考"行为**：LLM 本质是模式匹配，并非真正进行 System 2 推理，结论需谨慎解释。
2. **复杂社会偏见的 prompt 简化风险**：将多维社会偏见压缩为 prompt 指令可能无法覆盖所有偏见形态。
3. **prompt 一致性假设的局限性**：不同模型或模型更新版本对相同 prompt 的响应可能不一致。
4. **算力限制**：未测试 70B 级模型（如 Llama-2 70B、Mixtral MoE 45B）的效果。
5. **未探索更多高级 prompt 策略**：如 Tree-of-Thought、Self-Consistency、Directional Stimulus Prompting 等。
6. **任务泛化性存疑**：框架有效性依赖模型的 latent space 中包含相关公平性知识，对不含此类信息的模型可能无效。

## 研究启发与可借鉴点
1. **System 2 思维框架可迁移到其他可控生成任务**：将"慢思考"结构化 prompt 设计思路推广至事实核查、内容安全、敏感信息过滤等场景。
2. **Role-based prompt 优于 Instruction-based 的发现具有普遍参考价值**：在需要行为约束的任务中，设定角色 persona 可能比直接指令更有效，可复用至 RLHF 对齐或合规生成任务。
3. **Implication 机制的设计思想可复用于 self-correction 框架**：先让模型识别问题所在（implication/reasoning），再基于该认知进行修正，这一两步范式适用于多个生成优化场景。
4. **固定 implication 实验的启示**：论文附录实验表明"针对具体输入的定制化推理"优于"通用偏见描述"，说明 context-aware 的解释生成是关键。
5. **与细粒度偏见类型分析结合的机会**：当前工作按通用维度评估，可进一步探索不同 prompt 策略对性别/种族/宗教/职业等细分偏见类型的差异化效果。

## 关键术语表
- **System 2 Thinking**：源自 Kahneman 的双系统决策理论，指缓慢、费力、逻辑性的思考过程，本文用于指导结构化 prompt 设计。
- **Implication Prompting (IP)**：让 LLM 先生成对自身输出中偏见含义的推理说明（implication），再据此生成去偏文本的三步提示框架。
- **Self-Refinement (SR)**：将模型自身先前生成的（可能含偏见的）输出作为参考，通过多步迭代 prompt 逐步修正文本的去偏方法。
- **Stereoset ICAT Score**：StereoSet 基准上的 Idealized Context Association Test 分数，综合衡量去偏效果与语言建模能力的权衡，越高越好。
- **Regard Score**：基于社会感知偏差的评估指标，通过正向/负向情感分类统计各人口群体获得的社会评价差异，理想值为 0。
- **Prefix Prompting (PP)**：最简单的去偏 prompt 策略，在用户输入前拼接去偏指令或角色设定，引导模型直接生成公平文本。
- **Token Vocabulary (V)**：模型支持的词表集合，所有 prompt 与输出均为 token 序列。
- **RealToxicityPrompts**：由 Gehman et al. (2020) 提出的毒性评估数据集，用于测量 LLM 生成内容的毒性程度。

## 可复现要素
- **数据集**：StereoSet、Regard、RealToxicityPrompts、TruthfulQA、BoolQ；论文未明确声明开源状态，但这些均为社区公开数据集。
- **代码/权重**：模型权重从 Huggingface 下载，训练代码未在论文中提供开源链接（论文未提及代码仓库 URL）；实验在单张 NVIDIA V100 (32GB) 上完成。
- **关键超参**：temperature=1.0；StereoSet 实验使用 repetition penalty=1.3；默认解码策略为 beam search。
- **计算资源**：单张 32GB NVIDIA V100 GPU。
