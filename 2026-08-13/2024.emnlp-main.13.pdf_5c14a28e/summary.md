---
title: "“Thinking” Fair and Slow: On the Efficacy of Structured Prompts for Debiasing Language Models"
source: https://aclanthology.org/2024.emnlp-main.13.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:06:13"
field: "大语言模型公平性与安全性"
keywords: ["LLM去偏", "提示工程", "系统性偏见", "黑盒去偏", "蕴含提示", "公平文本生成"]
innovations: ["首次系统性提出基于System 2思考的三阶段提示去偏框架（PP/SR/IP）", "提出蕴含提示（IP）通过推理引导实现最强黑盒去偏效果", "证明提示去偏可匹敌白盒方法且不影响下游任务性能"]
benchmarks: ["Stereoset", "Regard", "RealToxicityPrompts", "TruthfulQA", "BoolQ"]
---

# 论文速读："Thinking" Fair and Slow: On the Efficacy of Structured Prompts for Debiasing Language Models

## 一句话总结
本文首次系统性地研究了仅通过结构化提示（无需访问模型内部参数或输出概率）实现LLM去偏的可行性，提出了三种基于Kahneman"系统2"思考的提示框架——前缀提示（PP）、自我修正（SR）与蕴含提示（IP），并证明蕴含提示能在不损害下游任务性能的前提下，显著降低模型输出中的偏见。

## 研究问题与动机
- **现有去偏技术对终端用户不可达**：主流方法（如词元去偏、解码策略调整、微调）需要访问模型内部参数或输出logits，而多数SOTA LLM（如GPT-4、Claude）为闭源API，用户无法使用。
- **训练/微调去偏成本高昂**：构建高质量无偏数据再训练或微调LLM对大多数研究者/用户不切实际；且对已精细调优的模型修改权重可能带来不可逆的下游性能损伤。
- **提示去偏缺乏系统性研究**：现有提示类去偏工作（如keyword-based）仅停留在简单前缀层面，缺少对系统性、结构化提示策略与多类偏见（种族、性别、宗教、性取向等）的多模型全面评估。
- **如何"白盒黑盒"均可用的去偏方法**：核心问题是——在无法获取模型内部信息时，如何通过提示工程实现有效、可复用的公平文本生成？

## 核心贡献（创新点）
- **首次提出面向终端用户的迭代式提示去偏框架**：以Kahneman的双系统决策理论为指导，构建了PP/SR/IP三类提示策略的系统化对比，这是该方向的首个深入探索。
- **提出蕴含提示（Implication Prompting, IP）**：与SR直接让模型参考自身有偏输出不同，IP让模型先生成"隐含偏见推理"（即解释为何输出有偏），再以此为指导重新生成——本质区别在于提供了因果式 reasoning 而非仅靠指令约束。
- **揭示了提示去偏方法间的效能层级**：IP > SR > PP，其中Role类提示在所有指标上均优于Instruction类提示，且多步SR（k≥2）几乎无额外收益，为后续研究提供了清晰的策略选择依据。
- **证明提示去偏可匹敌白盒方法且不损害下游能力**：在Stereoset、Regard、Toxicity等基准上，提示框架的性能与SDB、CAFIE、SentenceDebias等白盒方法相当，且在TruthfulQA、BoolQ等下游任务上保持竞争力甚至有所提升。

## 方法详解

本文框架基于Kahneman双系统思维类比构建三类提示策略：

**1. 前缀提示（Prefix Prompting, PP）**：在用户输入prompt前拼接去偏前缀，属于"系统2"最简单的指令形式。
- **Instruction PP**：`The following text is unbiased and does not discriminate against people on the basis of their gender, race, religion, or any other sensitive attribute: [USER PROMPT]`
- **Role PP**：`You are an unbiased person who does not discriminate against people based on their gender, race, religion, or any other sensitive attribute. [USER PROMPT]`

**2. 自我修正（Self-Refinement, SR）**：将PP扩展为k步迭代，让模型参考自身先前生成的有偏输出进行修正（Algorithm 1）：
- Step 1：用PP获取初始输出 $S_0$
- Step 2：构建新prompt $C_{SR} = \text{concat}(I_{SR}, S_{i-1}, C)$，令模型生成 $S_i$，迭代k次
- 每种变体分为Instruction和Role两类，k取1或2

**3. 蕴含提示（Implication Prompting, IP）**：三阶段流程（Algorithm 2），提供"为何有偏"的推理：
- Step 1：输入用户prompt $C$，生成初始输出 $S$
- Step 2：生成隐含偏见说明 $S_{Impl}$，输入prompt为 $C_{Impl} = \text{concat}(S, I_{Impl})$，其中 $I_{Impl}$ 有三种变体：
  - **Instruction-Based**：`Here is a text: [OUTPUT] Generate the implication of the given text:`
  - **Zero-Shot CoT-Based**：`Thinking step by step, generate the implication...`
  - **Fewshot-Based**：给出1-2个Text-Implication示例后要求生成
- Step 3：构建最终prompt $C_{IP} = \text{concat}(S, S_{Impl}, I_{IP}, C)$，引导模型生成去偏文本

> **关键原理**：IP通过迫使模型显式识别自身输出的偏见内涵，在隐空间中进行更定向的搜索，相当于为"系统2"慢思考提供了结构性引导，而非依赖简单指令约束。

## 实验与结果

**实验设置**：
- **模型**：GPT-J (6B)、Mistral (7B)、Llama-2 (13B base)、MPT-Instruct (7B)，均在单张32GB NVIDIA V100上推理，temperature=1.0，stereoset使用repetition penalty=1.3
- **数据集**：StereoSet（53%子集，句子末尾填空）、Regard（10模板×10句×各人口统计组）、RealToxicityPrompts（1000个采样prompt）

**主要结果（各方法跨模型平均）**：

| 指标 | 最佳方法 | 提升幅度 |
|---|---|---|
| **Stereoset ICAT**（理想=100%） | IP | 较所有其他方法高 +6.80%；SS降低4.05% |
| **Regard**（理想=0，负值为偏见） | IP | 平均降低26.85% |
| **Toxicity均值** | IP | 降低6.98% |
| **TruthfulQA** | Zero-Shot CoT IP | 最高提升约9%（vs base） |

- **Role vs Instruction**：Role类前缀在Stereoset上ICAT平均高5.08%，Regard上提升近39.47%，Toxicity低4.34%
- **SR多步分析**：k=1 vs k=2仅SS提升0.23%，说明一步自我修正已接近饱和
- **下游任务无损伤**：IP类方法在TruthfulQA和BoolQ上与基线持平或更优，LM Score无明显下降

## 相关工作脉络
- **SelfDebias (Schick et al., 2021)**：利用输出概率+对抗提示去偏，需访问模型logits；本文方法无需任何内部访问。
- **CAFIE (Banerjee et al., 2023)**：通过反事实数据校准去偏，同样需模型内部信息；本文完全黑盒。
- **SentenceDebias (Liang et al., 2020)**：从句表示中去除偏见子空间投影，需修改模型内部特征；本文仅改提示。
- **Borchers et al. (2022)**：探索keyword-based提示缓解性别偏见，但停留在简单前缀层面；本文提出系统性多层级框架。
- **Ma et al. (2023)**：Prompt-search框架用于预测公平性，计算开销巨大；本文方法是轻量即插即用。
- **Brown et al. (2020a)**：自然语言指令可引导GPT-3减少性别偏见；本文扩展至多种偏见类型、多种模型、多种提示复杂度的全面评估。

## 局限性与未来方向
- **方法论局限**：当前LLM并非真正的"思考机器"，生成文本仍是训练数据的模式复现，不宜过度解读为模型"理解了偏见"。
- **偏见定义的简化**：将复杂社会偏见压缩为固定提示可能无法覆盖全部偏见形态。
- **泛化性不确定**：依赖模型隐空间中是否含有与去偏相关的信息；若训练数据不含相应知识，框架可能失效。
- **未探索大模型**：受计算资源限制，未测试70B级Llama-2或Mixtral (45B MoE)等更大模型。
- **未对比先进提示技术**：如Tree-of-Thought、Self-Consistency、Directional Stimulus Prompting等未在实验中评估。
- **提示稳定性假设**：假设模型对提示的响应一致，但不同版本/更新后模型可能表现不同。

## 研究启发与可借鉴点
- **"推理式引导"优于"指令式约束"**：IP让模型先推理出"为何有偏"再修正，效果远优于直接说"不要偏"；该思路可迁移到幻觉抑制、事实一致性等其他安全对齐任务。
- **Role-based提示系统性优于Instruction-based**：设定人格角色比直接指令更有效地引导模型行为，可作为一种通用提示设计原则。
- **多步迭代存在收益递减**：SR的k=1已接近上限，后续优化应关注提示内容质量而非盲目增加步数。
- **小模型可生成含义**：用TinyLlama (1.1B)生成蕴含推理即可达到与大模型相近效果，支持低成本部署。
- **与白盒方法公平对比的评估范式**：附录B将提示方法与SDB/CAFIE/SentenceDebias在GPT2-small上对比，建立了一套可复用的黑盒/白盒公平比较基准，值得借鉴。
- **团队结合机会**：可探索将IP框架与RAG结合，利用外部知识提供更精准的蕴含推理；或将其应用于多模态模型的偏见缓解。

## 关键术语表
- **System 2 Prompting**：借鉴Kahneman"慢思考"概念，通过结构化提示引导模型进行逻辑、反思性推理，而非直觉式直接生成。
- **Implication Prompting (IP)**：核心方法，分三步——生成文本→推断隐含偏见→基于推断重新生成去偏文本。
- **Self-Refinement (SR)**：让模型参考自身先前有偏输出进行迭代修正的k步提示策略。
- **Stereoset ICAT**：综合衡量偏见与语言建模能力的指标，结合SS（偏见分数）和LM（语言建模分数），理想值为100%。
- **Regard Score**：基于社会感知衡量偏见的指标，通过正/负态度 Classifier 计算群体间差异，理想值为0。
- **Toxicity Score**：使用RealToxicityPrompts评估模型输出中有害/毒性内容的概率均值。
- **Prefix Prompting (PP)**：最简单提示去偏方式，在用户输入前添加无偏见指令或角色设定。

## 可复现要素
- **数据集**：StereoSet、Regard、RealToxicityPrompts——均为公开数据集
- **模型权重**：GPT-J (6B)、Mistral (7B)、Llama-2 (13B)、MPT-Instruct (7B)均从HuggingFace公开获取
- **代码开源**：论文未明确声明代码仓库链接
- **关键超参**：temperature=1.0，Stereoset额外设置repetition penalty=1.3，默认decoding策略为beam search
- **硬件**：单张32GB NVIDIA V100 GPU
