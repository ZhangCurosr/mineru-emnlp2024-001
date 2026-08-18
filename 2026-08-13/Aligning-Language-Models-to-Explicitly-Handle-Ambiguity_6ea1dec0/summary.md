---
title: "Aligning-Language-Models-to-Explicitly-Handle-Ambiguity"
source: https://aclanthology.org/2024.emnlp-main.119.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:25:11"
field: "大语言模型对齐与鲁棒性"
keywords: ["歧义处理", "大语言模型对齐", "INFOGAIN", "澄清请求", "感知歧义", "数据筛选", "AmbigQA"]
innovations: ["提出APA框架，利用模型感知歧义度（INFOGAIN）驱动小样本对齐", "设计INFOGAIN指标量化模型对歧义的感知程度", "构建AmbigTriviaQA/AmbigWebQuestions/AmbigFreebaseQA三个跨域歧义评测数据集"]
benchmarks: ["AmbigQA", "SituatedQA", "AmbigTriviaQA", "AmbigWebQuestions", "AmbigFreebaseQA"]
---

# 论文速读：Aligning-Language-Models-to-Explicitly-Handle-Ambiguity

## 一句话总结
本文提出了**Alignment with Perceived Ambiguity (APA)**，一种利用模型自身对歧义的感知程度来选择和构建训练数据、对齐LLM的方法，使模型能在保持清晰问答能力的同时，显式地识别和请求澄清歧义查询。

## 研究问题与动机
- 用户与自然语言模型交互时，常因省略或表达不精确而产生**歧义查询**，模型任意回答可能导致误解，在医疗、法律等高可靠性领域后果严重。
- 现有LLM未显式训练以处理歧义，即便能感知歧义，也缺乏明确的信号（如不确定性表达或多解枚举）供外部确认。
- 模型对歧义的**感知程度取决于其内在知识范围**——知识丰富的模型更易识别歧义，而知识受限的模型可能误判为清晰，导致同一样本在不同模型下判断不一。
- 现有歧义处理方法（如AmigQA、Tree-of-Clarification）侧重生成式或检索式策略，本文从**对齐视角**切入，探索如何让模型主动请求澄清。

## 核心贡献（创新点）
1. **提出APA对齐框架**：利用模型自身的感知歧义度（perceived ambiguity）作为隐式线索来筛选歧义样本并进行监督微调；与现有方法的区别在于不依赖人工标注的歧义标签，而是以模型知识边界驱动的样本选择替代。
2. **设计INFOGAIN歧义量化指标**：通过比较原始查询与模型自消歧结果的平均熵差来衡量歧义感知程度；与仅用熵的baseline相比，INFOGAIN更贴近"模型是否真正感知到歧义"这一目标。
3. **构建三个新型评测数据集**：AmbigTriviaQA、AmbigWebQuestions、AmbigFreebaseQA，分别基于TriviaQA、WebQuestions、FreebaseQA注入歧义并经由人工校验；填补了跨领域歧义评测空白。
4. **证明高质量小样本优于全量训练**：APA仅使用约32%（LLAMA2家族）或13%（MISTRAL）的歧义样本即可超越FULL-SET，验证了"数据质量优先于数量"的原则。

## 方法详解
APA是一个四阶段流水线：

1. **初始预测评估（Stage 1）**：用标准问答模板对所有样本做前向推理，将预测正确的样本归入$D_{correct}$，错误的归入$D_{incorrect}$。
2. **感知歧义检测（Stage 2）**：对$D_{incorrect}$中的样本，让模型进行自消歧生成$\hat{x}_{disambig}$，并计算原始查询与消歧结果的**平均熵差**作为INFOGAIN：
   $$\text{INFOGAIN}_{x,\hat{x}_{disambig}} = \mathcal{H}_x - \mathcal{H}_{\hat{x}_{disambig}}$$
   其中$\mathcal{H}_x = \frac{1}{N}\sum_i \mathcal{H}_{x,i}$，$\mathcal{H}_{x,i} = -\sum_{v \in \mathcal{V}} p_{x,i}(v)\log p_{x,i}(v)$。INFOGAIN大于阈值$\epsilon$的样本被判定为"感知歧义"。
3. **澄清响应构建（Stage 3）**：为歧义样本生成澄清请求$y_{clarify}$，有两种变体：
   - **APA_FIXED**：从预定义短语列表中随机选取一条作为$y_{clarify}$。
   - **APA_GEN**：结合$x_{ambig}$和$\hat{x}_{disambig}$让模型生成针对具体歧义源的澄清请求。
4. **监督微调（Stage 4）**：将$D_{correct}$与$D_{ambig}$平衡混合（各$m=n$个样本），构成训练集$D$，使用标准因果语言建模损失训练：
   $$\min_\theta \sum_{(x,y)\in D} \sum_{i=1}^{|y|} -\log M_\theta(y_i | y_{<i}, t(x))$$

## 实验与结果
- **数据集**：训练/验证使用AmbigQA（包含歧义与非歧义查询）；OOD测试使用SituatedQA-Geo、SituatedQA-Temp，以及三个自建数据集AmbigTriviaQA、AmbigWebQuestions、AmbigFreebaseQA。
- **评估指标**：非歧义F1（$F1_u$）衡量清晰问答能力；歧义检测F1（$F1_a$）衡量澄清请求生成能力。
- **基线**：推理-only方法（DIRECT、AMBIG-AWARE、SAMPLE REP、SELF-ASK）；训练方法（FULL-SET、SUBSET_RAND、SUBSET_ENT）。
- **模型**：LLAMA2 7B/13B、MISTRAL 7B，采用QLoRA（r=4, α=16）高效微调。
- **最强结果（LLAMA2 7B）**：
  - APA_GEN在AmbigFreebaseQA取得最高$F1_a$ **84.90**，$F1_u$为**73.18**。
  - APAFIXED在AmbigTriviaQA取得$F1_a$ **75.50**，$F1_u$为**62.97**。
  - APA整体在全部数据集上均超越所有基线，相比SUBSET_ENT提升约**2-6分**。
- **消融结论**：
  - INFOGAIN选择显著优于RAND（约+1~4分）和MIN（最低INFOGAIN样本，导致性能下降）。
  - APA在所有阈值下均优于全量训练和随机采样。
  - APAFIXED通常略优于APA_GEN（生成式澄清任务更难）。

## 相关工作脉络
- **AmbigQA（Min et al., 2020）**：开放域QA中的歧义数据集，本文在其基础上扩展OOD评测，而非重新设计任务。
- **Cole et al. (2023) Sample Repetition**：通过多次采样的一致性衡量不确定性，用于歧义检测；本文将其作为推理型baseline对比，证明训练对齐优于纯推理方法。
- **Kim et al. (2023a) Tree-of-Clarification**：利用检索增强生成澄清树；本文聚焦于端到端SFT对齐路径，不需外部检索。
- **LIMA（Zhou et al., 2024）/AlpagaGas（Chen et al., 2024）**：小样本高质量数据对齐的思想，本文进一步验证"仅选感知歧义样本"同样有效。
- **AmbigCoref（Yuan et al., 2023）**：评估对歧义共指的敏感度，本文则面向生成式澄清请求。
- **RLHF/DPO**：主流对齐范式，本文指出未来可扩展至DPO等偏好优化方法。

## 局限性与未来方向
- 研究范围主要限于**短形式QA**，尚未扩展到长篇生成或复杂推理场景。
- 仅考虑**单轮查询**，未涉及多轮对话中的上下文依赖歧义（如Conversational QA场景）。
- 实验仅覆盖LLAMA2和MISTRAL系列，**模型架构和规模不够多样**，更大规模模型的倾向性有待验证。
- 当前仅使用**SFT**对齐，未探索RLHF或DPO等偏好优化方法的优势。
- 自消歧过程中存在少量**误分类**：如模型因幻觉补充信息导致误判歧义，或因同一标题的多版本存在而将清晰查询误标为歧义。

## 研究启发与可借鉴点
1. **INFOGAIN作为隐式数据筛选信号**：可用于其他需要区分"模型已知/未知"的任务，如事实性校验、知识边界检测，替代昂贵的标注成本。
2. **感知歧义驱动的小样本对齐**：证明只需精选模型"真正困惑"的样本（约30%数据），即可优于全量训练，为低资源指令微调提供了新思路。
3. **澄清请求生成的prompt模板设计**：APA_GEN中结合原始查询与消歧结果的prompt模板（Table 6）可直接迁移至其他需要模型主动询问的场景（如客服Agent）。
4. **Misaligned Clarification Request Rate (MCR) 评估指标**：本文为衡量"对齐后对原有能力的损害"设计了MCR，可推广至其他alignment任务的稳定性评估。
5. **自建数据集构建流程**：利用GPT-4注入歧义+二次验证+人工校验的闭环流程，可作为歧义类数据构建的标准范式参考。

## 关键术语表
**Perceived Ambiguity**：模型基于自身知识边界对输入歧义程度的主观判断，不同于人工标注的ground-truth歧义。
**INFOGAIN**：原始查询与模型自消歧结果之间平均熵的差值，用于量化模型从消歧过程中获取的信息量。
**Clarification Request**：模型针对歧义查询生成的请求澄清的回复，用于引导用户提供更多信息。
**Misaligned Clarification Request Rate (MCR)**：对齐后，原本能正确回答的非歧义样本被错误转为生成澄清请求的比例，衡量对齐对原有能力的破坏程度。
**AmbigQA**：包含歧义和非歧义查询的开放域QA数据集，本文作为in-domain训练数据。
**QLoRA**：基于4bit量化的低秩适配微调方法，本文用于高效对齐LLM。
**$F1_u$ / $F1_a$**：非歧义预测F1和歧义检测F1，分别评估模型在清晰问答和澄清请求上的综合性能。
**Ambiguation**：通过修改名词短语的具体性或省略关键细节来使原始查询产生歧义的数据构建过程。

## 可复现要素
- **数据集**：AmbigQA（开源）、SituatedQA（开源）；AmbigTriviaQA、AmbigWebQuestions、AmbigFreebaseQA为本刊新建数据集，论文声明代码和数据在GitHub公开（https://github.com/heyjoonkim/APA）。
- **代码/权重**：代码已开源；未提供预训练权重，需使用LLAMA2/MISTRAL开源权重自行微调。
- **关键超参**：QLoRA的$r=4$、$\alpha=16$；阈值$\epsilon=0.1$；学习率从$\{1e-3, 5e-4, 1e-4\}$中选取；batch size=32；训练epoch从$\{1, 2, 3\}$中选取；3次随机种子平均。
