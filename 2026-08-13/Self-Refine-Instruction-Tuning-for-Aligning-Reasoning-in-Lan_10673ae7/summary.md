---
title: "Self-Refine-Instruction-Tuning-for-Aligning-Reasoning-in-Lan"
source: https://aclanthology.org/2024.emnlp-main.139.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:30:32"
field: "语言模型推理能力对齐"
keywords: ["Chain-of-Thought", "Instruction-Tuning", "Direct Preference Optimization", "Small Language Models", "Teacher-Student Alignment", "Reasoning Transfer"]
innovations: ["提出两阶段Self-refine Instruction-tuning框架：Instruction-tuning初始化+DPO自我精炼", "设计DPO_CoT自生成偏好机制，无需独立Reward Model即可优化CoT推理风格", "系统性对比in-family与out-family对齐，揭示跨族教师在低资源场景下的泛化优势"]
benchmarks: ["GSM8K", "MultiArith", "OBQA", "CSQA", "PIQA", "SIQA", "MATH", "MMLU"]
---

# 论文速读：Self-Refine-Instruction-Tuning-for-Aligning-Reasoning-in-Lan

## 一句话总结
论文提出Self-refine Instruction-tuning方法，通过"大模型示范→小模型微调→DPO自我优化"的两阶段流程，实现小语言模型(SLM)与大语言模型(LLM)之间Chain-of-Thought推理能力的对齐，并在常识与数学推理任务上显著超越仅靠Instruction-tuning的基线。

## 研究问题与动机
- 现有SLM对齐方法主要依赖LLM生成的CoT标注进行SFT，但标注仅提供单一推理路径，缺乏对多步推理的泛化性。
- 同一问题存在多种有效CoT解释，单纯SFT难以让小模型学到通用的推理能力。
- 传统Teacher-Student方法在in-family和out-family设置下均存在性能差距，且out-domain泛化能力不足。
- 需要一种既能初始化推理能力、又能让小模型自我改进（self-refine）的对齐机制。

## 核心贡献（创新点）
- **提出Self-refine Instruction-tuning两阶段框架**：先通过LLM生成的CoT演示进行Instruction-tuning初始化，再用DPO让SLM自主采样并偏好优化，与仅SFT方法形成本质区别。
- **引入DPO_CoT自生成偏好机制**：学生模型同时生成CoT答案与非CoT答案，以正确CoT为偏好正样本、非CoT答案为负样本，无需独立训练Reward Model。
- **系统性对比in-family与out-family对齐**：揭示同族教师（Llama→Llama）与跨族教师（GPT→Llama）在不同任务上的泛化差异，证明self-refine能缩小性能差距。
- **验证低资源场景下的可持续性**：在减少25%-75%训练数据时，Self-refine阶段仍能保证稳定性能提升。

## 方法详解
- **第一阶段Instruction-tuning**：教师LLM以zero-shot CoT提示生成演示数据，构成三元组$(i, q, a_i)$，其中$i$为指令、$q$为问题、$a_i$为CoT答案。损失函数为：
  $$\mathcal{L}_{\text{inst}}(\theta) = -\mathbb{E}_{(i,q,a_i)\sim D}\left[\sum_{t=1}^{L}\log \pi_\theta(w_t | s_t, i)\right]$$
  模型在状态$s_t$下同时条件于输入$q$和指令$i$，强制对齐指令遵循与推理生成。
- **第二阶段Self-refine（DPO）**：对学生模型$\pi_{\text{inst}}$用$x = i + q + \text{"Let's think step by step"}$提示，分别采样得到CoT答案$y_w$（偏好）与非CoT答案$y_l$（不偏好）。DPO损失为：
  $$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{inst}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{inst}}(y_l|x)}\right)\right]$$
  其中$\beta=0.1$为温度超参。通过最大化偏好答案与非偏好答案的对数概率差，推动学生向教师CoT风格对齐。
- **实验配置**：使用QLoRA进行Instruction-tuning（学习率$2\times10^{-5}$，4 epoch），DPO使用HuggingFace DPO Trainer（学习率$10^{-6}$，warm-up 100步，batch size 128，最大1000步）。

## 实验与结果
- **数据集**：常识推理（OBQA, CSQA, PIQA, SIQA）、数学推理（GSM8K, MultiArith），另测MATH与MMLU验证泛化性。
- **模型设置**：教师模型Llama2-70b、Mixtral-7x8、GPT-3.5；学生模型Llama2-7b、Llama2-13b、Mistral-7b。
- **最强结果**：Llama2-7b教师→学生、Self-refine Instruction-tuning后，GSM8K准确率76.9%（vs基线CoT 68.2%）、MultiArith 85.8%（vs 69.5%），PIQA达84.6%（vs 61.6%）。
- **out-family提升**：GPT-3.5作为教师时，Llama2-7b在GSM8K达81.9%，跨任务Cross Self-refine亦表现接近in-domain。
- **低资源验证**：25%训练数据下，Self-refine仍保持性能优势，证明方法在少样本场景有效。
- **结论**：Self-refine阶段在in-domain与out-domain任务上均显著优于纯Instruction-tuning，且能缩小教师-学生性能差距。

## 相关工作脉络
- **Zelikman et al. (2022) Self-SFT**：让模型自我生成推理路径并SFT，本文扩展至多教师协同与DPO优化。
- **Magister et al. (2023) Teaching Small LMs to Reason**：用PaLM/GPT-3.5演示SFT小模型，本文引入Instruction-tuning+Self-refine双阶段。
- **Wang et al. (2023d)**：IT+RL对齐，但需独立Reward Model；本文DPO无需额外训练奖励模型。
- **Ranaldi & Freitas (2024)**：先前纯Instruction-tuning工作，本文在此基础上增加Self-refine阶段填补性能差距。
- **Luong et al. (2024) REFT**：SFT+RL融合，本文聚焦CoT风格迁移与偏好自生成。
- **Uesato et al. (2022) 过程/结果奖励**：训练外部奖励器重排序，本文直接用DPO实现偏好优化。

## 局限性与未来方向
- 仅评估英语任务，未覆盖多语言场景。
- 教师模型的预训练数据差异未被建模，分析局限于自然语言输出层面。
- CoT推理链可能产生"看似合理但过程错误"的答案，模型过度自信风险未充分讨论。
- 未来方向：扩展至非英语语言、探索更强安全与OOD泛化机制。

## 研究启发与可借鉴点
- **Dual-phase对齐范式**：先SFT初始化再DPO精炼的设计可迁移至其他模型压缩与能力对齐任务。
- **Self-generated preference策略**：用同一模型生成正负样本无需外部标注，适合资源受限场景。
- **Cross-setting evaluation思路**：训练任务与评估任务不一致时的性能保持，为领域自适应提供新思路。
- **CoT质量影响分析**：Appendix J表明误导性的多步推理会损害DPO效果，提示数据质量控制的重要性。
- **低资源可扩展性**：25%数据仍有效，为少样本场景的推理对齐提供实用方案。

## 关键术语表
- **Chain-of-Thought (CoT)**：将复杂推理分解为多步中间步骤的输出形式，激发模型逐步思考。
- **Small Language Model (SLM)**：参数量较小的语言模型，推理成本更低但原生能力弱于LLM。
- **Large Language Model (LLM)**：大规模预训练语言模型，具备涌现推理能力。
- **Direct Preference Optimization (DPO)**：无需显式奖励模型，直接通过偏好对优化策略的RL方法。
- **Instruction-tuning**：面向特定任务的SFT变体，以指令-输入-输出三元组训练模型遵循指令。
- **In-family/Out-family alignment**：教师与学生来自同模型族（如Llama→Llama）或不同模型族（如GPT→Llama）的对齐设置。
- **Self-refine**：模型在初始化后通过偏好优化自主迭代改进自身输出的过程。
- **Cross Self-refine**：在不同任务上进行训练与优化的设置，用于评估泛化能力。

## 可复现要素
- **数据集**：OBQA、CSQA、PIQA、SIQA、GSM8K、MultiArith均公开于Hugging Face；MATH与MMLU亦为开源基准。
- **代码与权重**：论文声明代码与演示数据已随提交公开（具体链接见原文附录）。
- **关键超参**：QLoRA阶段学习率$2\times10^{-5}$、weight decay $10^{-4}$、cosine scheduler warm-up 0.03、4 epoch；DPO阶段学习率$10^{-6}$、$\beta=0.1$、warm-up 100步、batch size 128、max steps 1000。
- **训练硬件**：4×Nvidia RTX A6000（48GB VRAM）。
- **教师生成参数**：GPT-3.5 temperature=0.7，Llama2-70 temperature=0.1。
