---
title: "Instruction-Pre-Training-Language-Models-are-Supervised-Mult"
source: https://aclanthology.org/2024.emnlp-main.148.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:21:50"
field: "语言模型预训练"
keywords: ["instruction pre-training", "supervised multitask learning", "synthetic data generation", "language model pre-training", "instruction synthesizer", "few-shot learning"]
innovations: ["提出 Instruction Pre-Training 框架，将大规模合成指令-响应对嵌入预训练语料实现监督式多任务预训练", "构建基于 7B 开源模型的指令合成器，通过多轮推理低成本生成 2 亿条多样化指令对", "证明预训练阶段注入监督信号可显著提升数据效率并加速后续指令微调"]
benchmarks: ["MMLU", "ARC-e/c", "BoolQ", "SIQA", "WinoGrande", "PIQA", "OBQA", "HellaSwag", "PubMedQA", "ConvFinQA", "FiQA SA"]
---

# 论文速读：Instruction-Pre-Training-Language-Models-are-Supervised-Mult

## 一句话总结
本文提出了 **Instruction Pre-Training**，通过将原始语料与基于开源模型高效合成的 2 亿条指令-响应对（覆盖 40+ 任务类别）相结合，实现大规模**监督式多任务预训练**，在从零预训练和领域自适应持续预训练中均显著提升了 LM 性能与数据效率。

## 研究问题与动机
- **监督式多任务学习难以规模化**：GPT-2 等开创的无监督因果语言建模已成为 LM 预训练标准范式（Vanilla Pre-Training），但 post-training 阶段指令微调证明监督式多任务学习仍有巨大潜力，却因数据生成成本难以扩展。
- **现有合成数据方法依赖昂贵闭源大模型**：Self-Instruct、Mammoth2 等工作使用 GPT-4 等大模型生成合成数据，成本高昂；需要更低成本的可扩展方案。
- **预训练阶段缺乏高质量监督信号**：当前 pre-training 以无监督主，supervised signal 仅在 post-training 阶段注入，存在训练目标对齐断层问题。

## 核心贡献（创新点）
- **提出 Instruction Pre-Training 框架**：将大规模原始语料与合成指令-响应对混合后进行语言建模预训练，把监督信号前移到 pre-training 阶段；与既有工作的本质区别在于从"post-training 微调阶段做指令学习"转向"pre-training 阶段就注入多任务监督信号"。
- **构建基于开源模型的指令合成器**：以 Mistral-7B-v0.1 为基座进行多任务微调，单 A100-80GB 约 1 天即可合成 1B token 原始语料的指令对，远低于使用 GPT-4 等闭源模型的成本；与既有工作（Self-Instruct、Genie 等依赖强模型蒸馏）的本质区别在于使用低成本开源模型、直接从原始语料生成而非从已有数据集蒸馏。
- **多轮推理合成少样本示例**：通过逐轮叠加历史文本与指令对，自动构建 few-shot 预训练示例；与既有方法（单次生成 or 规则构造）的区别在于无需人工规则，synthesizer 通过迭代学习示例模式。
- **系统性实验验证**：在通用预训练（500M/1.3B 从零训练）和领域自适应持续预训练（Llama3-8B → 金融/生物医学）两场景下验证，500M 模型仅用 100B tokens 达到 Pythia-1B（300B tokens）水平，Llama3-8B 在领域任务上媲美甚至超越 Llama3-70B。

## 方法详解
- **指令合成器（Instruction Synthesizer）**：
  - **数据收集**：采样并重组多种基于上下文的 task completion 数据集（百科、社交媒体、学术测试等），每样本包含一段 raw text 及其对应的多条 instruction-response 对；覆盖 free-form completion、multiple-choice、含 CoT 的两种格式，确保多样化。
  - **微调策略**：将同数据集的多条 one-shot 示例（raw text + instruction-response pairs）拼接成一个序列输入模型，**仅对 instruction-response pairs 部分计算 loss**，引导模型聚焦生成任务输出；学习率 5e-6、batch size 16384 tokens、max sequence length 4096。
  - **模板设计**：使用 `<CON>` 包裹 raw text，`<QUE>`/`<ANS>`/`</END>` 分隔指令与响应，便于推理后直接提取。
  - **多轮推理（Multi-round Inference）**：每轮将上一轮的文本与生成的指令对前置到当前轮输入，使 synthesizer 基于已有模式生成新指令对，逐步构建 M-shot 示例。
- **LM 预训练（Instruction Pre-Training）**：
  - 使用 Longpre et al. (2023) 模板多样化指令格式，将 M 轮合成的 raw text 与 instruction-response pairs 拼接为 M-shot 示例。
  - 预训练目标与 Vanilla Pre-Training 一致：next-token prediction，对所有 token 计算 loss。
  - **从零预训练**：仅转换 1/5 原始语料为指令增强语料（40M 条），剩余部分保持原始；混合 synthesizer 微调数据（重复 4 次）增强多样性。
  - **领域自适应持续预训练**：转换全部领域语料，并与 general instructions 混合以增强 prompt 能力。

## 实验与结果
- **数据集**：
  - 从零预训练：RefinedWeb 子集（200M 条文本，约 100B tokens），合成 200M 指令-响应对（约 10B tokens），覆盖 49 个任务类别。
  - 领域持续预训练：PubMed Abstracts（生物医学）、金融新闻语料。
- **评估基线**：Vanilla Pre-Training（同等 token 数）、Mix PT（原始语料+微调数据混合）、Llama3-70B（参考）。
- **主要结果**：
  - **500M 模型（100B tokens）**：MMLU 25.3，在多个通用 benchmark（ARC-e 54.8、BoolQ 62.0、SIQA 47.2 等）上显著优于 Vanilla PT 和 Mix PT；达到 Pythia-1B（300B tokens）的平均性能水平（46.6 vs 47.1）。
  - **1.3B 模型（100B tokens）**：达到 BLOOM-3B（341B tokens）水平（49.7 vs 50.1）。
  - **指令微调衔接**：Instruct PT 预训练模型在后续 instruction tuning 过程中更快提升、更高收敛（Figure 4），减少所需 tuning 步数。
  - **领域持续预训练（Llama3-8B）**：生物医学平均 61.3 vs Vanilla PT 58.4、Llama3-70B 63.9；金融平均 74.7 vs Vanilla PT 72.0、Llama3-70B 71.9，**超越 Llama3-70B**。
- **消融实验**（Table 4）：去除领域语料（w/o Corpora）性能下降；替换为规则方法（Rule-based）效果有限；单轮合成（1-shot）降低 prompting 能力。
- **合成数据质量**（Table 6）：整体响应准确率 77.5%，上下文相关性 92.9%，49 类任务；生物医学准确率 86.2%/相关性 99.4%；金融准确率 69.8%/相关性 85.8%。

## 相关工作脉络
- **Self-Instruct / WizardLM**：在 post-training 阶段利用强模型或规则生成指令数据进行微调；本文与之互补，将监督信号前置到 pre-training 阶段，且使用开源模型而非闭源强模型。
- **AdaptLLM（Cheng et al., 2023）**：使用规则方法从领域语料构建指令增强数据；本文证明基于学习的 synthesizer 能产生更高多样性的任务，显著提升性能。
- **SlimPajama / Doremi**：通过数据选择与混合优化预训练数据分布；本文探索正交方向——在原始语料上叠加大规模监督信号而非替代原始语料。
- **Mammoth2 / Genie**：使用闭源大模型（GPT-4）迭代合成指令数据；本文以 7B 开源模型为基础，大幅降低成本，可规模化扩展到更大语料。
- **In-Context Pretraining（Shi et al., 2023）**：在文档边界外注入示例；本文将其思想前移至 pre-training 阶段，直接在训练语料中嵌入合成指令对。

## 局限性与未来方向
- **合成数据准确率约 70%，存在幻觉风险**：低质量数据可能误导预训练模型；未来可引入 post-verification 技术过滤低质量样本。
- **预训练规模限于百亿级 tokens**：对比 Llama 等万亿 token 训练规模，需进一步探索合成数据的 scaling law 及数量-质量平衡。
- **单轮/固定轮次合成**：未来可探索与迭代增强方法（如 LLM2LLM、Mammoth2）结合，进一步提升合成质量。
- **评估集中于通用与两个垂直领域**：需扩展到更多领域以验证泛化性。

## 研究启发与可借鉴点
- **多轮推理构建少样本示例**：通过逐轮累积历史输出构建 few-shot 数据，无需人工设计示例模式，可迁移到任何需要合成 few-shot 数据的场景。
- **Loss 仅计算在生成目标上**：微调 synthesizer 时只对 instruction-response pairs 计算 loss，有效引导模型聚焦任务输出，避免被 raw text 稀释梯度。
- **预训练与微调阶段的任务对齐**：Instruction Pre-Training 使 pre-training 阶段即接触多任务指令格式，显著加速后续 instruction tuning，为"pre-training 即对齐"提供了实证。
- **低成本开源模型合成大规模数据**：证明 7B 开源模型经多样数据微调即可胜任大规模数据合成，为低预算团队提供可行路径。
- **与团队方向结合机会**：可探索将本方法应用于中文语料或特定垂直领域（如医疗、法律）的预训练，结合领域知识约束进一步提升合成质量。

## 关键术语表
- **Instruction Pre-Training**：将合成指令-响应对嵌入原始预训练语料，在预训练阶段即引入监督式多任务学习信号的框架。
- **Instruction Synthesizer**：基于开源 LM 微调而成的模型，给定原始文本自动生成多条多样化 instruction-response 对。
- **Instruction-Augmented Corpora**：原始文本与合成指令-响应对拼接后构成的增强版预训练语料。
- **Multi-round Inference**：指令合成器的推理策略，逐轮将历史文本与已生成指令对前置，逐步构建 few-shot 示例。
- **Vanilla Pre-Training**：传统仅使用原始无标注语料进行因果语言建模的预训练方式。
- **One-shot / Few-shot Example**：前者指单条 raw text 配一条指令对；后者指多条示例拼接，供 LM 预训练时学习任务模式。
- **Domain-Adaptive Continual Pre-Training**：在通用预训练模型基础上，使用特定领域语料继续进行预训练以适应垂直场景。
- **Contamination（数据污染）**：评估集样本出现在训练数据中的现象，本文合成数据引入的污染极低（几乎为 0）。

## 可复现要素
- **代码/模型/数据**：已开源，详见 https://github.com/microsoft/LMOps；HuggingFace 模型：https://huggingface.co/instruction-pretrain
- **数据集**：RefinedWeb（公开）、PubMed Abstracts（公开）、金融新闻语料（公开）；合成数据随项目发布。
- **关键超参**：Synthesizer 微调——Mistral-7B-v0.1 基座，batch size 16384 tokens，max seq length 4096，学习率 5e-6，cosine scheduler，Adam(0.9,0.95)，warmup 1000 steps；预训练——500M/1.3B 模型 max seq length 2048，学习率 3e-4/2e-4，batch size 0.5M/1M tokens。
- **硬件**：Synthesizer 微调使用 4×A100-80GB；预训练使用 8×A100-80GB。
