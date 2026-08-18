---
title: "Unsupervised-Human-Preference-Learning"
source: https://aclanthology.org/2024.emnlp-main.200.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:55:00"
---

# 论文速读：Unsupervised Human Preference Learning

## 一句话总结
本文提出了一种基于小型“偏好代理”（Preference Agent）的无监督个性化方法，通过让轻量级模型生成自然语言偏好规则来引导大型预训练 LLM，在无需微调大模型权重的前提下，以极低的数据与计算成本实现高度贴合个体用户风格的内容生成。

## 研究问题与动机
- **核心问题**：现有 LLM 虽具备强大的通用推理与知识能力，但缺乏对个体用户偏好的感知，难以稳定生成符合特定用户风格与语境的内容。
- **ICL 不足**：上下文学习（In-Context Learning）在个人数据量极少时，难以捕捉复杂、微妙且可能自相矛盾的用户偏好细节，泛化能力受限。
- **PEFT/微调不足**：参数高效微调（如 QLoRA）在极少量个人样本上容易产生过拟合或分布偏移，且直接修改大模型权重的计算与部署成本过高。
- **动机**：如何在个人数据稀缺、算力受限（仅限消费级设备）的现实约束下，以数据与计算高效的方式将强大的开源/闭源 LLM 对齐到个体用户的独特偏好。

## 核心贡献（创新点）
- **偏好代理架构**：设计了一个轻量级“方向盘”模型 $M_S$，专门学习将用户输入映射为自然语言偏好规则，与大模型 $M_L$ 解耦，避免直接修改大模型参数。
- **无监督隐式偏好蒸馏**：利用 $M_L$ 自身的零样本输出与 Ground Truth 作差，自动提取风格与偏好修正规则，无需人工标注偏好标签即可完成代理训练。
- **显著的性能提升**：在 GPT-4o 与人工双轨评测中，该方法相比 Few-shot、Naive Fine-tuning 等基线胜率大幅提升，部分场景最高可达 80% 以上。
- **开源三个大规模偏好数据集**：构建并公开了 Enron 邮件、New Yorker 文章与 LAMP 3U 商品评论的意图标注版本，填补了个性化文本生成的评测资源空白。

## 方法详解
- **双模型解耦设计**：小模型 $M_S$（Llama-3-8B-Instruct）作为偏好代理负责“理解与表达偏好”；大模型 $M_L$（Llama-3-70B-Instruct / Claude 3.5 Sonnet / Gemini 1.5 Pro）作为内容生成器负责“执行任务”。
- **训练流水线**：
  1. **零样本基线生成**：$M_L$ 对训练集输入 $\mathbf{X}=(\mathbf{u}, \mathbf{m})$ 生成无偏好内容 $\mathbf{Y}_z = M_L(\mathbf{X})$。
  2. **偏好规则提取**：$M_L$ 对比 $\mathbf{Y}_z$ 与 Ground Truth $\mathbf{G}$，识别差异并生成自然语言规则 $\mathbf{P} = M_L(\mathbf{Y}_z, \mathbf{G})$。
  3. **代理微调**：使用 QLoRA（16GB 显存可运行）微调 $M_S$，学习映射 $\mathbf{X} \rightarrow \mathbf{P}$，得到偏好代理 $M_A$。
- **推理对齐**：对测试输入 $\mathbf{x}$，$M_A$ 生成规则 $\mathbf{p} = M_A(\mathbf{x})$，随后 $M_L$ 结合原始输入与规则输出个性化结果 $y_a = M_L(\mathbf{x}, \mathbf{p})$。
- **规则生成策略对比**：蒸馏式（$R_3$，依赖基线差值，效果最佳）、直接式（$R_1$，仅凭真值生成）、加入 Thinking Tokens 的审慎提示（$R_2$，引入中间推理链）。
- **评估公式**：在测试集 $\mathcal{T}$ 上计算 $\operatorname{Score}(\mathcal{T}) = \sum \operatorname{Eval}(y_a^{(i)}, y_z^{(i)} | \mathbf{x}^{(i)})$，以正分偏向对齐输出、负分偏向基线；主指标为 Win Rate。

## 实验与结果
- **数据集**：Enron Email Corpus（短文本，40,240 封邮件，191 发送者）、New Yorker（长文创作，1,525 篇文章，401 位作者）、LAMP 3U Amazon Reviews（中长度评论，22,500 条，15 用户）。
- **基线设置**：Small/Large Zero-shot、Few-shot、Naive Finetune（同 QLoRA 配置直接拟合输入输出）、No Baseline Agent（无零样本差值监督）。
- **核心数值**（Table 2，综合 Win Rate）：
  - GPT-4o 评测：Large Baseline Preference Agent 达 **84.1%**，显著高于 Few-shot（62.0%）、Naive Finetune（83.9%）、No Baseline（63.4%）。
  - 人工评测：Preference Agent
