---
title: "Towards-Tool-Use-Alignment-of-Large-Language-Models"
source: https://aclanthology.org/2024.emnlp-main.82.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:34:25"
field: "大语言模型对齐与安全"
keywords: ["Tool Alignment", "Helpfulness", "Harmlessness", "Autonomy", "DPO", "SFT", "Tool Use"]
innovations: ["提出H2A原则统一对齐有用性、无害性与自主性", "构建ToolAlign多维权一直齐数据集", "验证SFT+DPO两阶段训练在工具对齐中的有效性"]
benchmarks: ["ToolAlign", "ToolSword", "MetaTool", "ToolBench"]
---

# 论文速读：Towards-Tool-Use-Alignment-of-Large-Language-Models

## 一句话总结
论文提出H2A原则（有用性Helpfulness、无害性Harmlessness、自主性Autonomy），构建ToolAlign数据集并通过SFT+DPO训练，使LLM在工具使用中实现能力、安全与效率的统一对齐。

## 研究问题与动机
1. **现有工作偏重有用性**：ToolBench等数据集仅关注工具调用能力，忽视安全性与自主性，导致模型在工具场景中可能输出有害内容或滥用工具。
2. **有害指令风险**：LLM可能被诱导执行隐私窃取、恶意代码生成等危险操作，需具备识别并拒绝有害指令的能力。
3. **工具响应安全性**：外部工具可能遭受攻击或被劫持，返回钓鱼链接、恶意脚本等不安全内容，LLM需能识别并拒绝此类响应。
4. **工具调用成本**：对于可直接回答的问题（如常识问答），盲目调用工具会增加时间与金钱成本，LLM应具备自主判断能力。

## 核心贡献（创新点）
1. **提出H2A对齐原则**：首次系统化定义工具使用场景下LLM应遵循的三项原则（有用性、无害性、自主性），填补多维权一直齐空白。
2. **构建ToolAlign数据集**：基于ToolBench扩展，包含46k指令微调数据（40k有用性+2.8k无害性+3.9k自主性）和10k偏好数据，兼顾三维能力。
3. **验证SFT→DPO两阶段训练有效性**：证明SFT负责建立无害性与自主性基础，DPO进一步优化有用性，两阶段缺一不可。
4. **对齐开源模型性能接近GPT-4**：AlignToolLLaMA-DPO在有害指令拒绝率（98.7%）和自主性（100%）上超越GPT-4（85.6%和11.0%）。

## 方法详解
**H2A原则框架**：
- **Helpfulness**：准确理解用户指令，调用外部工具并综合工具响应提供信息。
- **Harmlessness**：拒绝不安全用户指令，识别并拒绝对不安全工具响应。
- **Autonomy**：对于可直接回答的问题，不调用工具直接响应以节省成本。

**ToolAlign数据集构建**：
- **有用性数据**：从ToolBench采样40k指令-响应对。
- **无害性数据**：
  - 有害指令：从ToolBench改写1k条（添加隐私/违法内容）+ 从ARTD采样1k条并重写格式，共2.8k条。
  - 有害工具响应：四类（明显有害内容、钓鱼网站、攻击附件、敏感信息请求），改写841条工具响应。
- **自主性数据**：从Alpaca采样3.9k条并重写为ToolBench格式。
- **偏好数据**：10k指令，每指令采两响应（ChatGPT vs ToolLLaMA/AlignToolLLaMA-SFT），由ChatGPT评估生成偏好对。

**训练流程**：
1. **SFT**：ToolLLaMA在ToolAlign指令数据上训练2 epoch，batch size=64，lr=5e-5，warmup=4%。
2. **DPO**：AlignToolLLaMA-SFT在偏好数据上训练1 epoch，batch size=8，lr=1e-6，β=0.05。

## 实验与结果
**评估基准**：ToolAlign测试集（有用性I1-I/C/T、有害指令HI、有害工具响应HTR、自主性AU）。

**主要结果**：
- **有用性**：AlignToolLLaMA-DPO平均Pass Rate达49.8%，显著高于SFT的27.3%，接近GPT-4的57.2%。
- **无害性**：AlignToolLLaMA-DPO在HI拒绝率98.7%（SFT 96.4%），HTR拒绝率100%（SFT 100%），ToolLLaMA均为0%。
- **自主性**：AlignToolLLaMA-DPO/SFT均达100%直接响应率，ToolLLaMA仅22%。
- **最强对比**：AlignToolLLaMA-DPO在自主性（100% vs GPT-4的11.0%）和无害性（HI 98.7% vs GPT-4的85.6%）上超越GPT-4。

**泛化验证**：在ToolSword和MetaTool上，AlignToolLLaMA-DPO在Jailbreak攻击拒绝率（87.1% vs GPT-4的89.0%）和自主性（MetaTool 98.0% vs GPT-4的28.0%）表现优异。

## 相关工作脉络
1. **ToolLLaMA (Qin et al., 2023)**：专注有用性工具学习，未涉及安全与自主性对齐，本文在其基础上扩展。
2. **ToolSword (Ye et al., 2024)**：仅评估工具使用安全性，无训练数据与方法，本文填补对齐方法空白。
3. **MetaTool (Huang et al., 2023)**：仅关注自主性评估，本文统一评估三维能力。
4. **Safety-Tuned LLaMA (Bianchi et al., 2023)**：通用指令安全对齐，在工具场景中效果有限（Qwen2-7B-Instruct在HTR拒绝率为0%）。
5. **DPO (Rafailov et al., 2024)**：本文采用DPO优化偏好，验证其在工具对齐中的有效性。

## 局限性与未来方向
1. **单轮对话限制**：未充分验证多轮交互场景下的工具调用与安全性。
2. **上下文长度限制**：多轮对话需维护长上下文和历史交互记录，本文未涉及。
3. **人类价值复杂性**：H2A原则仍需更深入理解人类价值观的细微差异。
4. **未来方向**：扩展到多轮对话场景，增强模型在真实复杂交互中的适用性。

## 研究启发与可借鉴点
1. **多维权一直齐框架**：H2A原则可作为其他场景（如Agent、RAG）对齐的参考模板。
2. **SFT+DPO两阶段训练**：证明基础能力通过SFT建立，偏好优化通过DPO进一步提升，该范式可迁移。
3. **格式一致性技巧**：改写ARTD/Alpaca指令以匹配ToolBench格式，避免模型学习数据集shortcut，值得借鉴。
4. **偏好数据构建策略**：利用强模型（ChatGPT）作为参考生成偏好对，高效且可扩展。

## 关键术语表
**H2A**：Helpfulness, Harmlessness, Autonomy的缩写，工具使用对齐三原则。
**ToolAlign**：本文构建的数据集，包含指令微调数据（46k）和偏好数据（10k）。
**SFT**：Supervised Fine-Tuning，监督微调，用于建立基础能力。
**DPO**：Direct Preference Optimization，直接偏好优化，用于进一步提升性能。
**Pass Rate (PR)**：评估有用性的指标，衡量响应是否完成任务。
**Refusal Response Rate (3R)**：评估无害性的指标，衡量模型拒绝有害内容的比例。
**Direct Response Rate (DR2)**：评估自主性的指标，衡量模型不调用工具直接回答的比例。
**ToolBench**：基准数据集，包含3000+真实工具和126k指令，本文有用性数据来源。

## 可复现要素
- **数据集**：ToolAlign已开源（https://github.com/zhiyuanc2001/ToolAlign）
- **代码**：已开源
- **基线模型**：ToolLLaMA（开源）、LLaMA-2-Chat-7B、Qwen2-7B-Instruct
- **关键超参**：SFT学习率5e-5，batch size 64，2 epochs；DPO学习率1e-6，batch size 8，1 epoch，β=0.05
- **硬件**：4x Nvidia A100 40GB，bfloat16精度
