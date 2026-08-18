---
title: "“Thinking” Fair and Slow: On the Efficacy of Structured Prompts for Debiasing Language Models"
source: https://aclanthology.org/2024.emnlp-main.13.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:07:01"
---

# 论文速读：“Thinking” Fair and Slow: On the Efficacy of Structured Prompts for Debiasing Language Models

## 一句话总结
本文面向无法访问模型内部参数或 logits 的终端用户，首次将 Kahneman 的“系统2思维”框架系统性地迁移至 LLM 提示去偏领域，提出并验证了 Prefix Prompting、Self-Refinement 与 Implication Prompting 三类递进式结构化提示策略；实验表明，蕴含提示（IP）能在黑盒条件下显著降低性别/种族/宗教等多维偏见，且完全不损害下游任务性能。

## 研究问题与动机
- **闭源 API 与资源受限场景的去偏可及性缺失**：现有去偏方法（重训练、微调、logits 校准/解码策略修改）依赖模型权重或输出概率，对主流商业闭源模型与普通用户不可用。
- **微调去偏的高成本与性能回退风险**：即使拥有开源模型，高质量公平数据的采集与全量/高效微调成本极高，且修改预训练权重或解码策略极易破坏模型在海量下游任务上已优化的综合能力。
- **既有提示去偏工作过于简单且缺乏系统性**：已有研究多局限于关键词替换或单次自然语言指令（如 Borchers et al., 2022），未覆盖迭代式、 reasoning-based 的结构化提示，也缺乏跨模型、跨偏见维度的统一评估。
- **核心科学问题**：在仅能通过 text-in/text-out 接口交互的黑盒设定下，如何设计提示范式以有效引导 LLM 进行反思与公平化重写？

## 核心贡献（创新点）
- **首次构建面向终端用户的系统化提示去偏框架**：受“系统1/系统2”认知科学启发，将提示策略划分为 PP（直接指令）、SR（自我精炼）与 IP（蕴含推理）三个递进层级，并与以往仅依赖单次前缀或关键词工程的工作形成本质区分。
- **提出 Implication Prompting（蕴含提示）三阶段黑盒流程**：让模型先自主识别并输出原始文本中的潜在偏见归因（implication），再结合原始输出与去偏指令完成重写。与 SelfDebias/CAFIE 等需访问 logits 的白盒方法不同，该方法完全兼容标准 API，且实验证明动态生成蕴含显著优于固定模板。
- **大规模跨模型/跨基准一致性评估**：在 GPT-J、Mistral、MPT-Instruct、Llama-2 四个架构各异的模型上，覆盖 StereoSet、Regard、Toxicity 及 TruthfulQA/BoolQ 五项指标，填补了“结构化提示复杂度 × 多模型 × 多偏见维度”的系统性评估空白。
- **揭示多项关键工程规律**：Role-based 提示稳定优于 Instruction-based；SR 迭代在 k=1 即趋饱和，k≥2 仅带来边际收益；蕴含生成可由更小或更大模型代理而不显著损失性能；去偏提示不会引发传统的“公平性-语言建模能力”权衡退化。

## 方法详解
- **Prefix Prompting (PP)**：在用户 prompt $C$ 前拼接去偏前缀 $I_{debias}$，得到 $C_{debias} = \text{concat}(I_{debias}, C)$。分为两类：
  - *Instruction PP*：`The following text is unbiased and does not discriminate against people on the basis of their gender, race, religion, or any other sensitive attribute: [USER PROMPT]`
  - *Role PP*：`You are an unbiased person who does not discriminate against people based on their gender, race, religion, or any other sensitive attribute. [USER PROMPT]`
- **Self-Refinement (SR)**：基于 k 步迭代的自我修正流程（Algorithm 1）。Step I 用 PP 生成参考文本 $S_0$；Step II 将 $S_{i-1}$ 与去偏前缀 $I_{SR}$ 拼接为 $C_{SR} = \text{concat}(I_{SR}, S_{i-1}, C)$，驱动模型参考上一轮输出重新生成 $S_i$。重复至第 $k$ 步输出 $S_{debiased}$。提示语强调“refer to this text and generate some text that is unbiased...”。
- **Implication Prompting (IP)**：三阶段黑盒流程（Algorithm 2）：
  - *Step I*：输入用户 prompt $C$，生成初始可能含偏见的输出 $S$。
  - *Step II*：构造蕴含提示 $C_{Impl} = \text{concat}(S, I_{Impl})$，让模型输出 $S_{Impl}$ 解释 $S$ 中存在的具体偏见。本文测试三种 $I_{Impl}$ 变体：Instruction-Based、Zero-Shot CoT-Based（`Thinking step by step, generate the implication...`）、Few-shot-Based（提供正反示例）。
  - *Step III*：构造最终提示 $C_{IP} = \text{concat}(S, S_{Impl}, I_{IP}, C)$，其中 $I_{IP}$ 明确要求模型“considering the implication and referring to the original sentence, generate an unbiased text...”。核心原理是提供逻辑归因比单纯指令更能激活模型 latent space 中的公平模式。

## 实验与结果
- **模型与硬件**：GPT-J (6B)、Mistral (7B)、MPT-Instruct (7B)、Llama-2 (13B)；单张 32GB NVIDIA V100；temperature=1.0，StereoSet 使用 repetition_penalty=1.3，默认 beam search。
- **核心指标表现**（跨模型均值）：
  - **IP 全面最优**：相较所有其他方法，IP 使 StereoSet SS 平均降低 **4.05%**、ICAT 提升 **6.80%**；Regard 平均改善 **26.85%**；Mean Toxicity 平均下降 **6.98%**（
