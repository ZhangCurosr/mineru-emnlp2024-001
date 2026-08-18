---
title: "Alignment-Enhanced-Decoding-Defending-Jailbreaks-via-Token-L"
source: https://aclanthology.org/2024.emnlp-main.164.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:16:16"
field: "大语言模型安全与对齐"
keywords: ["jailbreak防御", "大模型安全", "对齐增强解码", "Competitive Index", "自适应解码", "LLM安全性"]
innovations: ["首次提出Competitive Index量化模型目标竞争风险", "无需训练的运行时token级概率分布自适应修正", "结合模型自我评估logits的动态对齐增强解码框架"]
benchmarks: ["AdvBench", "MMLU", "GMS8K", "Alpaca", "GCG", "Auto-DAN", "ICA", "Refusal_Suppression"]
---

# 论文速读：Alignment-Enhanced-Decoding-Defending-Jailbreaks-via-Token-Level-Adaptive-Refining-of-Probability-Distributions

## 一句话总结
本文提出了对齐增强解码（Alignment-Enhanced Decoding, AED），一种无需额外训练、通过在解码阶段自适应修正token概率分布来防御jailbreak攻击的方法；该方法引入Competitive Index量化目标竞争风险，并结合模型自我评估的post-alignment logits动态调整输出分布，在保持有用性的同时显著提升安全性。

## 研究问题与动机
1. **核心问题**：大语言模型（LLMs）虽经过对齐训练，但仍易受jailbreak攻击绕过安全限制，生成有害内容。
2. **现有方法不足**：当前防御手段主要分为扰动类（如PPL、Re-tokenization）和检测类（如Self-Defense），但均未触及jailbreak失败的底层原因。
3. **根本原因**：Wei et al. (2024) 指出对齐失败源于"Competing Objectives"——模型在helpfulness与harmlessness之间权衡时，面对jailbreak提示可能优先满足帮助性目标而忽视安全性。
4. **方法设计动机**：既然风险源于目标竞争，防御应在解码阶段直接干预token分布，而非仅在输入侧处理。

## 核心贡献（创新点）
1. **提出Competitive Index**：首次定义指标$I = S/S_t$量化模型预测下一token时目标竞争程度，$S$为top-p候选集大小，$I>1$表示高风险对抗场景。
2. **提出AED框架**：通过自适应解码在每个token生成步骤动态修正概率分布，无需任何额外训练即可增强安全对齐。
3. **设计基于自我评估的post-alignment logits**：利用已生成文本作为辅助输入推导模型自我评估得分，反映对齐后的token倾向。
4. **实验验证广泛有效性**：在5个主流开源模型（Llama2、Llama3、Vicuna、Guanaco、Gemma）和4种jailbreak攻击（GCG、Auto-Dan、ICA、Refusal_Suppression）上验证方法有效，同时在3个无害数据集上保持有用性。

## 方法详解

**整体流程**：对每个生成步$t$，AED首先计算原始logits $\mathbf{L}_{\text{model}}$和post-alignment logits $\mathbf{L}_{\text{post}}$，再通过Competitive Index自适应混合两者得到修正后logits $\mathbf{L}_{\text{AED}}$。

**关键公式与步骤**：

1. **Competitive Index计算**：
   - 基于top-p采样确定候选集 $\mathcal{P}_c$，候选集大小$S = |\mathcal{P}_c|$
   - $S_t$为无害样本中$S$的最大值（模型特定的阈值）
   - Competitive Index $I = S/S_t$，阈值$I_t = 1$

2. **Post-alignment logits推导**（自我评估）：
   - $\mathbf{L}_{\text{model}} = \text{LLM}(y_n | x_1, \cdots, x_m, y_1, \cdots, y_{n-1})$
   - $\mathbf{L}_{\text{post}} = \text{LLM}(y_n | y_1, \cdots, y_{n-1})$（前置"Assistant:"避免空输入）

3. **自适应混合解码**：
   - $I_{\text{model}} = f(\mathbf{L}_{\text{model}})/S_t$，$I_{\text{post}} = f(\mathbf{L}_{\text{post}})/S_t$
   - 调控系数 $c = \sigma(S_t \cdot (I_{\text{model}} - I_{\text{post}} - B_{\text{bias}}))$，$\sigma$为sigmoid函数
   - 修正logits $\mathbf{L}_{\text{AED}} = (1-c) \cdot \mathbf{L}_{\text{model}} + c \cdot \mathbf{L}_{\text{post}}$
   - 最终分布 $\mathbf{P}_{\text{AED}} = \text{softmax}(\mathbf{L}_{\text{AED}})$

**物理意义**：当遭遇jailbreak时$I_{\text{model}}$增大导致$c \to 1$，此时AED更依赖$\mathbf{L}_{\text{post}}$；由于对齐样本中安全候选的post-alignment logit更高，从而提升安全响应概率。

## 实验与结果

**实验设置**：
- **模型**：Llama2-7B-Chat-HF、Llama3-8B-Instruct、Vicuna-7B、Guanaco-7B、Gemma-1.1-7B-IT
- **攻击**：GCG、Auto-DAN、ICA（IC）、Refusal_Suppression
- **无害数据集**：MMLU、GMS8K、Alpaca（各90 prompt）
- **有害基准**：AdvBench

**主要结果**（Rejection Rate, RR）：

| 模型 | 攻击 | No Defense | PPL | Self-Defense | Re-tokenization | **AED** |
|------|------|-----------|-----|-------------|----------------|--------|
| Llama2 | GCG | 75.5% | 100% | 76.6% | 5.7% | **92.5%** |
| Llama2 | AutoDAN | 43.5% | 0% | 53.3% | 4.4% | **79.5%** |
| Llama2 | ICA | 100% | 0% | 100% | 52.2% | **100%** |
| Llama2 | Refusal_Sup | 54.0% | 0% | 90.0% | 6.7% | **91.0%** |
| Vicuna | GCG | 60.0% | 100% | 73.3% | 5.7% | **93.6%** |
| Vicuna | AutoDAN | 45.5% | 0% | 33.3% | 2.2% | **76.3%** |
| Llama3 | GCG | 73.3% | 100% | 82.2% | 1.1% | **85.0%** |
| Llama3 | AutoDAN | 74.0% | 0% | 71.1% | 2.2% | **90.0%** |
| Gemma | AutoDAN | 22.0% | 0% | 21.1% | 5.9% | **34.0%** |

- **最强结果**：Llama2在ICA上达到100%拒答率；Llama2在GCG上达92.5%，相比Base提升约17个百分点。
- **GPT-4评估**（Llama2）：AED在各攻击下均取得最低有害性评分（GCG: 2.06, AutoDAN: 2.47, ICA: 1.07, Refusal_Sup: 1.8）。

**有效性**：AED在全部5个模型和4种攻击上均显著优于基线。

## 相关工作脉络
1. **Wei et al. (2024)**：提出"Competing Objectives"框架解释对齐失败原因，是本文理论基础的直接来源。
2. **Jain et al. (2023)**：baseline PPL和Re-tokenization方法的提出者，属于扰动类防御。
3. **Phute et al. (2024)**：Self-Defense方法的提出者，通过让模型自我判断"harmful"与否进行防御，属于二分类检测类。
4. **Zou et al. (2023)**：GCG和AdvBench攻击的提出者，代表自动化对抗性jailbreak攻击的代表性工作。
5. **Liu et al. (2023a)**：Auto-DAN攻击的提出者，通过LLM自动生成隐蔽jailbreak提示。
6. **Xu et al. (2024) SafeDecoding**：另一项基于解码过程的防御工作，通过比较原始模型与微调模型的分布差异选词；本文指出与其方法本质不同（使用Competitive Index而非分布距离），但复现结果与原文有出入。

## 局限性与未来方向
1. **Competitive Index内部差异未解释**：论文未深入分析为何同一jailbreak攻击下$I$值存在巨大差异（部分可达阈值100倍），以及不同模型间$S_t$差异的原因。
2. **$S_t$依赖无害样本标定**：阈值$S_t$需从特定无害数据集（MMLU/GMS8K/Alpaca）中计算得出，可能不适用于领域外分布。
3. **仅在前30个token生效**：为控制计算开销，AED仅在生成前30个token时介入，对长文本的后半段无防护。
4. **潜在系统提示缓解效应**：Fig.5显示添加系统提示可显著降低$I$，表明部分攻击可能被标准prompt engineering缓解，方法通用性有待进一步验证。

## 研究启发与可借鉴点
1. **Competitive Index的思路可迁移至其他任务**：该指标量化了模型内部决策不确定性，可推广至多轮对话、agent交互等场景的安全评估。
2. **Self-evaluation logits的设计新颖**：利用已生成文本的前缀重新推导logits作为"对齐后"信号，无需额外训练，可借鉴到prompt-based防御或其他需要校准输出的场景。
3. **无需训练的运行时干预**：AED在推理阶段动态调整分布，不改变模型权重，这对部署受限的场景（如API服务）具有实用价值。
4. **与团队方向结合机会**：可将Competitive Index作为风险信号接入现有的LLM安全防护流水线，或与模型压缩、蒸馏方法结合以进一步降低计算开销。

## 关键术语表
- **Competitive Index (I)**：衡量模型在预测下一token时目标竞争程度的指标，$I = S/S_t$，$S$为top-p候选集大小，$I>1$表示高风险对抗场景。
- **Candidate Count (S)**：top-p采样下累积概率达到阈值$p_0$所需的最少token数量，反映候选分布的"竞争"程度。
- **Post-alignment Logits**：仅基于已生成文本（不含用户输入）推导的logits，表征模型对当前生成路径的"自我评估"。
- **Jailbreak Attack**：通过精心设计的提示绕过LLM安全限制，诱导其生成有害内容的攻击方式。
- **Rejection Rate (RR)**：防御方法的有效指标，$RR = 1 - ASR$，ASR为Attack Success Rate，RR越高防御效果越好。
- **Not Rejection Rate (NRR)**：衡量模型错误拒绝无害输入的比率，NRR越低说明有用性保持越好。

## 可复现要素
- **代码开源**：https://github.com/GIGABaozi/AED
- **数据集**：GCG、Auto-DAN、ICA、Refusal_Suppression、AdvBench、MMLU、GMS8K、Alpaca均为公开数据集
- **模型权重**：使用5个开源模型（Llama2-7B-Chat-HF、Llama3-8B-Instruct、Vicuna-7B、Guanaco-7B、Gemma-1.1-7B-IT）
- **关键超参**：top-p $p_0$（实验用[0.1, 0.2, 0.4, 0.8, 0.9]均有效）、$B_{\text{bias}} = 1 \times S_t$（最优）、生成步数$N=30$
- **PPL阈值**：各模型在不同数据集上需单独标定（见Tab.1）
