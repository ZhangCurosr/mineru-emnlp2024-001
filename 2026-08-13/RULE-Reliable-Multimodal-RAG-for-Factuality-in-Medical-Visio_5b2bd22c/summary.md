---
title: "RULE-Reliable-Multimodal-RAG-for-Factuality-in-Medical-Visio"
source: https://aclanthology.org/2024.emnlp-main.62.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:16:44"
---

# 论文速读：RULE-Reliable-Multimodal-RAG-for-Factuality-in-Medical-Visio

## 一句话总结
提出RULE框架，通过统计校准检索上下文数量以控制事实性风险，并结合针对“检索过度依赖”现象自动构造的偏好数据对Med-LVLM进行DPO微调，在医学VQA与报告生成任务上显著提升了模型的事实准确性。

## 研究问题与动机
1. **Med-LVLMs事实性脆弱**：现有医学大视觉语言模型在复杂病例解读中易产生幻觉，生成内容与医学事实相悖，直接影响临床决策安全。
2. **RAG引入的检索数量困境**：直接套用固定Top-K检索，过少时无法覆盖细粒度医学特征，过多时冗余/错误上下文会干扰模型生成。
3. **检索过度依赖（Over-Reliance）**：模型可能原本凭自身知识能答对，但引入RAG后反而被错误/无关检索带偏，导致正确回答变为错误。
4. **现有解码纠偏方法稳定性不足**：Greedy/Beam/DoLa/OPERA/VCD等Logits操纵方法在不同数据集上Precision-Recall波动剧烈，缺乏可证明的风险控制保障。

## 核心贡献（创新点）
1. **提出RULE端到端可信RAG框架**：首次将RAG系统性引入Med-LVLMs事实性增强，同时解决“检索数量难定”与“模型过度依赖”两个核心痛点，区别于以往仅关注检索精度或仅做解码后处理的工作。
2. **基于Learn-then-Test框架的事实性风险控制（FRC）**：通过计算不同$k$值下的事实性风险概率边界，并结合FWER控制程序（如Bonferroni校正）自动筛选可接受的$k$集合，为检索数量提供严格统计保证，区别于经验阈值或网格搜索。
3. **知识平衡偏好微调（KBPT）**：创新性地以“无检索答对、有检索答错”作为厌恶样本自动构造偏好数据集，利用DPO损失微调模型使其在生成时动态平衡内在知识与外部检索，区别于依赖人工标注或通用指令数据的偏好对齐方法。
4. **广泛且显著的性能提升**：在三个跨模态/跨科室医学基准（VQA与报告生成）上达到SOTA，平均事实准确率提升47.4%，且兼容不同代际骨干模型（LLaVA-Med-1.0/1.5）。

## 方法详解
- **多模态对比检索模块**：采用ResNet-50作为视觉编码器、BioClinicalBERT作为文本编码器，在医疗图文对上使用对比学习损失（Eq. 2）微调，最大化同图-报告对相似度、最小化跨对相似度。推理时提取目标图像特征，检索Top-K相似报告作为外部知识注入生成Prompt。
- **事实性风险控制（FRC）**：对候选检索数$k \in C_K$，计算经验事实性风险 $FR(k) = 1 - \text{ACC}(\mathcal{M}(x, (q, T_k)))$。分别计算基于KL散度的概率 $p_{k1} = \exp(-n h_1(FR(k) \wedge \alpha, \alpha))$ 与基于二项分布的概率 $p_{k2} = e \cdot \mathbb{P}(\text{Bin}(n, \alpha) \le \lceil n FR(k) \rceil)$，取 $p_k = \min(p_{k1}, p_{k2})$。随后应用Bonferroni等FWER控制程序，若 $p_k \le \delta / |C_K|$ 则将$k$纳入可接受集合 $\hat{\Lambda}$。Proposition 1证明：在概率至少 $1-\delta$ 下，$\sup_{k \in \hat{\Lambda}} FR(k) \le \alpha$ 成立。
- **知识平衡偏好微调（KBPT）**：从独立数据集中采样，识别满足 $a_b = y$（无检索正确）且 $a_f \neq y$（有检索错误）的样本，构造偏好对 $(y_w, y_l) = (y, a_f)$。基于DPO损失（Eq. 4）在该偏好数据集 $\mathcal{D}_o$ 上微调Med-LVLM：
  $\mathcal{L}_{kbpt} = -\mathbb{E}[\log \sigma(\alpha \log \frac{\pi_\theta(y_w|x)}{\pi_o(y_w|x)} - \alpha \log \frac{\pi_\theta(y_l|x)}{\pi_o(y_l|x)})]$，迫使模型降低对不可靠检索的权重，强化对自身参数的信任与调用。

## 实验与结果
- **数据集与任务**：IU-Xray、Harvard-FairVLMed（眼底/眼科）、M
