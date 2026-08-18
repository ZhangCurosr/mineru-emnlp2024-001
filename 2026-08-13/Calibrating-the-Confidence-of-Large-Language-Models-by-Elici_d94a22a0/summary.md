---
title: "Calibrating-the-Confidence-of-Large-Language-Models-by-Elici"
source: https://aclanthology.org/2024.emnlp-main.173.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:30:00"
---

# 论文速读：Calibrating-the-Confidence-of-Large-Language-Models-by-Elici

## 一句话总结
针对RLHF对齐后LLM普遍存在的“过度自信”问题，本文提出一种即插即用的置信度校准方法UF Calibration，将模型置信度解耦为“对问题的不确定性”与“对生成答案的忠实度”，无需访问内部logits即可在黑盒与白盒模型上实现优异且鲁棒的校准效果，并配套提出IPR与CE两个多维评估指标。

## 研究问题与动机
- RLHF等对齐技术虽提升了模型有用性与安全性，但优化目标偏向指令遵循与人类偏好，导致模型输出概率与答案正确率之间严重失配（过自信）。
- 现有Logit-based方法（如Temperature-Tuning）通常需要极高温度（>2.0）平滑分布，易破坏输出连贯性，且无法直接应用于不提供per-token log-prob的商业黑盒模型。
- 现有Verbalization-based方法依赖复杂提示工程要求模型自报告置信度，但模型自我认知能力有限，倾向于输出固定高置信表达，且额外提示指令常损害MCQA原始准确率。
- 传统校准评估过度依赖ECE单一指标，无法捕捉可靠图的单调性与置信度分布的聚集偏置，缺乏对“真正良好校准”的系统刻画。

## 核心贡献（创新点）
- 提出UF Calibration框架：将置信度解耦为问题不确定性（基于采样答案频率的信息熵）与答案忠实度（基于贪心选项替换构建的层级链），与以往方法的本质区别在于彻底摆脱了对模型内部概率分布或强自我感知提示的依赖，转而通过外部行为交互量化置信。
- 设计层级忠实度 elicitation 机制：利用"All other options are wrong."作为干预提示，通过贪心重选构建忠实度链并施加指数衰减权重（$\tau^i$），与仅依赖单次采样频率或静态logit的基线形成根本性区分。
- 提出IPR与CE双指标：IPR从可靠图单调性角度度量低/高置信答案的准确率排序一致性；CE从分布均匀性角度揭示大模型盲目聚集于固定置信区间的隐蔽现象，二者共同补全了ECE的评估盲区。
- 深入探讨“Truly Well-Calibrated Confidence”：论证真正良好的校准应在ECE、IPR与CE三者间取得平衡，而非单一优化，为社区校准研究提供了新的理论参照系。

## 方法详解
- **采样与不确定性估计（Sampling & Uncertainty）**：对问题 $Q$ 进行 $K$ 次采样（默认 $K=10$, $T=1.0$, top-p=1.0），统计各候选答案频率分布 $\mathcal{P}_{\text{sampled}}$。问题不确定性定义为标准化信息熵：$\text{Uncertainty}(Q) = -\frac{\sum_{i=1}^{M} p_i \log p_i}{\log M}$（$M$ 为选项数），熵值越接近1表示模型对该问题越不确定。
- **忠实度 elicitation（Fidelity Eliciting）**：针对每个采样答案 $(a_i, o_i)$，将其选项内容替换为固定短语"All other options are wrong."，并使用**贪心解码**再次查询。若模型仍选该项则忠实度高；若改选其他项，则移除原选项继续贪心，直至模型选择该短语，由此构建层级忠实度链 $\mathcal{C} = (d_1 \to d_2 \to \cdots \to d_n)$。从右至左第 $i$ 个元素的归一化忠实度权重为 $\frac{\tau^i}{\sum \tau^i}$（默认 $\tau=2$），再按采样频率加权融合所有链得到最终忠实度 $F(a_i) = \sum_j \mathcal{P}_{\text{sampled}}(\mathcal{C}_j) \cdot \text{Fidelity}_{\mathcal{C}_j}(a_i)$。
- **置信度融合（Confidence Estimation）**：最终置信度定义为不确定性补集与忠实度的乘积：$\text{Conf}(Q, a_i) = (1 - \text{Uncertainty}(Q)) \cdot F(a_i)$。该方法最多两阶段调用模型，对黑盒RLHF-LMs高度友好。
- **新型评估指标**：
  - **IPR**：$\text{IPR}_M = \frac{\text{IP}}{C_K^2}$，统计可靠图中逆序对比例，衡量置信度排序与实际准确率的单调性。
  - **CE**：$\text{CE}_M = -\frac{\sum p_i \log p_i}{\log M}$，计算各置信分箱密度分布的熵，值越大表示置信度分布越均匀。

## 实验与结果
- **数据集与模型**：4个MCQA基准（ARC-Challenge, MMLU, CommonSenseQA, TruthfulQA），6个RLHF-LM（GPT-3.5-Turbo, GPT-4-Turbo, Baichuan2-13B-Chat, LLaMA2-7B/13B/70B-Chat），全程0-shot。
- **最强结果与提升**：
  - GPT-3.5-Turbo在TruthfulQA上 $\text{ECE}_{10}=0.074$（较Sampled的0.147降幅达49.7%），IPR=0.133，CE=0.775；MMLU上 $\text{ECE}_{10}=0.088$ 优于Verb(0.138)与Ling(0.197)。
  - LLaMA2-13B-Chat在ARC-Challenge上 $\
