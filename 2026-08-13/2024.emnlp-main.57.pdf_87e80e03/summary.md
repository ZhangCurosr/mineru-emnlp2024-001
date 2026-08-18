---
title: "RoseLoRA: Row and Column-wise Sparse Low-rank Adaptation of Pre-trained Language Model for Knowledge Editing and Fine-tuning"
source: https://aclanthology.org/2024.emnlp-main.57.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:10:58"
---

# 论文速读：RoseLoRA: Row and Column-wise Sparse Low-rank Adaptation of Pre-trained Language Model for Knowledge Editing and Fine-tuning

## 一句话总结
本文提出 RoseLoRA，一种在 LoRA 的低秩因子矩阵上施加行/列稀疏约束的参数高效微调方法，通过敏感性评分逐步筛选关键参数进行更新；在知识编辑与五大类通用下游任务上，该方法以更少的更新参数量（低至 0.0037%）实现了优于 LoRA、AdaLoRA、ReFT 等基线的性能，并显著缓解微调引发的知识遗忘。

## 研究问题与动机
- 现有 PEFT 方法（尤其是 LoRA 家族）在推理时将低秩矩阵乘积 $BA$ 作为**稠密矩阵**加到预训练权重上，导致所有参数均被更新，缺乏细粒度控制。
- 知识编辑等任务要求精准修改特定内部事实知识，同时严格保留无关知识；LoRA 的稠密更新会引发不必要的参数扰动，造成严重的知识遗忘与编辑精度下降。
- 直接在乘积矩阵 $BA$ 上施加 $\ell_0$ 稀疏约束是 NP-hard 的，且即便 $A$ 与 $B$ 各自稀疏，其乘积仍可能高度稠密（如文中 Example 1 所示），传统剪枝方法无法直接套用。
- 亟需一种既能保持 PEFT 轻量高效，又能实现对预训练权重**选择性、稀疏化更新**的新范式。

## 核心贡献（创新点）
1. 提出 RoseLoRA 行列稀疏低秩适配框架，通过筛选任务关键参数实现精准更新。（与 AdaLoRA 等仅控制秩但更新仍为稠密矩阵的方法不同，RoseLoRA 在权重增量矩阵层面实现真正的细粒度稀疏选择。）
2. 将难以优化的乘积 $\ell_0$ 稀疏约束转化为对低秩因子 $A$ 的行和 $B$ 的列的稀疏约束，并给出乘积稀疏度的理论下界。（区别于传统剪枝仅对单一完整权重矩阵施加稀疏约束，该方法巧妙规避了低秩乘积结构下的 NP-hard 优化难题。）
3. 设计基于敏感性评分与三次方衰减预算的动态迭代剪枝优化策略。（不同于静态一次性剪枝，该策略在训练过程中渐进式收紧稀疏度，兼顾了优化稳定性与最终稀疏表达力。）
4. 在知识编辑与五大类通用 NLP 任务上进行系统验证，证明极低更新比例下仍可超越 SOTA。（与 ReFT 等方法相比，RoseLoRA 不依赖表征空间的线性假设，直接在预训练权重空间操作，理论适用性更广。）

## 方法详解
- **基础结构**：继承 LoRA 形式，权重更新表示为 $W = W^o + BA$，其中 $A \in \mathbb{R}^{r \times d_2}$，$B \in \mathbb{R}^{d_1 \times r}$，$r \ll \min(d_1, d_2)$。
- **稀疏约束转化**：原始目标 $\min_{A,B} \mathcal{L}(W^o+BA)$ s.t. $\|BA\|_0/(d_1d_2)\leq\tau$ 难以直接优化。作者证明：若限制 $A$ 的每一行与 $B$ 的每一列满足稀疏度 $\leq\tau$，则乘积稀疏度满足理论下界：
  $s(BA) \geq \max\{0,\, 1 + \sum_{i=1}^r [s(A_{i*}) + s(B_{*i}) - s(A_{i*})s(B_{*i})] - r\}$。
- **优化算法**：
  1. **梯度更新**：$\tilde{A}^{(t)} = A^{(t)} - \nabla_A \mathcal{L}$，$\tilde{B}^{(t)} = B^{(t)} - \nabla_B \mathcal{L}$。
  2. **敏感性评分**：采用 $I(W_{ij}) = |W_{ij} \cdot \nabla_{W_{ij}} \mathcal{L}|$ 衡量参数重要性，并使用 EMA 平滑 $\bar{I}^{(t)} = \beta \bar{I}^{(t-1)} + (1-\beta)I^{(t)}$ 降低方差。
  3. **行列剪枝**：对 $A$ 的每一行和 $B$ 的每一列，保留重要性分数 top-$\tau^{(t)}$ 的元素，其余置零。
  4. **动态预算调度**：稀疏预算 $\tau^{(t)}$ 按三次方策略（cubic strategy）从 1 逐渐衰减至目标 $\tau$，保障训练稳定性。
- **知识编辑扩展**：额外引入 $\ell_2$ 范数约束 $\|A\|_F^2 \leq \alpha,\|B\|_F^2 \leq \alpha$，并在每步剪枝后对 $A,B$ 进行截断以满足约束。

## 实验与结果
- **基准与基线**：覆盖知识编辑（WikiData_recent, WikiData_counterfact, ZsRE, WikiBio）、常识推理、算术推理、指令遵循、GLUE 共五类二十余数据集；基线包括 AdaLoRA、ROME、FT-L、MEMIT、Prefix Tuning、Adapter、LoRA、DoRA、LoReFT、ReFT 等。主实验基于 LLaMA-7B / LLaMA-2-7B 与 RoBERTa-large。
- **知识编辑**：RoseLoRA 在四数据集 AVG 均大幅领先，WikiData_recent 达 73.7（次优 MEMIT 61.2），编辑成功率最高且本地性/可迁移性最优。
- **常识推理**：参数量仅 0.03%，AVG 80.7%，超越 LoRA (74.7) 与 LoReFT (80.2)，8 个数据集中 5 个第一。
- **算术推理**：AVG 45.9%，保持 LoRA (46.9) 的 97% 性能，但更新参数仅为 LoRA 的 1/22，较 LoReFT 提升约 6.3%。
- **指令遵循**：参数仅 0.0037%，Alpaca-Eval Win-rate 85.77%，超越 LoReFT (85.60) 及 LoRA (81.48)。
- **NLU (GLUE)**：RoBERTa-large 上 AVG 89.0，超越 LoRA (88.1)；RTE 提升 3.4%，证明亦适用于编码器架构。
- **遗忘测试**：在 Commonsense170K/Math10K/Ultrafeedback 微调后，RoseLoRA 在 TriviaQA、MMLU、ARC-c 上的表现显著优于 LoRA（如 Commonsense170K 后 TriviaQA 从 9.0 升至 47.8）。
- **数据规模敏感性**：随训练数据减少，RoseLoRA 优势扩大；仅用 12.5% Math10K 数据时已在 GSM8K 上反超 LoRA，凸显小样本场景价值。

## 相关工作脉络
- **LoRA
