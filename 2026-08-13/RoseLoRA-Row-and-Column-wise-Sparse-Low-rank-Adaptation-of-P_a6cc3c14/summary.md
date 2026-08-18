---
title: "RoseLoRA-Row-and-Column-wise-Sparse-Low-rank-Adaptation-of-P"
source: https://aclanthology.org/2024.emnlp-main.57.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:29:24"
---

# 论文速读：RoseLoRA-Row-and-Column-wise-Sparse-Low-rank-Adaptation-of-P

## 一句话总结
本文提出了一种行与列方向稀疏的低秩自适应方法（RoseLoRA），通过将稀疏约束从低秩矩阵乘积转化为对两个低秩矩阵行列方向的独立约束，实现预训练语言模型参数的精准选择性更新，在知识编辑和通用微调任务上均显著优于现有PEFT方法。

## 研究问题与动机
- **LoRA密集更新的局限性**：现有LoRA家族方法在推理时通过低秩矩阵乘积 $\pmb{B}\pmb{A}$ 更新全部预训练参数，属于稠密更新，缺乏对模型内部知识的细粒度控制。
- **知识编辑任务的特殊性**：知识编辑要求仅修改特定事实性知识，同时严格保留其余预训练知识；LoRA的密集更新极易引发不必要的参数扰动与知识遗忘。
- **直接稀疏化的优化难题**：对乘积 $\pmb{B}\pmb{A}$ 直接施加 $\ell_0$ 稀疏约束是NP-hard问题，且即便分别对 $\pmb{A}$ 和 $\pmb{B}$ 独立稀疏化，其乘积仍可能是稠密的（如Example 1所示）。
- **缺乏兼顾效率与选择性的PEFT框架**：亟需一种既能保持低秩参数高效性，又能实现精确、稀疏参数更新的微调范式。

## 核心贡献（创新点）
1. **提出RoseLoRA稀疏自适应框架**：将稀疏约束从难优化的矩阵乘积转移至低秩矩阵 $\pmb{A}$ 的各行与 $\pmb{B}$ 的各列，实现轻量级且精准的选择性参数更新。
2. **给出乘积稀疏性的理论下界**：严格证明Proposition 1，表明在行列稀疏约束下，$\pmb{B}\pmb{A}$ 的稀疏程度存在可计算的数学下界，确保更新并非偶然稀疏。
3. **设计基于敏感度的迭代剪枝策略**：利用权重与梯度的乘积作为敏感度指标，结合指数移动平均平滑与三次方衰减预算，动态保留每行/列最重要的参数。
4. **统一支持知识编辑与通用微调**：在基础框架上额外引入 $\ell_2$ 范数约束，使同一方法可同时胜任知识编辑（高选择性）与通用下游任务（高效微调）。
5. **大规模实验验证优越性**：在5个基准、20+数据集上全面超越LoRA、AdaLoRA、ReFT、ROME等基线，参数量最低降至0.015%，且显著缓解微调遗忘。

## 方法详解
- **基础形式**：继承LoRA，将权重更新表示为 $\pmb{W} = \pmb{W}^o + \pmb{B}\pmb{A}$，其中 $\pmb{A} \in \mathbb{R}^{r \times d_2}, \pmb{B} \in \mathbb{R}^{d_1 \times r}$，$\pmb{W}^o$ 冻结。
- **稀疏约束转化**：原优化目标为 $\min_{A,B} \mathcal{L}(\mathcal{D}; \pmb{W}^o + \pmb{B}\pmb{A}) \quad \text{s.t.} \quad \frac{\|\pmb{B}\pmb{A}\|_0}{d_1 d_2} \leq \tau$。由于难以直接优化，转化为对每行/列施加独立 $\ell_0$ 约束：
  $$\frac{\|\pmb{A}_{i*}\|_0}{d_2} \leq \tau, \quad \frac{\|\pmb{B}_{*i}\|_0}{d_1} \leq \tau, \quad i=1,\dots,r$$
- **理论下界（Proposition 1）**：证明 $\pmb{B}\pmb{A}$ 的稀疏性满足：
  $$s(\pmb{B}\pmb{A}) \geq \max\left\{0,\ 1 + \sum_{i=1}^r \big(s(\pmb{A}_{i*}) + s(\pmb{B}_{*i}) - s(\pmb{A}_{i*})s(\pmb{B}_{*i})\big) - r\right\}$$
  表明行列稀疏度越高，乘积稀疏性越有保障。
- **优化流程**：
  1. **梯度更新**：对 $\pmb{A}, \pmb{B}$ 执行标准SGD得到 $\tilde{A}^{(t)}, \tilde{B}^{(t)}$。
  2. **敏感度计算**：$I(W_{ij}) = |W_{ij} \cdot \nabla_{W_{ij}} \mathcal{L}|$，并通过EMA平滑 $\bar{I}^{(t)} = \beta \bar{I}^{(t-1)} + (1-\beta)I^{(t)}$ 降低方差。
  3. **行列剪枝**：对每行 $\pmb{A}_{i*}$ 和每列 $\pmb{B}_{*i}$，保留重要性得分top-$\tau^{(t)}$比例的元素，其余置零。
  4. **预算衰减**：$\tau^{(t)}$ 采用三次方衰减策略，从 $1$ 平滑过渡至目标稀疏度 $\tau$，再稳定至训练结束，保障训练稳定性。
- **知识编辑扩展**：额外添加 $\ell_2$ 范数约束 $\|\pmb{A}\|_F^2 \leq \alpha, \|\pmb{B}\|_F^2 \leq \alpha$，剪枝后对矩阵进行裁剪以满足约束，防止编辑幅度过大破坏原始知识。

## 实验与结果
- **数据集与设置**：基于LLaMA-7B/7B-chat和RoBERTa-large，覆盖知识编辑（WikiData_recent, WikiData_counterfact, ZsRE, WikiBio）、常识推理（8数据集）、算术推理（4数据集）、指令遵循（Alpaca-Eval v1.0）与NLU（GLUE 8数据集）。
- **知识编辑最强结果**：在WikiData_recent上Edit Success达98.4%，AVG达73.7；WikiData_counterfact AVG 76.7；ZsRE Edit Success 100%；WikiBio AVG 84.6。全面超越ROME、MEMIT、AdaLoRA与FT-L，且可迁移性（Portability）与局部性（Locality）显著提升。
- **通用微调对比**：
  - **常识推理**：参数量仅0.03%（与LoReFT相当，远低于LoRA的0.83%），AVG准确率80.7%，在8个数据集中5个排名第一。
  - **算术推理**：AVG 45.9%，保留LoRA 97%精度，参数量仅为LoRA的1/22；较LoReFT提升约6.3%。
  - **指令遵循**：参数量0.0037%，Win-rate 85.77%，超越LoReFT（85.60）与LoRA（81.48）。
  - **NLU（GLUE）**：RoBERTa-large上AVG 89.0，超越LoRA（88.1）；RTE提升3.4%。
- **遗忘与数据效率**：微调后在TriviaQA、MMLU等未见任务上保留超90%性能，显著优于LoRA；当训练数据仅12.5%时，RoseLoRA在GSM8K上反超LoRA，证明其在小样本场景的优势。

## 相关工作脉络
- **LoRA家族**：通过低秩矩阵近似权重更新，但更新为稠密矩阵；本文方法通过行列稀疏约束实现真正的参数选择性更新。
- **AdaLoRA / SoRA**：对LoRA进行通道/维度级剪枝以提升计算效率，但更新仍为稠密；本文在稀疏粒度与选择精度上更进一步。
- **知识编辑基线（ROME/MEMIT/
