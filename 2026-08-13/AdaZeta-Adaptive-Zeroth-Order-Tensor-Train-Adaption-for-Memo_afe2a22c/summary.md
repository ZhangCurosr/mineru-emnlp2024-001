---
title: "AdaZeta-Adaptive-Zeroth-Order-Tensor-Train-Adaption-for-Memo"
source: https://aclanthology.org/2024.emnlp-main.56.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:13:10"
field: "大模型高效微调与零阶优化"
keywords: ["zeroth-order fine-tuning", "tensor-train decomposition", "parameter-efficient fine-tuning", "memory-efficient LLM", "gradient-free optimization", "convergence analysis"]
innovations: ["提出AdaZeta框架，将TT张量适配器与亚线性自适应查询调度结合以同时提升ZO微调精度与稳定性", "设计并行收缩TT因子策略，解决顺序收缩在前向耗时上的瓶颈以适配ZO双前向需求", "给出非凸ZO优化的收敛上界，显式揭示维度d与查询调度对收敛速率的影响机制"]
benchmarks: ["GLUE (SST-2/5, QNLI, MNLI, SNLI, RTE, MR)", "SuperGLUE (CB, BoolQ, WSC, WIC, COPA, ReCoRD)", "SQuAD"]
---

# 论文速读：AdaZeta—Adaptive Zeroth-Order Tensor-Train Adaption for Memory-Efficient LLM Fine-Tuning

## 一句话总结
针对零阶（ZO）微调 LLM 时性能显著下降且容易发散的两大痛点，作者提出 **AdaZeta** 框架：用**张量训练（TT）分解的超轻量适配器**降低可训练参数维度以改善梯度估计质量，并用**亚线性增长的自适应查询数调度**消除大规模 ZO 微调的发散风险；在 Roberta-Large 与 Llama-2-7B 上均超越 MeZO / MeZO-LoRA / Sparse-MeZO，同时保持约 8× 的显存压缩。

## 研究问题与动机
- **全参数一阶微调（FT）显存成本过高**：随着 LLM 规模增长，反向传播图占用显存越来越大。
- **MeZO 等纯前向 ZO 方法性能缺口大且易发散**：仅靠两次前向差值估计梯度，维度越高噪声越大；尤其在 Llama-2-7B 上频繁出现 loss 发散（见图 1）。
- **现有改进路线各有缺陷**：增大 batch size 会抵消显存优势；MeZO-SVRG 需额外副本导致显存翻倍；Sparse-MeZO 依赖权重阈值剪枝、跨任务不稳定；MeZO-LoRA 因 LoRA 极低 rank 表示能力不足，提升有限。
- **ZO 精度本质依赖维度**：随机零阶梯度估计（RGE）的方差与可训练参数维度 $d$ 正相关，因此**减少 $d$** 比"加大 batch"更契合显存-效率双重目标。

## 核心贡献（创新点）
1. **提出 AdaZeta 框架**：将 TT 分解的张量适配器与自适应查询调度结合，在 Roberta-Large 和 Llama-2-7B 上全面优于 MeZO / MeZO-LoRA / Sparse-MeZO，收敛更快且不发散。
2. **设计并行收缩张量适配器**：把 TT 因子序列分组并行收缩，解决原 TT 方法顺序收缩导致的前向耗时增加问题，适配 ZO 每步需两次前向的瓶颈。
3. **提出亚线性增长的自适应查询调度** $Q_k = \min(\alpha e_k^\beta, Q_{max})$，理论证明可保证非凸 ZO 优化收敛，且比"增大 batch size"更省显存。
4. **给出收敛速率理论界**：将收敛上界显式表达为查询调度 $\{Q_k\}$ 和维度 $d$ 的函数，揭示 $Q=1$ 在 $d$ 大时无法趋于零、而亚线性递增可消除该问题的机制。
5. **实测显存-时间权衡**：在 Llama-2-7B 上以 14 GB 完成 SST-2 微调（较 FT 缩减 >8×），且在多任务上以 <1K 步达到 MeZO-LoRA 需 ~6K 步才能达到的 loss 水平，实现 **6× 步数加速**。

## 方法详解
### 3.1 零阶估计基础
在 mini-batch $\mathcal{B}$ 上定义损失 $\ell(w; \mathcal{B})$，训练步 $k$ 的随机零阶梯度估计（RGE）为：
$$
\nabla \hat{\ell}(w_k) = \sum_{q=1}^{Q_k} \frac{\ell_\mathcal{B}(w_k + \epsilon z_q) - \ell_\mathcal{B}(w_k - \epsilon z_q)}{2\epsilon} z_q, \quad z_q \sim \mathcal{N}(0, I_d)
$$
仅需两次前向，无需反向传播图；估计噪声方差与维度 $d$ 成正比，因此降低 $d$ 是关键。

### 3.2 AdaZeta 框架两组件
**(1) 快速前向张量适配器**
- 用 **Tensor-Train (TT)** 分解替代标准权重矩阵 $W \in \mathbb{R}^{m \times n}$：将其重塑为 $2o$ 阶张量后分解为因子序列 $[\mathcal{G}_1, \dots, \mathcal{G}_{2o}]$，其中 $\mathcal{G}_i \in \mathbb{R}^{r_{i-1} \times k_i \times r_i}$，$r_0=r_{2o}=1$。
- 每个编码器/解码器块中，**attention 与 FFN 之后**插入两个 TT 层+中间非线性层构成的适配器；**冻结 LayerNorm**（与 Yang et al. 2024a 不同），避免 ZO 对缩放系数的噪声估计破坏性能。
- 原始 TT 前向为顺序收缩，作者提出**二分/三分/四分组的并行收缩**：如二分情形 $W = \mathrm{R}(\prod_{i=1}^o \mathcal{G}_i \cdot \prod_{j=o+1}^{2o} \mathcal{G}_j)$，避免高维中间张量、提升前向速度。
- 可训练参数较 LoRA 降低 **80%+**。

**(2) 自适应查询数调度**
- 每轮epoch开始时更新查询数：$Q_k = \min(\alpha e_k^\beta, Q_{max})$，其中 $e_k = \lfloor k / \lceil D/B \rceil \rfloor$，$\alpha \in (0,1)$、$\beta \in (0,1)$。
- 中等规模模型（Roberta-Large）固定 $Q=1$（噪声已小）；大规模模型（Llama-2-7B）使用 $\alpha=0.85, \beta=0.45, Q_{max}=20$。
- 每 epoch 内 $Q_k$ 固定，避免频繁动态调整开销。

### 3.3 理论分析要点
- 在非凸设置下给出 Theorem 1 收敛界：
$$
\mathbb{E}[\|\nabla \ell(w_T)\|^2] \le \mathcal{O}\!\left(\frac{R + \epsilon^2 L + C(d,\epsilon)\sum_k \frac{1}{Q_k}}{K \epsilon}\right)
$$
- 当 $Q_k \equiv 1$ 时，$C(d,\epsilon) \propto d$，上界难以趋于零；而取 $\alpha=\beta=0.5$ 的亚线性调度可使 $\sum 1/Q_k = \mathcal{O}(\sqrt{K})$，从而随 $K\to\infty$ 梯度范数平方期望趋于零。
- 明确揭示**维度 $d$ 与查询调度共同决定收敛**，为 TT 降维 + 自适应查询的双重设计提供理论支撑。

### 算法伪码（Alg. 1）
每步 $k$：按调度计算 $Q_k$ → 循环 $Q_k$ 次做 $+ \epsilon z_q$ 与 $- \epsilon z_q$ 两次前向得到 $\ell_+^q, \ell_-^q$ → 恢复原权重 → 平均得梯度估计 → 做 SGD 式更新 $w \leftarrow w - \eta \nabla \hat{\ell}$。

## 实验与结果
### 实验设置
- 平台：NVIDIA A100-40GB。
- 模型：Roberta-Large（7 任务）、Llama-2-7B（10 任务，来自 SuperGLUE + SQuAD/DROP）。
- 低资源设定：Llama 实验每任务 1K train / 500 val / 1K test（prompt-based fine-tuning）。
- 基线：FT(AdamW)、Zero-shot、ICL、LP、MeZO、MeZO-LoRA、Sparse-MeZO。

### Roberta-Large（表 1）
- 多数任务中 AdaZeta 超越 MeZO-LoRA：BS=16 时 7 任务中 5 项占优；BS=64 时同样显著。
- **RTE 66.8**（vs MeZO-LoRA 52.7 / 63.9）、**SST-5 48.3**（vs 44.8 / 43.0）、**MR 87.0**（vs 85.7 / 87.4）。
- 一致性：BS=16 与 BS=64 结果接近，表明低维度带来的噪声抑制使收敛更稳定。

### Llama-2-7B（表 2）
- **8/10 任务胜过全部 ZO 基线**：RTE 74.0（MeZO 54.6 / MeZO-LoRA 59.6）、CB 75.0（74.0）、BoolQ 79.4（71.6）、MultiRC 68.2（67.2）、COPA 94.0（89.0）、SQuAD 80.0（80.0）等。
- 部分任务（RTE、CB、COPA）甚至超越一阶 FT / LoRA。
- 图 1 显示：达相同 loss 0.4 时 MeZO-LoRA 需 ~6K 步，AdaZeta 不到 1K 步，**6× 步数加速**。

### 显存与时间（表 3、表 4、图 3）
- SST-2 上 Llama-2-7B 微调仅用 **14 GB**，较 FT（~118 GB）减 **>8×**。
- 达相同 loss 阈值的 GPU 小时：SST-2 仅 1.1h（vs MeZO-LoRA(BS=16) 0.6h 略慢但远优于 BS=64 的 3.0h；WIC 1.0h 最低）。
- 相比 LoRA/r=1/BS=1 仍节省 **2.5× 显存**；即使对比 LoRA/r=1/BS=1，AdaZeta/r=8/BS=16 仍省约 **50%**。

## 相关工作脉络
1. **MeZO (Malladi et al. 2023)**：首次提出纯前向 ZO LLM 微调；本文在其基础上引入 PEFT + 自适应查询，显著缩小 FO-ZO 性能差距并解决发散。
2. **MeZO-LoRA (Malladi et al. 2023)**：把 LoRA 适配到 ZO；本文指出 LoRA 极低 rank 表达能力不足，转而使用 TT 适配器（参数再降 80%+）。
3. **Sparse-MeZO (Liu et al. 2024)**：按权重幅值剪枝减少可训练维度；本文认为其跨任务/超参不稳定，采用结构化 TT 分解更可靠。
4. **MeZO-SVRG (Gautam et al. 2024)**：把 SVRG 方差缩减移植到 ZO；需保留额外参数副本使显存翻倍；本文用亚线性查询替代，不增显存。
5. **ZO-AdaMU (Jiang et al. 2024)**：把一阶自适应动量策略适配到 ZO；但引入 optimizer state 显存开销；本文通过降维从根源降噪。
6. **LoRA / Adapters / Prefix-tuning (Hu et al. 2021; Houlsby et al. 2019; Li & Liang 2021)**：PEFT 经典工作；本文将其思想引入 ZO 场景并做针对性改造（冻结 LN、并行收缩）。
7. **Tensor-Train 分解 (Oseledets 2011; Novikov et al. 2015)** 及 **LoRTTA (Yang et al. 2024a)**：TT 前作在 FO 中表现优异但顺序收缩速度慢；本文提出并行收缩使其适配 ZO 双前向需求。

## 局限性与未来方向
- **查询当前串行执行**：每步 $Q_k$ 次扰动按 for 循环顺序完成，限制进一步提速；作者建议未来做多 GPU 并行或分布式查询。
- **超参依赖经验调优**：TT 因子形状、秩、bottleneck、$\alpha/\beta/Q_{max}$ 需实验试错确定；未给出自动搜索策略。
- **仅评估 NLU 与生成类基准**：尚未在代码生成、多模态、长上下文等场景验证泛化性。
- **LayerNorm 冻结可能损失一定表征力**：虽然实验表明冻结 LN 避免噪声放大，但在某些任务上或许可以学习部分缩放。
- **自适应查询调度未与其他 PEFT 方法结合的系统评估**：附录提到未来可将该调度应用到 MeZO-LoRA。

## 研究启发与可借鉴点
1. **"降维优于加样本"的 ZO 降噪思路**：ZO 方差正比于维度 $d$，因此从模型结构上压缩 $d$（TT / 结构化 adapter）比暴力增大 batch 更符合显存约束；该原则可迁移到其他 black-box 优化场景。
2. **并行收缩 TT 因子**：把顺序张量网络转为分组并行，是加速任何基于 TT/CPT 结构的推理-训练流程的通用技巧，可复用到 FO 微调与推理压缩。
3. **亚线性查询调度机制**：$Q_k \propto e_k^\beta$ 的思想可推广至其他一阶/二阶混合优化、强化学习策略梯度、神经网络 ODE 敏感性分析等需要 Monte Carlo 估计梯度的场景。
4. **冻结 LayerNorm 以规避 ZO 噪声**：在 ZO 调优中避免对"尺度因子"做噪声估计是一条实用经验，对任何使用 RMSNorm/LayerNorm 的 ZO 方法都有参考价值。
5. **与 LORA/QLoRA 的结合潜力**：将 TT 适配器与低比特量化（QLoRA 路线）组合，或把自适应查询用于 MeZO-LoRA，都是可立即展开的后续实验方向。

## 关键术语表
- **Zeroth-Order (ZO) 优化**：仅依赖目标函数值而不需梯度的优化方法，常用随机扰动差分估计梯度。
- **Randomized Zeroth-Order Gradient Estimation (RGE)**：对参数施加随机高斯扰动、通过两次前向损失差估计梯度。
- **Tensor-Train (TT) 分解**：把高维张量/矩阵分解为低秩因子链，用极少量参数近似原始权重。
- **张量适配器 (Tensorized Adapter)**：基于 TT 分解的轻量适配器，参数量较 LoRA 再降 80%+。
- **自适应查询调度 (Adaptive Query Schedule)**：按亚线性函数逐步增加每步扰动查询次数以控制梯度估计方差。
- **MeZO / MeZO-LoRA / Sparse-MeZO**：系列 ZO 微调方法，分别代表基础版、LoRA 版与稀疏剪枝版。
- **显存缩减 8×+**：ADAZeta 在 Llama-2-7B 上仅需 14 GB 而非 FT 的 118 GB，主要来自无反向传播图 + 冻结主体参数。

## 可复现要素
- **数据集**：GLUE/SuperGLUE 标准子集（SST-2/5, QNLI, MNLI, SNLI, RTE, MR, CB, BoolQ, WSC, WIC, COPA, ReCoRD）及 SQuAD；多为公开 benchmark。
- **代码/权重**：论文未提供开源仓库链接与模型权重（论文未提及）。
- **关键超参**：Roberta-Large：lr ∈ {1e-4, 5e-5}，BS ∈ {16, 64}，ε=1e-3，bottleneck=64，rank=5；Llama-2-7B：同上 lr/BS，α=0.85, β=0.45, Q_max=20，bottleneck ∈ {8, 64}，rank ∈ {5, 8, 16}；FT lr 用 1e-6 或 5e-7。
- **硬件**：NVIDIA A100-40GB。
