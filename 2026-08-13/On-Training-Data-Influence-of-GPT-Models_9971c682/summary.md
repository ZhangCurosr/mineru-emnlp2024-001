---
title: "On-Training-Data-Influence-of-GPT-Models"
source: https://aclanthology.org/2024.emnlp-main.183.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:13:39"
field: "训练数据归因与可解释性"
keywords: ["Training Data Attribution", "GPT", "Training Dynamics Simulation", "Influence Functions", "Instruction Tuning", "Markov Process Modeling"]
innovations: ["提出GPTfluence，首次系统性地将训练数据影响分析扩展到GPT自回归生成模型", "基于n阶马尔可夫过程和预训练编码器特征模拟，突破Simfluence仅预测test loss且无法泛化的局限", "支持test loss、BLEU、ROUGE-L等多指标轨迹预测，覆盖NLU和NLG任务"]
benchmarks: ["FLAN (RTE, SST-2, BoolQ, WebNLG, WMT-16 DE/EN)", "Pythia model series (14M-2.8B)", "GPTDynamics dataset"]
---

# 论文速读：On Training Data Influence of GPT Models

## 一句话总结
论文提出了 **GPTfluence**，一种基于特征模拟的训练数据影响分析方法，通过 n 阶马尔可夫过程模拟 GPT 模型训练动态，能够跨模型规模（14M–2.8B 参数）和任务类型（NLU/NLG）准确预测训练样本对测试点性能轨迹的影响，并具备对未见数据的强泛化能力。

## 研究问题与动机
1. **GPT 架构的训练数据影响研究几乎空白**：现有 TDA 工作（如 TracIn、Simfluence）主要针对 BERT/T5 等 Encoder 架构，缺乏对 GPT 等自回归生成模型的系统性研究。
2. **现有方法仅关注 test loss，忽视生成任务关键指标**：BLEU、ROUGE 等生成质量指标对 GPT 下游任务评估至关重要，但已有方法无法有效预测这些轨迹。
3. **Simfluence 泛化性不足**：其参数学习依赖于训练数据索引映射，无法推广到未见过的训练样本/测试样本，限制了实际应用场景。
4. **大模型训练成本制约影响力分析**：随着模型规模扩大，逐步骤梯度计算（如 TracIn-CP）的计算开销急剧增长，需要更高效的模拟方法。

## 核心贡献（创新点）
1. **提出 GPTfluence，首个面向 GPT 模型的系统性训练数据影响模拟框架**：通过预训练编码器+线性投影的方式学习每步训练样本对测试指标的乘性/加性影响因子，区别于仅依赖梯度内积的 TracIn/Grad-Dot。
2. **从 n 阶马尔可夫过程角度建模训练动态，突破 Simfluence 的一阶局限**：将测试指标当前值建模为前 n 步历史值与当前训练批次的加权组合，捕获更丰富的时序依赖关系；Simfluence 等价于 n=1 且输入为样本索引的特例。
3. **支持多指标预测（test loss + BLEU + ROUGE-L），覆盖 NLU 和 NLG 两类任务**：解决了先前方法仅能预测 test loss 的核心缺陷，其中 TracIn/Grad-Dot 根本无法直接适配 BLEU/ROUGE 指标。
4. **释放 GPTDynamics 数据集（350+ 次训练动态记录，6 种模型尺寸，5 类 NLP 任务）**：填补了 GPT 训练动态数据公开资源的空白，促进该领域后续研究。
5. **验证了模型无关的未训练数据泛化能力**：GPTfluence 可通过样本参数化机制处理训练 curriculum 和测试样本均未见的新场景，而 Simfluence 在同类设置下表现严重退化。

## 方法详解
**总体流程**分为三步：（1）收集训练动态数据集 GPTDynamics；（2）训练特征化模拟器；（3）自回归模拟新的性能轨迹。

**训练动态收集**：从完整数据集 $\mathcal{D}$ 中采样 K 个子集进行 K 次训练，记录每个训练步 t 的 batch $c_t$ 和测试样本 $z'$ 在各训练步的目标指标 $y_t = \phi(\theta_t, z')$，构成 GPTDynamics $\mathcal{D}_{run} = \{c^k, y^k\}_{k=1}^K$。

**n 阶马尔可夫模拟公式**：
$$\phi_t(z') = \sum_{j=1}^{n} \alpha_j(c_t) \phi_{t-j}(z') + \beta(c_t)$$
其中 $\alpha_j(c_t)$ 为乘性聚合系数，$\beta(c_t)$ 为加性偏移量，均由当前训练批次内各样本的影响因子求和得到：$\alpha_j(c_t) = \sum_{i \in c_t} A_{i,j}$，$\beta(c_t) = \sum_{i \in c_t} B_i$。

**影响因子计算**：使用冻结的预训练编码器 $\Psi(\cdot)$（如 MiniLM-L6-v2 / BERT / Pythia）将训练样本 $z_i$ 和测试样本 $z'$ 编码为嵌入 $h_{z_i}, h_{z'}$，再通过 Frobenius 内积计算各影响因子：
$$A_{i,j} = \langle \mathbf{W}_{(j)}^\top h_{z_i}, \mathbf{U}_{(j)}^\top h_{z'} \rangle_F, \quad B_i = \langle \mathbf{W}'^\top h_{z_i}, \mathbf{U}'^\top h_{z'} \rangle_F$$
其中权重矩阵 $\mathbf{W}, \mathbf{U}$ 为可学习参数。

**训练目标**：L2 正则化最小二乘回归
$$\Theta^* = \arg\min_\Theta \sum_{t \in T}(y_t - \hat{\phi}_t(z'))^2 + \lambda \|\Theta\|_2^2$$

**推理方式**：给定新训练 curriculum 和测试样本 $z'$，从初始指标值出发，自回归地逐步步进预测完整性能轨迹。

## 实验与结果
**实验设置**：
- 骨干模型：**Pythia** 系列（14M、70M、160M、410M、1B、2.8B 共 6 种尺寸）
- 任务：NLU（RTE、SST-2、BoolQ）+ NLG（WebNLG、WMT-16 DE/EN）
- 训练范式：Instruction Tuning + Fine-tuning 两种
- 评估指标：All-steps MSE、All-steps MAE、Final-step Spearman's ρ
- 基线：TracIn-CP（10-steps / all-steps）、Grad-Dot、Simfluence

**核心结果（Instruction Tuning，test loss 预测）**：

| 模型尺寸 | 方法 | RTE MSE ↓ | RTE Spearman ↑ | WMT-16 DE/EN MSE ↓ | WMT-16 DE/EN Spearman ↑ |
|---|---|---|---|---|---|
| 410M | Ours | **0.220** | **0.644** | — | — |
| 410M | Simfluence | 1.477 | 0.426 | 0.016 | 0.997 |
| 1B | Ours | **0.099** | **0.757** | — | — |
| 1B | Simfluence | 0.889 | 0.360 | 0.171 | 0.925 |
| 2.8B | Ours | **0.132** | **0.969** | 0.001 | **0.999** |
| 2.8B | Simfluence-linear | 2.032 | 0.845 | 0.063 | 0.991 |

- GPTfluence 在全部规模上持续超越所有基线，Spearman 相关系数最高达 **0.999**（WMT-16 DE/EN，2.8B 模型）。
- 410M 模型在 WebNLG 上 MSE 从 Simfluence 的 0.036 降至 **0.002**，提升超 17 倍。
- 在 WMT-16 DE/EN BLEU 预测上，410M 模型 MSE 从 32.15 降至 **7.71**（提升约 4.2 倍），Spearman 从 0.83 升至 **0.92**。

**Fine-tuning 平均提升**（对比 Simfluence）：MSE 降低 **42%**，MAE 降低 **28%**。

**未见数据泛化**（Fine-tuning，RTE）：
- 训练集未见、测试集已知：MSE = 0.346，Spearman = **0.913**
- 测试集未见、训练集已知：MSE = 0.351，Spearman = −0.024（困难情形，但仍有参考意义）
- 两者均未见：MSE = 0.984，Spearman = −0.048

**误标数据检测应用**：GPTfluence 在 SST-2 上可在早期阶段（仅需检查少量样本）比 TracIn-CP 更快识别出标签错误样本，同时最终测试准确率提升更显著。

## 相关工作脉络
1. **TracIn（Pruthi et al., 2020）**：基于梯度内积追踪训练样本对测试损失的累计影响，适用于任意模型架构但计算开销高，主要聚焦 test loss 预测。GPTfluence 与其本质区别在于用数据驱动模拟替代梯度追踪，支持多指标预测和更好泛化。
2. **Simfluence（Guu et al., 2023）**：首个基于训练动态模拟的方法，但参数学习与样本索引一一对应，无法泛化至未见数据；仅预测 test loss。GPTfluence 通过编码器特征表示和 n 阶建模超越其限制。
3. **Influence Functions（Koh & Liang, 2017）**：基于二阶逆 Hessian 的理论影响力函数，理论上精确但实际不可扩展到大模型。GPTfluence 采用一阶近似+数据驱动模拟，实用性强。
4. **Grad-Dot（Charpiat et al., 2019）**：仅在最终模型权重上计算梯度内积，忽略训练全过程，近似粗糙。GPTfluence 利用全轨迹信息，精度更高。
5. **Trak（Park et al., 2023）**：基于哈希投影的规模化影响力估计，侧重大规模场景的近似计算。GPTfluence 侧重精确模拟和指标扩展，而非纯缩放优化。
6. **Datamodels（Ilyas et al., 2022）**：预测模型整体行为而非单个样本影响，属于宏观数据模型。GPTfluence 聚焦个体样本级别的细粒度影响估计。

## 局限性与未来方向
1. **高度依赖完整的训练动态收集**：需运行多次训练以获取足够动态数据，计算成本高；虽然可通过 checkpoint 间隔采样（如间隔 10 步）减少约 90% 收集时间，但精度有所下降。
2. **数据集范围受限**：仅使用 FLAN 子集，涵盖 5 个任务和 6 种模型尺寸，缺乏多语言、多领域和更大规模模型的验证。
3. **未扩展到超大模型（> 2.8B）**：受计算资源限制，13B/72B 级别模型的泛化能力尚待验证。
4. **未探索其他架构和模态**：方法目前仅针对 GPT 架构和 NLP 任务，扩展到 Transformer Encoder、多模态模型的能力未知。
5. **未来方向**：① 将方法推广至更多任务和模型架构（包括视觉/多模态）；② 减少训练动态收集的开销；③ 扩大数据集覆盖范围。

## 研究启发与可借鉴点
1. **冻结预训练编码器提取特征 + 轻量线性投影模拟**的设计模式值得迁移：将复杂模型动态压缩为低维嵌入后做回归预测，可在不影响语义表征的前提下大幅降低计算开销，适用于其他序列模型的 TDA 分析。
2. **n 阶马尔可夫过程的训练动态建模思路**可推广：不仅限于 test loss，也可用于模拟 RLHF 奖励分数轨迹、PPL 轨迹、收敛速度等指标，帮助理解训练过程中的关键拐点。
3. **BLEU/ROUGE-L 等生成指标的时间序列模拟**是重要突破：以往 TDA 工作几乎未触及生成质量指标的预测，本文的方法可直接用于生成任务的数据选择和质量优化。
4. **GPTDynamics 数据集的发布**为后续工作提供了宝贵基准：350+ 条训练动态记录覆盖多尺寸/多任务，可作为 TDA 方法公平比较的标准数据集。
5. **误标数据检测的应用案例**展示了 TDA 方法的实用价值：可将 GPTfluence 与数据清洗流程结合，在训练前自动识别并纠正低质量标注样本，提升下游任务性能。

## 关键术语表
**GPTfluence**：本文提出的特征化模拟方法，通过预训练编码器提取样本嵌入并用 n 阶马尔可夫过程模拟训练动态，预测训练样本对 GPT 模型测试性能的影响。

**Training Data Attribution (TDA)**：训练数据归因，量化单个训练样本对模型预测或性能指标影响的理论框架与方法集合。

**TracIn / TracIn-CP**：基于梯度追踪的训练数据影响估计方法，TracIn-CP 是其在特定 checkpoint 间隔上的近似版本，通过梯度内积累加估计影响。

**Simfluence**：基于数据驱动模拟的训练数据影响方法，学习乘性和加性参数预测测试损失的变化轨迹，但对未见数据泛化能力有限。

**n 阶马尔可夫过程**：假设当前状态仅依赖于前 n 个历史状态的过程；本文将其用于建模测试指标随训练步的演化轨迹，优于 Simfluence 的一阶假设。

**Frobenius 内积（$\langle \cdot, \cdot \rangle_F$）**：矩阵或向量的逐元素乘积求和运算，在本文中被用于计算训练样本和测试样本嵌入之间的交互影响因子。

**GPTDynamics 数据集**：论文公开的训练动态数据集，包含 350+ 次训练运行的完整动态记录和对应指标轨迹，覆盖 6 种模型尺寸和 5 类 NLP 任务。

**Instruction Tuning vs Fine-tuning**：Instruction Tuning 使用统一指令格式对多任务进行微调；Fine-tuning 针对单一任务单独优化，两者均属于后训练适配策略。

## 可复现要素
- **数据集**：FLAN 数据集子集（5 个任务：RTE、SST-2、BoolQ、WebNLG、WMT-16 DE/EN）；GPTDynamics 数据集已公开，代码及数据在 https://github.com/ernie-research/gptfluence
- **代码**：已开源（见上述 GitHub 链接）
- **模型权重**：使用 Pythia 系列公开模型（14M–2.8B），LoRA fine-tuning 配置：rank=8，alpha=4，dropout=0.05
- **关键超参**：Markov order n=1（instruction tuning 和 fine-tuning 均用一阶）；编码器使用 MiniLM-L6-v2（冻结）；L2 正则化 λ=1e-5；optimizer AdamW (0.9, 0.999)，lr=1e-3，warmup=200，batch=128，max epochs=300，early stopping
- **生成评估**：top-p sampling，temperature=0.2，top-p=0.95
- **实验硬件**：NVIDIA Tesla V100 GPU
