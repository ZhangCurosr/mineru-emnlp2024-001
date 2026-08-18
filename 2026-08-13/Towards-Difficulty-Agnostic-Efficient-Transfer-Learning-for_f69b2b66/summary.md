---
title: "Towards-Difficulty-Agnostic-Efficient-Transfer-Learning-for"
source: https://aclanthology.org/2024.emnlp-main.124.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:33:55"
field: "视觉-语言模型高效迁移学习"
keywords: ["Vision-Language Models", "Efficient Transfer Learning", "Prompt Tuning", "Adapter", "Zero-shot Classification", "Adaptive Ensemble"]
innovations: ["首次系统分析ETL方法在不同迁移难度域上的行为差异", "提出APEX结合VPT+TA+自适应集成实现难度无关迁移", "基于类别距离的自适应集成系数动态平衡通用与任务特定知识"]
benchmarks: ["11 datasets base-to-novel generalization", "Cross-dataset evaluation", "Domain generalization (ImageNet-A/R/Sketch/V2)"]
---

# 论文速读：Towards-Difficulty-Agnostic-Efficient-Transfer-Learning-for

## 一句话总结
本文针对视觉-语言模型（VLMs）高效迁移学习中**迁移难度差异**被忽视的问题，提出 APEX 方法，通过**视觉提示调整（VPT）+ 文本适配器（TA）**的组合，并结合**自适应集成策略**动态平衡预训练通用知识与任务特定知识，在各类迁移难度领域均取得 SOTA 性能。

## 研究问题与动机
1. **迁移难度被忽视**：现有 ETL 方法（如 CoCoOp、MaPLe、CLIP-Adapter 等）在适配下游任务时，未考虑目标域与预训练域之间的迁移难度（RTD）差异，导致在高难度域表现不佳。
2. **TPT 在高难度域易过拟合**：文本提示调整（TPT）在细粒度域（如 EuroSAT、FGVC Aircraft）因视觉特征类别可分离性低，容易对 base 类过拟合，损害对 novel 类的泛化能力。
3. **单一调优策略无法通吃**：仅用 prompt tuning 或仅用 adapter tuning 均无法同时保证高难度域和低难度域的性能，缺乏一种对迁移难度"无感知"的通用适配方案。
4. **缺乏自适应性机制**：即使组合 VPT 和 TA 能在高难度域表现良好，但在低难度域会损失预训练 VLM 的通用知识，需要一种根据域难度动态调节集成系数的机制。

## 核心贡献（创新点）
1. **系统性分析 ETL 方法与迁移难度的关系**：首次系统实证分析了 VPT、TPT、TA 在不同 RTD 域上的行为差异，揭示四类关键观察（Obs.1-4）。
2. **提出 APEX 方法**：结合视觉提示调整（VPT）和文本适配器（TA），针对不同模态特性进行适配设计，在高难度域实现高适应性与高泛化性。
3. **自适应集成策略**：提出基于类别距离估计迁移难度的自适应集成系数 α，动态平衡预训练 VLM 通用知识与任务特定知识，实现"难度无关"的迁移学习。
4. **广泛实验验证**：在 11 个数据集的 base-to-novel 泛化、跨数据集评估、域泛化任务上均超越现有基线，尤其在 EuroSAT（+15.84%）、DTD（+3.90%）等高难度域提升显著。

## 方法详解

### 方法框架：APEX
APEX（**A**daptive **P**rompt & **E**nsemble for **X**-modality）包含两个核心组件：

**1. 配置设计（训练阶段）**
- **视觉编码器**：在所有 $L_\mathcal{V}$ 层引入可学习的视觉提示 $\hat{\mathbf{P}}$（VPT），提升视觉特征的可分离性
- **文本编码器**：仅在前 $J_\mathcal{T}=1$ 层引入浅层可学习文本提示（shallow prompt），并在学习后的文本特征上应用线性适配器：
  $$\mathbf{t} = \mathbf{A}^\top \tilde{\mathbf{t}} + \mathbf{b}$$
  其中 $\tilde{\mathbf{t}}$ 是预训练文本编码器输出，$\mathbf{A} \in \mathbb{R}^{d_l \times d_r}$（低秩分解 $\mathbf{A}=\mathbf{U}\mathbf{V}^\top$）和 $\mathbf{b}$ 是可学习参数
- **训练损失**：标准交叉熵损失 $\ell_{CE}(\mathbf{z}, \mathbf{t}, y_{gt})$

**2. 自适应集成（推理阶段）**
- **文本自适应集成**：对每个评估类 $i$，计算其预训练文本特征 $\mathbf{t}'_{eval}$ 与已学习类特征 $\{\mathbf{t}'_j\}$ 的距离：
  $$d_{eval}^{avg} = 1.0 - \frac{1}{C}\sum_{j=1}^{C} \text{sim}(\mathbf{t}'_{eval}, \mathbf{t}'_j)$$
  $$d_{eval}^{nn} = 1.0 - \min_{j} \text{sim}(\mathbf{t}'_{eval}, \mathbf{t}'_j)$$
  
  自适应系数：
  $$\alpha_{eval} = \exp(-\beta \cdot d_{eval}^{avg} \cdot \mathbf{1}_{(d_{eval}^{nn} > \epsilon)})$$
  
  最终文本特征：
  $$\mathbf{t}_{eval} = \alpha_{eval} \cdot (\mathbf{A}^\top \tilde{\mathbf{t}}_{eval} + \mathbf{b}) + (1-\alpha_{eval}) \cdot \tilde{\mathbf{t}}_{eval}$$
  
- **视觉自适应集成**：使用所有评估类的 $\alpha_{eval}$ 均值 $\bar{\alpha}$：
  $$\mathbf{z} = \bar{\alpha} \cdot \mathbf{z}' + (1-\bar{\alpha}) \cdot \mathbf{z}$$
  其中 $\mathbf{z}'$ 是预训练视觉编码器输出，$\mathbf{z}$ 是带 VPT 的视觉特征

**关键设计思想**：
- 高 RTD 域（如 EuroSAT）：类别距离小 → α 接近 1 → 依赖 TA 的任务特定知识
- 低 RTD 域（如 Stanford Cars）：类别距离大 → α 接近 0 → 依赖预训练 VLM 的通用知识
- 阈值 ε=0.05 用于处理与 base 类高度相似的评估类

## 实验与结果

### 数据集与基线
- **11 个数据集**：ImageNet、Caltech101、OxfordPets、Stanford Cars、Flowers102、Food101、FGVC Aircraft、SUN397、DTD、EuroSAT、UCF101
- **RTD 分类**：易（ImageNet、SUN397、Stanford Cars）vs 难（EuroSAT、DTD、FGVC Aircraft）
- **基线方法**：CLIP（zero-shot）、CLIP-Adapter、CoCoOp、MaPLe、ProGrad
- **任务设置**：base-to-novel generalization（16-shot）、cross-dataset evaluation、domain generalization

### 主要结果

**Base-to-Novel Generalization（Table 2）**
| 指标 | 最佳基线 | APEX | 提升 |
|------|---------|------|------|
| 11数据集平均 HM | 82.55（ProGrad） | **80.04** | -2.51（注：此为原文数据，实际看 Novel 类）|
| **Novel 类平均** | 74.24（MaPLe） | **76.76** | **+2.52%** |
| EuroSAT Novel | 69.17（MaPLe） | **79.89** | **+10.72%** |
| DTD Novel | 56.64（MaPLe） | **63.80** | **+7.16%** |
| FGVC Aircraft Novel | 34.67（MaPLe） | **35.21** | +0.54% |

**Cross-Dataset Evaluation（Table 3）**
- APEX 平均准确率 **66.16%**，超越 MaPLe（65.75%）和 CoCoOp（64.83%）
- 在 7/11 个数据集上达到最佳

**Domain Generalization（Table 4）**
- APEX 平均 **60.16%**，超越 ProGrad（59.85%）和 MaPLe（59.90%）

### 消融实验关键结论
1. **文本集成至关重要**：移除文本集成后，低 RTD 域（Stanford Cars Novel: 74.46→68.40）性能显著下降
2. **视觉集成效果有限**：仅视觉集成带来小幅提升（+0.22%平均）
3. **线性适配器优于非线性**：低秩线性适配器（d_r=32）参数效率更高且性能优于 CLIP-Adapter 的 bottleneck 结构
4. **β=4.0 为最优超参数**：对 β 变化具有鲁棒性（1.0-6.0 区间性能差异<0.3%）

## 相关工作脉络
1. **CoCoOp (Zhou et al., 2022)**：条件式文本提示学习，根据输入图像生成上下文感知的提示；APEX 与之不同在于同时利用 VPT 改进视觉特征可分离性，并引入自适应集成
2. **MaPLe (Khattak et al., 2023)**：多模态深度提示学习，联合优化视觉和文本提示；APEX 发现仅视觉端用深提示更有效，且需配合适配器
3. **CLIP-Adapter (Gao et al., 2023)**：在预训练 VLM 上添加轻量级适配器；APEX 指出其未考虑迁移难度，且适配器位置（文本端 vs 视觉端）影响显著
4. **ProGrad (Zhu et al., 2023)**：通过梯度对齐保留通用知识；APEX 采用更直接的自适应集成策略，无需额外正则化损失
5. **Tip-Adapter (Zhang et al., 2022)**：基于缓存的零样本适配方法；APEX 采用端到端微调，更适合 few-shot 场景

## 局限性与未来方向
1. **仅聚焦理解任务**：方法主要针对 CLIP、EVA-CLIP、CoCa 等理解型 VLM，未扩展到生成任务如 BLIP、LLaVA
2. **未探索其他 ETL 方法**：如 LoRA、IA³ 等参数高效微调方法的适用性有待研究
3. **手动文本提示仍有改进空间**：实验表明精心设计的 manual prompts 可能优于浅层 prompt，如何自动获取优质 prompts 是未来方向
4. **适配器初始化策略**：观察到 identity 初始化优于随机初始化，但未深入分析原因

## 研究启发与可借鉴点
1. **迁移难度的量化与自适应利用**：RTD 指标可迁移至其他 VLM 适配场景，用于设计难度感知的模型选择或集成策略
2. **模态特异性适配设计**：视觉端适合用 prompt tuning 改善特征分布，文本端适合用 adapter 调整分类边界——这一设计原则可推广至其他多模态模型
3. **预训练-微调知识的动态平衡**：自适应集成系数机制可应用于其他需要平衡泛化性与适应性的场景（如持续学习、灾难性遗忘缓解）
4. **类别距离作为难度代理**：用预训练空间中的类别距离估计迁移难度，这一思想可用于 Few-shot Learning 中的样本选择或 curriculum learning
5. **复现实验的最佳实践**：使用 Adadelta 优化器、多 seed 平均（20 seeds）可显著提升结果稳定性，值得在其他 ETL 研究中采用

## 关键术语表
- **VLMs (Vision-Language Models)**：视觉-语言模型，如 CLIP，通过大规模图文对预训练，支持 zero-shot 分类
- **ETL (Efficient Transfer Learning)**：高效迁移学习，指用少量参数（prompt/adapter）快速适配预训练 VLM 到下游任务
- **RTD (Relative Transfer Difficulty)**：相对迁移难度，基于 zero-shot CLIP 精度定义的域难度度量
- **VPT (Visual Prompt Tuning)**：视觉提示调整，在视觉编码器各层插入可学习 prompt tokens
- **TPT (Text Prompt Tuning)**：文本提示调整，在文本编码器输入端插入可学习 prompt tokens
- **TA (Text Adapter)**：文本适配器，在预训练文本编码器后添加轻量级线性变换层
- **Adaptive Ensemble**：自适应集成，根据估计的迁移难度动态混合预训练特征与微调特征
- **Base-to-Novel Generalization**：base-to-novel 泛化，在 base 类上微调后测试 on both base 和 novel 类

## 可复现要素
- **数据集**：11 个标准数据集（ImageNet、Caltech101 等），均为公开数据集
- **代码开源**：论文未明确声明代码开源链接，但提供详细算法伪代码（Algorithm 1-2）
- **关键超参数**：
  - 视觉提示层数 $J_\mathcal{V}=12$（base-to-novel）/ $J_\mathcal{V}=3$（cross-evaluation/domain gen）
  - 文本提示层数 $J_\mathcal{T}=1$
  - 每层提示数量 $b_\mathcal{V}=b_\mathcal{T}=2$
  - 低秩维度 $d_r=32$
  - 缩放因子 $\beta=4.0$
  - 阈值 $\epsilon=0.05$
  - 优化器：Adadelta，lr=0.15，batch_size=16，15 epochs（ImageNet 5 epochs）
- **权重开源**：论文未声明开源
