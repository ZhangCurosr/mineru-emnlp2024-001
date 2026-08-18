---
title: "Learning-Planning-based-Reasoning-via-Trajectories-Collectio"
source: https://aclanthology.org/2024.emnlp-main.20.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:28:26"
field: "大语言模型推理与强化学习"
keywords: ["过程奖励模型", "DPO", "推理-as-规划", "离线模拟", "偏好优化", "大语言模型推理", "逻辑推理"]
innovations: ["提出离线模拟合成过程奖励的框架，将规划式推理转化为可学习的偏好优化问题", "设计PRM训练流程，通过结果监督信号估计中间状态价值，避免人工标注和在线搜索开销", "构建pDPO方法，在双正确轨迹间利用过程奖励差值构造偏好对，提升推理过程质量"]
benchmarks: ["LogiQA-v2", "ReClor", "GSM8K", "MATH"]
---

# 论文速读：Learning-Planning-based-Reasoning-via-Trajectories-Collectio

## 一句话总结
本文提出了一种**将规划式推理转化为离线学习问题**的框架：通过收集LLM生成的轨迹、基于部分轨迹探索离线模拟估计中间状态的过程奖励，训练过程奖励模型（PRM），再用合成过程奖励指导Direct Preference Optimization（DPO）训练策略模型。在7B参数规模下，该方法在逻辑推理 benchmarks 上超越了 GPT-3.5-Turbo 和 Mixtral-8×7B 等强基线，同时显著降低了对人工过程标注的依赖。

## 研究问题与动机
1. **LLM 推理过程存在幻觉和逻辑缺陷**：LLM 能生成看似合理的多步推理链，但中间步骤可能出现错误推导或冗余内容，导致"结论正确但过程不合理"。
2. **现有过程监督成本高昂**：Lightman et al. 的工作需要人工专家对中间步骤进行标注，成本远高于最终答案标注，难以扩展到大规模 LLM 训练。
3. **在线规划搜索延迟高、开销大**：RAP 等框架将推理建模为 MDP，通过 MCTS 在推理时在线搜索最优路径，但需频繁调用 LLM/验证器评估中间状态，带来巨大推理延迟和计算开销。
4. **如何在离线阶段利用结果监督信号合成高质量过程奖励，替代在线搜索和人工标注**，是本文要解决的核心问题。

## 核心贡献（创新点）
1. **提出离线模拟合成过程奖励的框架**：不同于 RAP 的在线 MCTS 搜索，本文通过从种子轨迹中采样中间状态并多次重试补全，利用结果监督信号（是否答对）估计中间步骤的期望回报，避免在线评估开销。
2. **设计过程奖励模型（PRM）训练流程**：将过程奖励估计问题建模为 K 分类任务，用交叉熵损失训练 PRM，缓解直接模拟估计的噪声问题，比启发式搜索更高效鲁棒。
3. **构建过程监督偏好优化（pDPO）**：在 vanilla DPO（仅基于最终答案正确性构建偏好对）基础上，进一步利用 PRM 计算的轨迹级累积奖励，对两个都答对的轨迹构造高质量偏好对（需奖励差超过阈值 σ），使模型学会生成更可靠的中间推理步骤。
4. **在逻辑推理和数学推理上验证有效性**：7B 模型（Llama2-7B-pDPO）在 LogiQA-v2 和 ReClor 上分别达到 55.5% 和 61.7%，超越 GPT-3.5-Turbo；Gemma-2B-pDPO 在 GSM8K 上达到 52.8%，超越 Gemma-7B-Instruct。

## 方法详解

### 3.1 自然语言推理的形式化定义
将推理任务建模为 MDP：轨迹 $\tau = \{s_0, a_0, ..., s_T, a_T\}$，其中 $a_t \sim \pi_\theta(a|c_t)$，$s_{t+1} \sim \pi_\theta(s|a_t, c_t)$，$c_t$ 为历史轨迹。最终结果奖励为：
$$r_f(\tau, y) = \begin{cases} 1, & \text{if } \tau \to y \\ 0, & \text{else} \end{cases}$$
目标是最大化期望结果奖励：$\arg\max_\theta \mathbb{E}_{x,y \sim \mathcal{D}, \tau' \sim \pi_\theta} r_f(\tau', y)$。

### 3.2 基于离线模拟的过程奖励估计
- **步骤1**：用 LLM 收集完整解轨迹作为种子轨迹。
- **步骤2**：从轨迹中采样中间推理步骤（action $a_t$ 或 state $s_t$）作为非叶节点。
- **步骤3**：以中间状态为起点，让 LLM 重试 $K$ 次完成推理，统计成功到达正确答案的轨迹数作为期望回报估计：
$$r_e(\tau_{t,a}, y) = \sum_{k=1}^{K} r_f(\tau^{k|\tau_{t,a}}, y)$$
- 核心假设：若某中间状态能高频到达正确答案，则该状态包含通向结论的关键信息。

### 3.3 过程奖励模型（PRM）训练
将估计的过程奖励作为标签，训练分类器 $f_{prm}: \mathcal{X} \times \mathcal{T} \to \mathbb{R}^K$，最小化交叉熵损失：
$$\mathcal{L}_{step} = -\log p_r, \quad p = f_{prm}(x, \tau)$$
动机：(1) 避免 MCTS 式逐个中间状态评估带来的巨大时间开销；(2) 训练 PRM 比直接使用含噪模拟值更鲁棒。

### 3.4 轨迹级奖励与偏好数据集构建
轨迹级奖励为各步骤 PRM 预测概率的累积乘积（超过阈值 $C$ 的类概率之和相乘）：
$$r_p(\tau) = \prod_{t}^{T} \prod_{*}^{\{a,s\}} \sum_{i \geq C}^{K} f_{prm}(\tau_{t,*})_i$$
构建两类偏好对：
- **结果偏好对** $\mathcal{D}_o$：正确 vs 错误轨迹。
- **过程偏好对** $\mathcal{D}_p$：两个都答对但过程奖励差超过 $\sigma$ 的轨迹对，奖励高的为 chosen，低的为 rejected。
- 总损失：$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}; \mathcal{D}_o \cup \mathcal{D}_p)$

### 3.5 Direct Preference Optimization (DPO)
损失函数：
$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(\tau_w|x)}{\pi_{ref}(\tau_w|x)} - \beta \log \frac{\pi_\theta(\tau_l|x)}{\pi_{ref}(\tau_l|x)}\right)\right]$$
关键超参：$\beta=0.1$（逻辑推理）/ $0.5$（数学推理），$C=2$（逻辑）/ $3$（数学），$\sigma=0.5$（LogiQA-v2）。

## 实验与结果

### 数据集
- **逻辑推理**：ReClor（训练集 4,638 题）、LogiQA-v2（训练集 12,567 题），均为多选格式。
- **数学推理**：GSM8K、MATH（使用 MetaMath 子集训练）。

### 基线对比
- 基础模型：Llama2-70B-Chat、Mixtral-8×7B-Instruct、GPT-3.5-Turbo、GPT-4-Turbo
- SFT、vanilla DPO、IPO、RFT、ReST-EM、Process PPO、GRPO

### 主要结果
| 模型 | LogiQA-v2 (Test) | ReClor (Test) |
|------|------------------|---------------|
| GPT-3.5-Turbo | 45.4 | 53.7 |
| Mixtral-8×7B-Instruct | 49.5 | 56.7 |
| Llama2-7B-SFT | 44.5/45.5 | 48.8/53.4 |
| Llama2-7B-DPO | 53.1 | 60.4 |
| **Llama2-7B-pDPO** | **55.5** (+2.4) | **61.7** (+1.3) |
| Iter-1-pDPO | 57.3 | 61.8 |
| Iter-1-process GRPO | 57.3 | 61.7 |

- **数学推理**（Gemma-2B）：pDPO 在 GSM8K 上达 **52.8%**，超越 Gemma-7B-Instruct（46.4%）；DeepSeekMath-7B-pDPO 在 MATH 上达 **46.8%**，超越 vanilla DPO（46.3%）。
- **训练效率**：DPO 类方法 < 16 小时（4×H100），PPO/GRPO > 40 小时。
- **低资源场景**：仅 40% 标注（3,234 题）时，pDPO 即显著超越 SFT 基线；且与全量 DPO 性能相当（53.5 vs 53.9）。
- **GPT-4 自动评估**：pDPO 在合理性（52.5% 胜率）、简洁性（~60% 更紧凑）、逻辑一致性三项均优于 vanilla DPO，整体约 67.8% 样本质量更优。

## 相关工作脉络
1. **RAP (Reasoning-as-Planning, Hao et al., 2023)**：将推理建模为 MDP，用 MCTS 在线搜索最优路径。本文与其本质区别在于：RAP 依赖在线评估中间状态（高延迟），本文通过离线模拟+PRM 学习将规划转化为学习问题，避免推理时搜索开销。
2. **Process Reward Model (Lightman et al., 2023)**：使用人工标注的 step-level feedback 训练 PRM。本文创新点在于：完全避免人工标注，通过结果监督信号离线估计过程奖励，大幅降低成本。
3. **DPO / IPO (Rafailov et al., 2023; Azar et al., 2023)**：仅基于最终答案正确性构建偏好对。本文扩展为同时利用过程奖励，在两个都答对的轨迹间构造高质量偏好对，使模型学会生成更合理的中间步骤。
4. **Rejection Sampling Fine-tuning (RFT, Yuan et al., 2023) / ReST-EM (Singh et al., 2023)**：基于结果过滤/重采样。本文的 pDPO 在过滤基础上进一步引入过程奖励排序，筛选出推理过程更优的样本。
5. **MATH-Shepherd (Wang et al., 2023)**：并发工作，同样采用离线模拟合成过程奖励。差异在于：MATH-Shepherd 聚焦数学推理且主要用 PRM 排序验证或 PPO 训练；本文聚焦逻辑推理并通过 PRM 构建偏好对用于 DPO，训练成本更低、过程更稳定。
6. **GRPO (Shao et al., 2024)**：基于组的相对策略优化。本文 pDPO 达到与 GRPO 相当甚至更好的性能，但训练时间仅为后者的 1/3 左右，且避免了 critic model 逼近分布的困难。

## 局限性与未来方向
1. **资源消耗仍然较大**：离线模拟需要大量 LLM 推理生成补全轨迹，限制了在更大模型（如 70B+）和更长上下文任务（如代码生成）上的实验。
2. **依赖初始策略模型能力**：若 base model 能力不足（如 Gemma-2B 在 MATH 上效果不佳），模拟阶段引入的噪声会影响过程奖励估计质量。
3. **未来方向**：作者提出探索从多角度利用弱监督信号合成过程奖励，进一步减少对人工结果标注的依赖，实现持续自我改进。

## 研究启发与可借鉴点
1. **"规划→学习"的范式转换**：将在线搜索型推理（MCTS/beam search）转化为离线偏好学习问题，是一种通用思路——凡是需要在线评估/搜索的任务，都可尝试用少量模拟数据训练替代模型来逼近搜索效果，从而大幅降低推理延迟。
2. **过程奖励的合成策略**：通过多次重试补全中间状态、统计成功率的方案，为无标注过程的强化学习提供了一个低成本的 PRM 训练方案，可迁移至代码生成、工具调用等长程推理任务。
3. **pDPO 的偏好对构建技巧**：在两个都答对的样本中，用过程奖励差值 $\sigma$ 筛选高质量偏好对，相比单纯的正确/错误二分类，能挖掘更多细粒度信号，这一技巧可推广至其他 DPO 应用场景。
4. **低资源高效训练**：仅用 40% 标注数据即可达到接近全量 DPO 的性能，结合 PRM 训练仅用 10% 数据的事实，表明该方法在数据稀缺场景下具有显著优势，适合标注成本高的专业领域推理任务。

## 关键术语表
- **Reasoning-as-Planning (RAP)**：将多步推理建模为马尔可夫决策过程（MDP），通过搜索算法（如 MCTS）在推理空间中寻找最优路径的框架。
- **Process Reward Model (PRM)**：对推理轨迹中每个中间步骤/状态赋予奖励分数的模型，用于过程监督而非仅依赖最终答案。
- **Direct Preference Optimization (DPO)**：一种无需显式奖励模型的偏好优化算法，直接将偏好数据转化为策略损失进行端到端训练。
- **pDPO (process DPO)**：在 vanilla DPO 基础上引入过程奖励，在两个均答对的轨迹间基于过程质量构造额外偏好对的改进方法。
- **Offline Simulation**：不在线进行搜索，而是预先从种子轨迹中采样中间状态并多次重试补全，用结果反馈估计中间状态价值的方法。
- **Trajectory Collection**：从 LLM 采样多条完整推理轨迹作为训练数据的基础步骤，为后续中间状态采样和 PRM 训练提供素材。
- **ReAct Format**：交替使用 Thought（推理）和 Action（选择事实/规则）的推理输出格式，本文在此基础上进行评估。
- **GRPO (Group Relative Policy Optimization)**：通过同一 query 的多组采样计算组内相对优势，用于策略优化的强化学习算法。

## 可复现要素
- **数据集**：LogiQA-v2、ReClor、GSM8K、MATH、MetaMath（开源数据集，训练子集由作者裁剪）
- **代码/权重**：论文未明确声明代码开源，未提供模型权重下载链接
- **关键超参**：$\beta=0.1$（逻辑推理）/ $0.5$（数学推理），$C=2$（逻辑）/ $3$（数学），$\sigma=0.3\sim0.5$（依数据集），每步重试 $K=10$，温度 $0.7$
- **硬件**：NVIDIA A100/H100
- **训练时长**：DPO 类方法 < 16 小时（4×H100），PPO/GRPO > 40 小时
