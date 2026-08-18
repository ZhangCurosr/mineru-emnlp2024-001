---
title: "Direct-Multi-Turn-Preference-Optimization-for-Language-Agent"
source: https://aclanthology.org/2024.emnlp-main.138.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:32:45"
field: "大语言模型对齐与偏好优化"
keywords: ["多轮偏好优化", "语言智能体", "直接偏好优化", "累积误差缓解", "状态-动作占据测度", "长度归一化"]
innovations: ["提出DMPO损失函数，通过SAOM约束和长度归一化将DPO扩展至多轮智能体任务", "从理论上解释了长度归一化在多轮偏好优化中消去配分函数的作用"]
benchmarks: ["WebShop", "ScienceWorld", "ALFWorld"]
---

# 论文速读：Direct-Multi-Turn-Preference-Optimization-for-Language-Agent

## 一句话总结
本文提出 **DMPO** 损失函数，通过引入状态-动作占据测度（SAOM）约束与轨迹长度归一化技术，直接优化多轮语言智能体任务中的强化学习（RL）目标，有效缓解行为克隆（BC）的累积误差问题。

## 研究问题与动机
1. **累积误差困境**：语言智能体采用行为克隆（BC）微调时，学习过程中的微小错误会在多轮交互中累积放大，导致性能显著下降。
2. **DPO 的多轮适用性不足**：现有的直接偏好优化（DPO）仅针对单轮偏好对齐设计，其推导依赖配分函数 $Z(s)$ 可消去的特性；但在多轮场景中，$Z(s)$ 依赖于当前状态，无法消去，直接应用 DPO 会导致次优性能。
3. **轨迹长度差异偏差**：在多轮任务中，偏好转折（preferred trajectory）与偏好缺失（dispreferred trajectory）往往具有不同的长度，直接比较会引入长度偏差。
4. **现有 RL 方法的复杂性**：传统基于 RL（如 PPO）的方法需要与环境持续交互且训练不稳定，而两阶段偏好学习方法（先学习奖励再优化策略）效率低下。

## 核心贡献（创新点）
1. **提出 DMPO 损失函数**：首次将 DPO 扩展至多轮智能体场景，通过替换约束条件实现 RL 目标的端到端直接优化，无需显式奖励模型或环境交互。
2. **理论揭示长度归一化的必要性**：从理论上证明，在 Bradley-Terry 模型中引入长度归一化是消去多轮场景下状态相关配分函数的关键，解释了此前 Empirical 研究中长度归一化有效的原因。
3. **引入 SAOM 约束缓解累积误差**：将传统策略约束替换为状态-动作占据测度（SAOM）约束，使策略更倾向于模仿专家的状态-动作分布，从而从理论上缓解累积误差。
4. **全面的实验验证**：在 WebShop、ScienceWorld、ALFWorld 三个多轮智能体基准上，DMPO 在干净和 noisy 设置下均显著优于 SFT、DPO、ETO 等基线方法。

## 方法详解
1. **RL 目标重构**：将 DPO 原始的基于策略约束的 RL 目标 $\max_{\pi_\theta} \mathbb{E}[\sum \gamma^t r(s_t,a_t)] - \beta D_{KL}[\pi_\theta(a_t|s_t) || \pi_{ref}(a_t|s_t)]$ 替换为基于 **状态-动作占据测度（SAOM）** 的约束：$\max_{\pi_\theta} \mathbb{E}_{(s,a)\sim d^{\pi_\theta}}[r(s,a)] - \beta D_{KL}[d^{\pi_\theta}(s,a) || d^{\pi_{ref}}(s,a)]$。
2. **配分函数的消去**：由于 SAOM $d^\pi(s,a)$ 是关于状态-动作对 $(s,a)$ 的分布，其归一化常数（配分函数 $Z$）不再依赖于特定状态 $s$，从而在后续推导中可以被消去。
3. **长度归一化机制**：在 Bradley-Terry 模型中，对偏好和偏好缺失轨迹的累计奖励分别除以归一化系数 $\frac{1-\gamma}{1-\gamma^T}$，以消除不同轨迹长度带来的偏差。
4. **DMPO 损失函数推导**：将奖励函数 $r(s,a) = \beta \log \frac{d^{\pi^*}(s,a)}{d^{\pi_{ref}}}(s,a) + \beta \log Z$ 代入带长度归一化的 BT 模型，经过推导得到最终可计算的损失函数（公式 16）：$\mathcal{L}_{DMPO} = -\mathbb{E}[\log \sigma(\sum_{t=0}^{T_w-1} \beta \phi(t,T_w) \log\frac{\pi_\theta(a_t^w|s_t^w)}{\pi_{ref}(a_t^w|s_t^w)} - \sum_{t=0}^{T_l-1} \beta \phi(t,T_l) \log\frac{\pi_\theta(a_t^l|s_t^l)}{\pi_{ref}(a_t^l|s_t^l)})]$，其中折扣函数 $\phi(t,T) = (1-\gamma^{T-t})/(1-\gamma^T)$。
5. **早期步加权特性**：DMPO 通过折扣函数 $\phi(t,T)$ 赋予早期状态-动作对更高的权重，这与多轮任务中早期决策对最终结果影响更大的直觉一致。

## 实验与结果
1. **数据集**：WebShop（1938 train, 200 test-seen）、ScienceWorld（1483 train, 194 test-seen, 241 test-unseen）、ALFWorld（3321 train, 140 test-seen, 134 test-unseen）。
2. **基线对比**：SFT、Best-of-N、RFT、PPO、ETO、DPO。基础模型为 Llama-2-7B-Chat 和 Mistral-7B-Instruct-v0.2。
3. **主要结果（Clean 设置，Llama-2-7B-Chat）**：
   - WebShop: DMPO (0.701±0.003) > ETO (0.698±0.003) > PPO (0.642) > SFT (0.631)
   - ScienceWorld: DMPO (0.724±0.005) > RFT (0.716) > Best-of-N (0.702) > SFT (0.568)
4. **主要结果（Noisy 设置）**：在两个基座模型上，DMPO 均在 Unseen 测试集和大多数 Seen 测试集上优于 DPO，证实了对噪声的鲁棒性。
5. **关键提升**：相对于 SFT，DMPO 在 WebShop 和 ScienceWorld 上分别带来约 5.2% 和 11.3% 的平均性能提升（Clean 设置），验证了其缓解累积误差的有效性。

## 相关工作脉络
1. **与 DPO (Rafailov et al., 2024) 的区别**：DPO 专为单轮偏好对齐设计，依赖配分函数 $Z(s)$ 可消去的假设；DMPO 通过 SAOM 约束和长度归一化，将 DPO 扩展至多轮序列决策场景。
2. **与 ETO (Song et al., 2024) 的联系与差异**：ETO 是首个尝试将 DPO 用于智能体的工作，但直接套用单轮 DPO 导致次优；DMPO 从理论层面修正了 DPO 在多轮场景下的缺陷。
3. **与 IPL/CPL (Hejna & Sadigh, 2023/2024) 的差异**：IPL 和 CPL 同样旨在避免显式奖励学习，但其损失函数仅适用于等长轨迹对；DMPO 通过长度归一化支持不等长轨迹。
4. **与模仿学习（SAOM 约束）的关联**：本文借鉴了传统模仿学习中使用 SAOM 约束缓解累积误差的思想（Abbeel & Ng, 2004; Ho & Ermon, 2016），并将其融入基于偏好的直接优化框架。
5. **与 SimPO (Meng et al., 2024) 的关联**：SimPO 在实验中观察到长度归一化有效，但未提供理论解释；本文从配分函数消去的角度给出了严格的理论推导。
6. **与 PPO/RL 方法的对比**：PPO 等方法需要与环境交互并可能不稳定；DMPO 作为离线偏好优化方法，无需在线交互，训练更简单稳定。

## 局限性与未来方向
1. **回合级任务表述的限制**：当前方法仅考虑回合级（turn-wise）任务，导致奖励信号稀疏；未来可探索 token-wise 任务表述以获得更稠密的奖励信号（参考 Rafailov et al., 2024a）。
2. **实验规模局限**：实验仅在 7B 参数规模的模型和模拟数据集上进行；未来需要在更大规模的模型和更复杂的真实世界数据集上验证方法的泛化能力。
3. **理论最优性的近似**：论文承认在动态语言环境中，由于状态转移函数的内在约束，无法严格优化任意奖励函数下的 RL 目标，DMPO 是一种良好的近似。

## 研究启发与可借鉴点
1. **SAOM 约束的工程化应用**：将强化学习中的状态-动作占据测度概念引入大语言模型偏好优化，为缓解累积误差提供了新的理论工具和实践路径，可迁移至其他序列决策任务。
2. **长度归一化的理论 justification**：本文不仅使用了长度归一化，还给出了其有效性的理论解释（消去状态相关配分函数），这种“先实验观察，后理论解释”的研究范式值得借鉴。
3. **超参数 $\gamma$ 的灵活调节策略**：论文发现较小的 $\gamma$ 有助于抵抗噪声轨迹，较大的 $\gamma$ 有助于从高质量轨迹中学习后期步骤，这种根据数据质量动态调整超参数的策略具有通用参考价值。
4. **从单轮到多轮的损失函数扩展思路**：通过识别单轮方法在多轮场景下的核心失效机制（配分函数不可消去），并针对性地修改约束条件（策略→SAOM），这一方法论可用于其他单轮算法的多轮扩展。
5. **简单有效的实验设计**：通过 Clean/Noisy 两种设置系统性地评估方法的鲁棒性，并结合消融实验（超参数分析、长度分析）全面验证方法各组成部分的贡献，实验设计严谨且易于复现。

## 关键术语表
**DMPO (Direct Multi-Turn Preference Optimization)**：本文提出的新型损失函数，用于直接优化多轮语言智能体任务的 RL 目标，无需显式奖励模型。
**SAOM (State-Action Occupancy Measure)**：状态-动作占据测度，描述智能体在策略 $\pi$ 下访问状态-动作对的分布，用于替代策略约束以缓解累积误差。
**配分函数 (Partition Function)**：$Z(s)$，在 RL 理论中用于归一化最优策略的函数；在多轮场景中它依赖于状态 $s$，是 DPO 直接应用的障碍。
**累积误差 (Compounding Errors)**：在序列决策中，学习者的微小错误随时间步累积放大，导致实际轨迹严重偏离专家轨迹的现象。
**Bradley-Terry (BT) 模型**：一种用于建模成对偏好概率的统计模型，DPO 和 DMPO 均基于此模型将奖励转化为偏好概率。
**行为克隆 (Behavioral Cloning, BC)**：通过监督学习将智能体策略拟合到专家轨迹的方法，易受累积误差影响。
**直接偏好优化 (Direct Preference Optimization, DPO)**：一种通过偏好数据直接优化策略的算法，避免了传统 RL 中奖励模型的训练，但最初仅适用于单轮场景。
**长度归一化 (Length Normalization)**：在计算轨迹奖励时除以归一化系数 $\frac{1-\gamma}{1-\gamma^T}$，以消除不同长度轨迹间的偏差。

## 可复现要素
- **数据集**：WebShop、ScienceWorld、ALFWorld 均为公开基准，统计数据在论文 Table 1 中给出。
- **代码/权重**：论文未明确声明代码开源状态，未提及预训练权重共享。
- **关键超参数**：$\beta \in \{0.1, 0.2, ..., 0.9\}$，$\gamma \in \{0.1, 0.2, ..., 0.9, 0.99\}$；SFT 阶段 batch size=64，学习率从 $\{1e-5, 2e-5, 3e-5\}$ 中选择；DMPO 阶段 batch size=32。
- **硬件**：8× NVIDIA A100 GPUs。
- **基座模型**：Llama-2-7B-Chat、Mistral-7B-Instruct-v0.2。
