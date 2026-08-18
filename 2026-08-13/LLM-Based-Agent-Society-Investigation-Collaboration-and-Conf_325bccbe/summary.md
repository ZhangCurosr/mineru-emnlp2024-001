---
title: "LLM-Based-Agent-Society-Investigation-Collaboration-and-Conf"
source: https://aclanthology.org/2024.emnlp-main.7.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:22:43"
field: "多智能体社会行为模拟"
keywords: ["LLM Agent", "Social Deduction Game", "Avalon", "Multi-Agent System", "Social Behavior", "Collaboration", "Confrontation"]
innovations: ["首次系统量化LLM代理在社交推理游戏中的六大社会行为", "提出包含记忆-分析-规划-行动-学习的端到端Avalon智能体框架"]
benchmarks: ["CGAgent", "Win Rate", "Quest Engagement Rate", "Failure Vote Rate"]
---

# 论文速读：LLM-Based-Agent-Society-Investigation-Collaboration-and-Conf

## 一句话总结
本文以"Avalon"社交推理游戏为测试平台，提出了一套完整的LLM多智能体框架，系统性地研究了LLM代理在不完全信息博弈中的协作与对抗社会行为，揭示了代理能够自发涌现出符合人类策略的社会互动模式。

## 研究问题与动机
- **核心问题**：现有LLM代理研究多聚焦诚实、合作等正面社会行为，对负面社会行为（如欺骗、对抗）缺乏深入探究。
- **评估空白**：已有工作仅基于有限游戏特征设计代理框架，对社会推理游戏的复杂社交动态分析不足。
- **方法局限**：先前基于RL或规则的方法难以捕捉LLM代理在复杂社交场景中的自然语言交互能力。
- **研究价值**：理解LLM代理的社会行为对其在真实社交场景（如信息传播、冲突解决）中的应用具有重要意义。

## 核心贡献（创新点）
- **全面的社会行为分析框架**：首次系统性地从团队协作、领导力、说服、伪装、对抗和信息共享六个维度量化分析LLM代理的社会行为。
- **端到端的Avalon智能体框架**：设计了包含记忆存储、记忆摘要、分析、规划、行动和响应生成及经验学习的完整模块链，优于传统基线方法。
- **角色自适应策略生成**：不同角色（Merlin、Percival、Morgana等）能够根据信息优势和阵营目标动态调整行为策略，而非采用统一模式。
- **自发性行为涌现验证**：证明无需显式指令，LLM代理即可自发展现伪装（Camouflage）、对抗（Confrontation）等复杂社会行为。
- **经验学习机制**：通过自我角色策略学习与他人角色策略学习，代理能够在多局游戏中持续优化决策能力。

## 方法详解
框架包含六大核心模块，遵循人类思维流程设计：

1. **Memory Storage（记忆存储）**：为每个代理维护独立的记忆池，记录结构化对话历史（角色名、自然语言回复、轮次、公开/私有标记），解决LLM输入长度限制问题。

2. **Memory Summarization（记忆摘要）**：每轮结束后使用摘要提示压缩历史，更新记忆：$\mathcal{M}_t = \langle \mathrm{SMR}(\mathcal{M}_{t-1}), (\mathcal{R}_t^{p_1} \cdots, \mathcal{R}_t^{p_6}, \mathbb{Z}_t) \rangle$

3. **Analysis（分析模块）**：分析其他玩家的角色身份与潜在策略：$\mathcal{H}_t^{p_i} = \mathrm{ANA}(\mathcal{M}_t, \mathcal{RL}^{p_i})$，帮助代理理解盟友与竞争对手。

4. **Planning（规划模块）**：基于记忆、分析结果、目标和初始策略生成动态计划：$\mathcal{P}_t^{p_i} = \mathrm{PLAN}(\mathcal{M}_t, \mathcal{H}_t^{p_i}, \mathcal{P}_{t-1}^{p_i}, \mathcal{RI}^{p_i}, \mathcal{G}^{p_i}, \mathcal{S}^{p_i})$

5. **Action（行动模块）**：执行五类行动——选择玩家、投票、完成任务、非语言信号、保持沉默，决策过程对主持人和其他玩家保密。

6. **Experience Learning（经验学习）**：包含**自我角色策略学习**（基于历史生成三条角色策略建议并融合）和**他人角色策略学习**（总结其他玩家的策略进行迁移）。

系统提示包含三要素：**Role Information**（角色名称与介绍）、**Goal**（胜利条件）、**Abstracted Strategy**（初始玩法规则）。

## 实验与结果
- **数据集/环境**：6人制Avalon游戏，分为Good Side（Merlin、Percival、Loyal Servant ×2）和Evil Side（Morgana、Assassin）。
- **模型**：gpt-3.5-turbo-16k，temperature=0.3，提取器temperature=0，每轮生成3条策略建议。
- **基线**：CGAgent（Xu et al., 2023a），原为狼人杀代理改造。
- **主要结果**：
  - **Good Side**: 90%胜率（基线60%）
  - **Evil Side**: 100%胜率（基线60%）
  - **Quest Engagement Rate**: 我方40.3% vs 基线33.1%
  - **Failure Vote Rate**: 我方84.0% vs 基线36.5%
- **消融实验**：
  - 无Analysis模块：Good/Evil胜率均降至60%
  - 无Strategy Learning：Good胜率50%，Evil胜率60%
  - 无Planning模块：Good胜率降至80%
  - 无Action模块：Evil胜率降至80%
- **结论**：Analysis和Strategy Learning模块对双方胜率影响最大；Evil侧的Planning和Action模块尤为关键。

## 相关工作脉络
- **Generative Agents (Park et al., 2023)**：模拟人类社会行为，但未涉及欺骗/对抗等负面社会行为，本研究填补此空白。
- **Plan4MC (Yuan et al., 2023)**：面向Minecraft的规划代理，缺乏社会行为分析维度。
- **GITM (Zhu et al., 2023)**：开放世界多智能体，专注于任务完成而非社会动态。
- **RGAgent (Akata et al., 2023)**：重复博弈中的LLM策略建模，未涉及不完全信息社交推理游戏。
- **CGAgent (Xu et al., 2023a)**：狼人杀代理基线，仅部分分析行为，本工作扩展至Avalon并实现更全面的行为量化。
- **ReCon (Wang et al., 2023c)**：递归沉思对抗虚假信息，但未系统分析团队协作与对抗行为。

## 局限性与未来方向
- **高成本与慢速**：每次交互需多次访问模型，导致成本高且交互速度慢。
- **不合理行为分布**：代理存在过度自我披露（Self-Disclosure）等问题，需进一步优化。
- **规模限制**：当前仅验证6人制Avalon，未扩展到更多玩家或不同游戏环境。
- **未来方向**：优化效率、探索其他社交推理游戏、深化动态社会交互理解。

## 研究启发与可借鉴点
- **模块化社会行为分析框架**：六维社会行为量化指标（LAR、QER、FVR等）可直接迁移至其他多智能体社交场景评估。
- **记忆压缩机制**：SMR模块的摘要策略可有效缓解长对话上下文管理问题，适用于任何需要历史跟踪的多轮对话系统。
- **自涌现行为验证方法**：通过对比有/无学习模块的行为分布差异，验证社会行为的自发性，此实验设计值得借鉴。
- **阵营差异化策略设计**：Good/Evil两侧采用不同模块重要性的发现提示我们：多智能体系统中应针对角色异质性设计差异化能力模块。
- **经验学习的双路径设计**：自我学习+他人学习的双轨机制为多智能体协同优化提供了可复用范式。

## 关键术语表
- **Avalon**：社交推理游戏，玩家分属Good/Evil阵营，通过讨论、投票和完成任务进行对抗，Good需完成3个任务，Evil需破坏3个任务或识别Merlin。
- **Leader Approval Rate (LAR)**：衡量领导力的指标，表示提案被通过的票数占比。
- **Quest Engagement Rate (QER)**：玩家参与任务小组的轮次占比，反映主动性和影响力。
- **Failure Vote Rate (FVR)**：投反对票的比例，Evil侧通过此指标 sabotage 任务。
- **Camouflage**：伪装行为，Evil侧隐藏真实身份或假装Good侧身份。
- **Self-Role Strategy Learning**：基于自身游戏历史生成并优化角色专属策略的学习机制。
- **Other-Role Strategy Learning**：总结其他玩家策略并进行跨角色迁移的学习机制。
- **Social Deduction Game (SDG)**：社交推理游戏，玩家通过不完全信息下的语言交互判断彼此身份。

## 可复现要素
- **数据集**：Avalon游戏（开源规则，非标准数据集）
- **代码**：https://github.com/3DAgentWorld/LLM-Game-Agent（已公开）
- **模型**：gpt-3.5-turbo-16k（商业API）
- **关键超参**：temperature=0.3（生成），temperature=0（提取），策略建议数=3
