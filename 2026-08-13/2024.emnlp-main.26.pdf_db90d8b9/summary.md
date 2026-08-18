---
title: "Strength Lies in Differences! Improving Strategy Planning for Non-collaborative Dialogues via Diversified User Simulation"
source: https://aclanthology.org/2024.emnlp-main.26.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:09:11"
---

# 论文速读：Strength Lies in Differences! Improving Strategy Planning for Non-collaborative Dialogues via Diversified User Simulation

## 一句话总结
本文针对非合作对话（如谈判与说服）中现有大模型智能体难以适配多样化用户行为的问题，提出了 TRIP（Tailored stRategIc Planning）方法，通过显式建模用户心理状态与未来行为的 ToM 模块，结合基于多样人群的用户模拟器进行强化学习训练，实现了针对特定用户的定制化策略规划与泛化能力提升。

## 研究问题与动机
1. **用户特征缺失**：现有基于 LLM 的非合作对话智能体在策略规划时仅依赖对话历史，未能将用户特定的性格、决策风格等显式特征纳入规划过程，导致策略呈现“一刀切”的僵化模式。
2. **训练范式单一**：当前训练策略规划器依赖单个用户模拟器进行交互强化学习，该模拟器行为模式局限（常偏向优先满足用户满意度），无法覆盖真实场景中多样的非合作抵抗行为，致使模型面对新用户时泛化能力严重不足。
3. **评测盲区暴露缺陷**：作者建立的细粒度评估协议表明，即便是当前 SOTA 方法 PPDPP，在 20 种不同人格画像下的性能波动也极为显著，约 17.77% 的场景下优势不具统计显著性，8.88% 的场景下甚至劣于无策略的 Standard 智能体，凸显了定制化解法的必要性。

## 核心贡献（创新点）
1. 提出用户感知策略规划模块（UASP），利用 LLM 的心智理论（ToM）能力显式推断用户心智状态与未来可能行动以指导策略选择；与已有工作仅从对话历史中隐式提取信息不同，本文通过开放推理主动构建“用户当前目标+下一步预期”的显式状态表征。
2. 设计基于人群（population-based）的强化学习训练范式，利用 40 个均衡分布的人格化用户模拟器进行同步交互训练；与 PPDPP 等依赖单一固定模拟器的 oversimplified 机制不同，本范式通过环境多样性强制策略规划器学习跨人格的泛化适应力。
3. 构建覆盖 20 种人格类别（Big-Five × Decision-Making Styles）的细粒度评估协议并系统量化现有方法的适应性瓶颈；区别于以往仅汇报整体平均指标的评测惯例，本文首次通过跨人格性能方差分析揭示了 SOTA 智能体在非合作对话中的“一刀切”缺陷。
4. 在 Craigslist-Bargain 与 PersuasionForGood 双基准上实现全面领先并通过真实人类交互评测验证；与 GDP-MCTS 等侧重树搜索或单一提示工程的方案相比，本文端到端联合优化了感知建模与训练数据分布，在提升成功率的同时显著降低了平均对话轮次。

## 方法详解
