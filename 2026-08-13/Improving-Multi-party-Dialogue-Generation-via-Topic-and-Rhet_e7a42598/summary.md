---
title: "Improving-Multi-party-Dialogue-Generation-via-Topic-and-Rhet"
source: https://aclanthology.org/2024.emnlp-main.189.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:21:56"
---

# 论文速读：Improving-Multi-party-Dialogue-Generation-via-Topic-and-Rhet

## 一句话总结
针对多方对话生成中现有方法过度依赖 reply-to 结构而忽视响应与目标话语在话题与逻辑层面连贯性的缺陷，本文提出 RL-TRC，通过引入话题连贯与修辞连贯的辅助任务及三类话语感知奖励的强化学习策略，显著提升了生成响应与目标话语的语义对齐与篇章一致性。

## 研究问题与动机
- **目标话语位置不固定与话题交织**：多方对话中存在多条并行且交叉的话题流，单纯建模 reply-to 拓扑结构无法让模型理解当前轮次应聚焦的真实话题，易导致“跑题”。
- **现有方法缺乏响应-目标话语的对齐保证**：SOTA 方法（如 EMMDG、MADNet）虽能利用 reply-to 结构生成相关响应，但实际样例中常出现话题错位（target 讨论 Opera 却回复 Firefox）或内部逻辑矛盾（“i use opera, but i don't use it”）的问题。
- **连贯性建模视角局限**：既有工作多将连贯性局限于单轮 persona 一致性（通过 NLI 或 VAE 消除冲突），未针对多方对话中“生成响应 ↔ 目标话语”这一动态对偶关系设计篇章级建模。
- **离散评价难以直接微分优化**：话题相关性与修辞合理性难以用标准交叉熵直接优化，需借助强化学习将不可微的篇章评估信号转化为训练梯度。

## 核心贡献（创新点）
- **提出 RL-TRC 框架**，首次在多方对话生成中同时从话题（topic）和修辞（rhetorical）两个篇章维度建模生成响应与目标话语的连贯性，填补了结构建模之外的语义对齐空白。
- **设计话题连贯任务（Topic Coherence Task）**，利用 ChatGPT 提取关键词并结合 PMI 构建话题连贯矩阵，显式引导模型预测与目标话语话题一致的响应关键词分布。
- **设计修辞连贯任务（Rhetorical Coherence Task）**，借助外部 discourse parser 识别响应与目标话语间的 discourse relation，通过分类头监督模型学习逻辑结构感知。
- **构建三种 discourse-aware 奖励（R_tc / R_rc / R_rt）**，将话题分类概率、修辞关系 KL 散度与 reply-to 识别 KL 散度加权融合，结合 actor-critic 策略端到端优化生成质量。

## 方法详解
- **编码器**：采用 BART-base 编码对话历史，输入格式为 `[CLS][SEP]p_1u_1[SEP]...[RT]p_t u_t[SEP]...[SEP]p_r`，其中 `[RT]` 标记目标话语，各话语语义 $h_{ui}$ 取自其前 `[SEP]` 的隐藏状态。
- **话题连贯任务**：用 ChatGPT 提取每条话语关键词（≤5个）。计算目标话语与黄金响应关键词间的点互信息 $PMI(w_i,w_j)=\log\frac{p(w_i,w_j)}{p(w_i)p(w_j)}$，构建话题连贯矩阵并选取 top-10 PMI 词得到连贯语义 $E_{ck}$。通过注意力融合 $\mathbf{h}_{ut}$ 与 $E_{ck}$ 预测生成关键词概率 $P_a$，以交叉熵 $\mathcal{L}_
