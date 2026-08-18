---
title: "Successfully Guiding Humans with Imperfect Instructions by Highlighting Potential Errors and Suggesting Corrections"
source: https://aclanthology.org/2024.emnlp-main.42.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:08:43"
field: "视觉-语言导航与人机协作"
keywords: ["hallucination detection", "vision-language navigation", "uncertainty communication", "human-AI collaboration", "grounded instruction generation", "synthetic data generation"]
innovations: ["首次证明长程任务中语言型不确定性沟通（错误高亮+修正建议）可显著提升人类导航表现", "提出双模型级联的幻觉检测与类型分类架构，结合对比学习与合成数据", "设计按需揭示候选修正的交互界面，避免概率数值的误读与信息过载"]
benchmarks: ["R2R", "Matterport3D"]
---

# 论文速读：Successfully Guiding Humans with Imperfect Instructions by Highlighting Potential Errors and Suggesting Corrections

## 一句话总结
论文提出 HEAR（Hallucination DEtection And Remedy）系统，通过在视觉导航指令中高亮潜在幻觉错误并给出修正建议，使人类用户能在语言模型可能产生不准确指令的情况下仍高效完成导航任务；80名用户实验表明，相比仅提供原始指令，HEAR 使成功率提升 13%、导航误差降低 29%。

## 研究问题与动机
- **语言模型在陌生情境下不可避免地会出错**（Kalai & Vempala, 2024; Wu et al., 2023），传统研究聚焦于"如何提高模型自身准确性"，而本文关注"如何让模型在出错时仍能成功辅助人类决策"。
- **现有不确定性沟通研究主要面向分类任务与数值概率**（Vodrahalli et al., 2022; Nizri et al.），缺乏对长程序列决策（如视觉导航）中语言型不确定性的研究。
- **前序幻觉检测工作（Zhao et al., 2023b）仅做错误检测，不提供修正建议，也未设计用户界面或开展真人评估**，因此无法直接帮助人类用户纠正指令。
- **HRI 研究中 Bansal et al. (2021) 发现 AI 解释不一定带来互补性能提升**，本文试图通过谨慎选择与信息呈现方式证明：合理的系统辅助可以显著优于单独的人类或 AI。

## 核心贡献（创新点）
1. **首次证明不确定性沟通对长程任务的有效价值**：在语言引导视觉导航这一多步决策设定下，通过错误高亮与修正建议显著改善人类用户表现（SR +13%，DIST -29%），而此前工作几乎都只关注短程分类任务。
2. **HEAR 双模型级联架构（检测 + 分类）实现可操作的幻觉纠正**：先判断短语是否为幻觉（intrinsic/extrinsic），再给出候选修正并排序，相比单模型方案在 R@3 指标上获得更好收益（Dev R@3=88.4 vs. One-stage 82.7）。
3. **一种规则 + LLM 混合的合成幻觉数据生成方法**：利用 GPT-3.5-turbo / GPT-4 改写方向短语、从 Matterport3D 词表生成房间/物体幻觉、从同/异指令插入句子构造 extrinsic 幻觉，共合成 164,939 对检测样本与 117,357 对分类样本，显著降低对人标注的依赖。
4. **面向真人导航任务的交互界面设计**：以二进制高亮替代易误读的置信度数值、仅在用户主动点击时弹出 Top-3 候选修正，既避免信息过载又保留用户探索空间，并引入 Check 按钮用于终点验证。

## 方法详解
- **问题设定**：给定目标路线 $r = (o_1, a_1, \dots, o_l, a_l)$，训练 speaker 模型 $S(\pmb{w} \mid r)$ 生成自然语言指令 $\pmb{w}$；指令中常出现与场景不一致的短语，即 hallucination。
- **幻觉分类**：
  - **Intrinsic hallucination**（需替换）：描述与路线矛盾（如"The door on the right"但实际在左侧）。
  - **Extrinsic hallucination**（需删除）：描述的路径在场景中不存在（如多余的 "Walk through the office"）。
- **Hallucination Detection 模型 $P_H$**：基于 Airbert（Guhur et al., 2021），输入为 $(r, \pmb{w}, i, j)$，用 `[BH]...[EH]` 包裹候选短语，输出分数 $s(x)$，置信度 $\sigma(s(x))$；采用 contrastive learning 训练。
- **Hallucination-Type Classification 模型 $P_I$**：同架构，但在已判定为幻觉的短语上预测是否为 intrinsic；评分函数：
  $$R(\hat{\pmb{x}}) = \begin{cases} P_I(z=1 \mid \pmb{x}, y_{\pmb{x}}=1) \cdot P_H(y=1 \mid \hat{\pmb{x}}) & \text{replacement} \\ P_I(z=0 \mid \pmb{x}, y_{\pmb{x}}=1) & \text{deletion ([REMOVE])} \end{cases}$$
- **候选生成**：房间/物体幻觉候选来自 Matterport3D 全部标签（平均每个 47.6 个）；方向幻觉候选由 GPT-4 生成（平均每个 5.9 个）。
- **界面策略**：每句指令最多高亮 3 处；阈值在手动标注 Dev 集上调以最大化 F-1；用户点击高亮短语才展示 Top-3 下拉建议。

## 实验与结果
- **数据集与基线**：Matterport3D + R2R（训练 4,675 条路线）；人类评估使用 Zhao et al. (2023a,b) 的 75 条 Test 路线中的 18 条；80 名 MTurk 用户（每条路线 5 人）；5 组对照：No communication、HEAR (no suggestion)、HEAR、Oracle (no suggestion)、Oracle。
- **内嵌评测**（Table 1）：
  - HEAR：Dev F-1=63.4 / R@3=88.4；Test F-1=66.5 / R@3=70.6
  - One-stage HEAR：Test R@3=86.2 但 F-1 较低（60.9），参数量减半可能是原因
  - HEAR-SameEnvSwap 在 F-1 上最优（Test 69.1），但 R@3 偏低（78.7）
- **人类导航主结果**（Table 4 / Figure 3）：
  - No communication：SR=68.9%，DIST=6.6m，Checks=2.9
  - HEAR：**SR=77.8%（↑8.9pp vs. No comm；相对提升约 13%）**，DIST=4.6m（↓1.9m，约 29%），Checks=4.1
  - Oracle：SR=87.8%，DIST=2.7m，接近上限
  - 主观问卷（Table 2）：HEAR 用户在 "easy to follow"（4.0）和 "confident"（4.2‡）上显著优于 No comm（p<0.004），且 mental burden 无显著上升（3.5 vs. 3.6）。
- **关键结论**：即便检测与修正均不完备（66.5% F-1），其"互补性"+"允许用户探索"的机制仍显著提升最终表现；Check 按钮点击增加表明用户更愿意继续尝试，而非放弃。

## 相关工作脉络
1. **Grounded instruction generation**：Anderson et al. (2018)、Fried et al. (2018b) 开创 vision-language navigation；Zhao et al. (2023a) 揭示模型指令与人类质量的差距，但止步于评价，未处理错误。
2. **Hallucination detection**：Dale et al. (2023)、Zhou et al. (2021) 聚焦机器翻译/摘要；Zhao et al. (2023b) 将检测迁移到视觉导航，但仅做 token/phrase 标注，无修正能力与用户评估。
3. **Uncertainty communication in HRI**：Vodrahalli et al. (2022)、Nizri et al. 探讨概率校准对决策的影响；Benz & Rodriguez (2023) 论证非校准概率也可能改善表现；以上工作均以分类任务为主。
4. **Human-AI complementary performance**：Bansal et al. (2021) 认为 AI 解释通常无法带来互补增益；本文给出反例——精心设计的错误提示与信息呈现能带来显著提升。
5. **Visual-language pretraining**：Guhur et al. (2021) 的 Airbert 提供房屋场景预训练底座；Majumdar et al. (2020) 的对比学习训练范式是本工作的基础。
6. **Interface design for navigation**：Ku et al. (2021) 的 Pangea 与 Zhao et al. (2023a) 的用户交互框架是本工作界面的直接继承。

## 局限性与未来方向
- 人类评估规模受限（80 人、525 次导航），优先保证每位用户对每路线的重复测量，牺牲了路线数量。
- 认知负荷仅通过自陈 Likert 量表评估，作者承认"不够 robust"，计划未来采用 NASA-TLX 等标准化工具。
- 缺少热身练习环节，用户仅观看 2 分钟视频教程，可能引入界面熟悉度噪声。
- 行为分析粒度有限：未记录 Check 按钮点击时间点，无法刻画用户在导航过程中的动态纠错轨迹。
- 性能提升的归因难以完全拆解——无法区分究竟多少收益来自"高亮本身"、多少来自"候选修正的质量"，因评估仅看终点距离。
- 未来可探索：更细粒度的眼动/按键日志分析；把合成数据生成方法迁移至其他视觉-语言任务（如文档理解、医疗报告生成）；研究"不完美但透明"的 AI 助手在不同用户群体（如专家 vs. 新手）中的差异化效果。

## 研究启发与可借鉴点
1. **"不完美但可沟通"的 AI 辅助范式**：不必追求模型端 100% 准确，而是通过错误提示、修正候选、探索自由度设计来提升最终任务成功率；这对当前大模型部署具有强烈现实参考价值。
2. **合成幻觉数据的规则+LLM 混合生成管线**：从 R2R 指令出发，通过同环境/同指令对象置换、GPT 改写方向、跨指令句子插入三类扰动构建正负样本，这种"程序化错误注入 + LLM 语义改写"的模式可直接复用到其他 grounded instruction 或 captioning 数据的评估增强中。
3. **信息呈现的"按需揭示"策略**：用高亮而非概率数值呈现不确定性，并通过用户主动点击触发候选列表，既降低认知负荷又保留探索空间——可作为 HRI 系统 UI 设计的通用准则。
4. **Check 按钮作为"隐性探索激励"的发现**：提供纠错能力反而让用户更愿意多尝试、不轻易放弃，提示我们在设计交互式 AI 时应重视"容错"对任务坚持度的正向外部性。
5. **与团队方向的潜在结合**：若团队研究方向涉及多模态生成、幻觉检测、人机协作或长程决策，本文的双模型级联 + 对比学习 + 合成数据管线是一组可直接移植的方法组件。

## 关键术语表
- **HEAR**：Hallucination DEtection And Remedy，本文提出的用于高亮视觉导航指令中潜在幻觉并提供修正建议的系统。
- **Intrinsic hallucination（内在幻觉）**：指令中描述的对象/方向与实际路线不一致、需被替换的短语。
- **Extrinsic hallucination（外在幻觉）**：指令中描述了路线上根本不存在的对象/动作、需被删除的短语。
- **Airbert**：Guhur et al. (2021) 在 Airbnb 家居图像-文本对上进行域内预训练的双模态视觉-语言编码器，本工作以其为底座微调。
- **Contrastive learning objective**：本工作用于训练检测与分类模型的损失函数，使正确短语与错误短语在表征空间中分离。
- **R2R（Room-to-Room）**：Anderson et al. (2018) 构建的室内视觉导航数据集，提供路线与人类标注的自然语言指令对。
- **Matterport3D**：包含真实住宅 RGB 图像的 3D 模拟器，是本文实验环境的底层数据源。
- **Success Rate (SR) / Navigation Error (DIST)**：R2R 标准评估指标，分别指终点距真实目标 ≤3m 的比例与欧氏距离误差。

## 可复现要素
- **数据集**：R2R（Anderson et al., 2018）训练/验证/测试集 + Matterport3D 场景；论文声明将发布合成数据集与人类交互数据（MIT License），但截至论文发表时**未提供现成公开链接**。
- **代码/权重**：论文未提供 GitHub 仓库或模型权重下载链接，仅声明基于 PyTorch 1.7.1、Huggingface Transformers 4.5.1 实现。
- **关键超参**（Table 3）：
  - Learning rate = $10^{-5}$；Batch size = 128；Optimizer = AdamW
  - Training iterations = $5 \times 10^5$；Max instruction length = 60
  - Image feature size = 2048；Embedding dropout = 0.1；Hidden size = 768
  - Transformer layers = 12；Transformer dropout = 0.1
  - 参数量 ≈ 250M；单卡 RTX A4000 训练时间 ~72h
- **GPT 提示词**：附录 A.1 给出 GPT-3.5-turbo（生成方向幻觉）与 GPT-4（生成候选修正）的完整 prompt，可复现。
