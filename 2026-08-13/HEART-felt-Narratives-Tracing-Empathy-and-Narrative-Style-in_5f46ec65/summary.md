---
title: "HEART-felt-Narratives-Tracing-Empathy-and-Narrative-Style-in"
source: https://aclanthology.org/2024.emnlp-main.59.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:20:33"
field: "计算社会科学/自然语言处理"
keywords: ["叙事共情", "LLM叙事分析", "HEART分类体系", "结构方程模型", "情感生动性", "众包研究"]
innovations: ["提出HEART理论驱动的叙事风格分类体系，首次系统化量化叙事风格对共情的影响", "验证GPT-4在复杂主观叙事特征提取上达到人类专家水平，优于传统词典方法", "揭示vividness of emotions通过narrative transportation影响共情的实证路径及个性化机制"]
benchmarks: ["HEARTful Stories Dataset", "EMPATHICSTORIES", "EMPATHICSTORIES++"]
---

# 论文速读：HEART-felt-Narratives-Tracing-Empathy-and-Narrative-Style-in

## 一句话总结
论文提出了 HEART（Human Empathy and Narrative Taxonomy）理论驱动的叙事风格分类体系，利用 GPT-4 等 LLM 提取个人故事中的叙事元素，并通过 N=2,624 人的大规模众包研究量化了叙事风格（特别是情感生动性和情节容量）对共情的影响路径。

## 研究问题与动机
1. **现有研究偏重内容而非风格**：叙事共情研究多关注故事内容本身和读者特征，但对"故事如何被讲述"（叙事风格）如何影响共情的理解仍不充分。
2. **传统方法难以捕捉复杂风格特征**：先前的词频/词典方法（如 LIWC）只能提取简单语言特征，无法捕捉 plot shifts、vividness of emotions 等抽象叙事技巧。
3. **LLM 在叙事分析中的潜力未被验证**：虽有少数工作探索 LLM 用于叙事理解，但对其能否可靠提取叙事风格元素、以及如何支撑社会行为学洞察的研究仍缺失。
4. **缺乏系统性的叙事风格-共情分类框架**：现有理论框架零散，缺少一个融合心理学与文学研究的系统化分类体系。

## 核心贡献（创新点）
1. **提出 HEART 理论驱动的分类体系**：整合 Keen、van Krieken 等人的叙事心理学理论，构建了涵盖 Character Identification、Plot、Point of View、Setting 四大类共 12+ 个叙事元素的分类框架。*区别于以往零散的叙事特征研究，本文提供了一个统一的理论框架。*
2. **验证 LLM 在叙事风格提取上的性能**：首次系统评估 GPT-4 和 Llama 3 提取 HEART 分类元素的性能，发现 GPT-4 在多数特征上与人类专家标注高度一致（如 Character vulnerability ρ=80.15%）。*区别于仅用词典的方法，LLM 能捕捉复杂、主观的风格特征。*
3. **发布 HEARTful Stories Dataset**：基于 EMPATHICSTORIES/EMPATHICSTORIES++ 构建了 874 篇个人叙事数据集，并通过 N=2,624 人的众包研究标注了共情反应。*这是首个大规模个人叙事共情数据集。*
4. **揭示叙事风格影响共情的实证路径**：通过结构方程模型（SEM）发现，vividness of emotions 通过 narrative transportation 影响共情；character development 和 plot volume 直接关联高共情故事；共情具有高度个性化特征。*区别于以往仅预测共情分数的工作，本文揭示了具体的作用机制。*

## 方法详解
1. **HEART 分类体系构建**：
   - **Character Identification**（7 个元素）：Flatness/roundness（角色扁平/圆润）、Emotional subject（情感表达及生动性）、Cognitive subject（认知过程）、Moral subject（道德评价）、Action subject（角色行动）、Subject perception（感知与身体感觉）、Temporal references（时间指向）
   - **Plot**（3 个元素）：Plot volume（情节容量/事件频率与重要性）、Emotion shifts（情绪转变）、Resolution（冲突解决）
   - **Point of view**：视角（如第一人称）
   - **Setting**：环境描写的生动性
2. **LLM 提取**：使用 GPT-4 和 Llama 3 8B Instruct，配合为人类标注员设计的相同 codebook 和提示词，对 50 篇抽样故事进行评分（Likert scale）。采用 Krippendorf's alpha（KA）、pairwise agreement（PPA）、Spearman's ρ 评估与人类专家的一致性。
3. **LLM vs. 词典基线**：选取 4 个可与 LIWC-22 映射的特征（Optimistic tone、Cognition、Vivid emotions、Character vulnerability），对比 GPT-4/Llama 3 与 LIWC 在相关性与人类标注一致性上的表现。
4. **众包共情研究**：N=2,624 名 Prolific 参与者阅读 874 篇故事，每篇至少被 3 人独立评分。测量工具包括 State Empathy Scale（共情）、TS-SF（narrative transportation）、SITES/TEQ（trait empathy）等，同时收集人口统计信息。
5. **结构方程模型（SEM）**：使用 semopy 库，将 narrative style → narrative transportation → empathy 的路径建模，并纳入 reader characteristics（trait empathy、相似经历）的交互效应。

## 实验与结果
- **数据集**：874 篇过滤后的个人叙事（来自 EMPATHICSTORIES + EMPATHICSTORIES++），经伦理审查豁免。
- **LLM 性能**：GPT-4 在 Character vulnerability（KA=62.89, ρ=80.15%）、Optimistic tone（ρ=68.06%）、Resolution（ρ=61.59%）等特征上与人类高度一致；Llama 3 整体低于 GPT-4 但在部分特征（如 Cognition ρ=52.89%）上表现更优。LIWC 在 Cognition 上优于 GPT-4，但在 Character vulnerability 上显著差于 GPT-4（p<0.001）。
- **共情分析发现**：
  - **Vividness of emotions** 显著提升 narrative transportation（ρ 路径显著），进而影响共情；
  - **Character development** 和 **Plot volume** 在高共情故事中显著更多（Mann-Whitney u-test, p=0.03）；
  - 同一故事的不同读者间共情评分标准差显著大于零（p<0.001），显示共情的**高度个性化**；
  - **Trait empathy** 与 **vividness of emotions** 存在显著交互效应（est=0.252, p<0.001）：高特质共情者对情感生动性的反应更强。
- **最强结果**：GPT-4 在 Character vulnerability 提取上与人类一致性最高（ρ=80.15%）；vividness of emotions 是共情的最强预测路径之一。

## 相关工作脉络
1. **Roshanaei et al. (2019)**：使用 LIWC 词典量化情感词汇与共情关系。本文扩展至更复杂的叙事风格特征，并验证 LLM 在多数维度上优于 LIWC。
2. **Zhou et al. (2021)**：基于 interdependent thinking 和 integrative complexity 预测共情。本文提供更细粒度的叙事风格分类并揭示具体作用路径。
3. **Michelmann et al. (2023)**：证明 LLM 可近似人类做叙事事件分割。本文聚焦于更抽象的叙事风格元素（如情感生动性），而非事件边界。
4. **Shen et al. (2023, 2024)**：EMPATHICSTORIES 数据集的构建者。本文在其基础上引入叙事风格标注和共情众包研究，从"共情相似度建模"扩展到"风格→共情路径分析"。
5. **Keen (2006)**：叙事共情理论奠基者，提出 characterization、narrative situation 等概念。本文将其操作化为可计算的分类体系。
6. **van Krieken et al. (2017)**：character identification 的语言线索框架。本文在此基础上扩展为包含 Plot、Point of View、Setting 的完整 HEART 体系。

## 局限性与未来方向
1. **部分叙事特征标注一致性低**：Bodily sensations 和 Evaluations 的 KA 较低（人体感知和评价类特征主观性强），被排除在实证分析之外；未来可尝试更细粒度的频次标注。
2. **样本人口多样性不足**：Prolific 众包参与者以白人为主，限制了结果的普适性；需在更多样化的人群和叙事类型中验证。
3. **未让参与者评价多篇故事**：单次阅读限制了 within-subject 分析深度；未来可增加重复测量。
4. **统计建模侧重可解释性而非预测性能**：SEM 用于揭示机制而非优化预测；未来可探索将叙事特征融入复杂 transformer 模型以提升共情预测。
5. **LLM 存在系统性偏差**：GPT-4 倾向于高估 Evaluations 和 Cognition，混淆情感反应与评价/认知过程；Llama 3 对 imagery 类特征敏感度较低。

## 研究启发与可借鉴点
1. **LLM 作为复杂叙事特征提取器**：论文完整展示了如何用 LLM + 精心设计的 codebook 替代/辅助人工标注高主观性文本特征，这一范式可迁移到其他人文计算任务（如文学分析、话语研究）。
2. **结构性因果路径分析方法**：将叙事风格 → narrative transportation → empathy 的路径建模，并引入 reader characteristics 作为调节变量，为 NLP 中的社会行为洞察提供了可复用的 SEM 分析框架。
3. **"共情个性化"的发现对模型设计有启发**：同一故事引发的高度个体差异提示，未来共情预测模型应放弃单一平均目标，转而建模个性化的 reader-text interaction。
4. **完整的标注协议开源**：论文的 Appendix C 提供了详细的 codebook 和 prompt 模板，可直接借鉴或微调用于其他叙事分析任务。
5. **数据集构建流程参考**：从已有语料库（EMPATHICSTORIES）出发 → 过滤规则（去除有害内容、过短文本）→ 抽样标注 → 众包评估，这一流程对构建新的社会计算数据集有参考价值。

## 关键术语表
**HEART Taxonomy**：Human Empathy and Narrative Taxonomy，本文提出的理论驱动叙事风格分类体系，包含 Character Identification、Plot、Point of View、Setting 四大类。
**Narrative Transportation**：叙事沉浸/叙事运输，指读者在心理上"进入"故事世界、与叙事高度融合的认知-情感状态，是共情的重要中介变量。
**State Empathy Scale**：状态共情量表，测量读者在阅读当前故事时的即时共情反应（12 项 Likert 题项）。
**Trait Empathy (SITES/TEQ)**：特质共情，读者相对稳定的共情倾向，通过单项目量表或多伦多共情问卷测量。
**Krippendorf's Alpha (KA)**：克龙巴赫-阿尔法的推广形式，用于衡量多标注者之间的一致性，适用于连续或有序数据。
**Structural Equation Modeling (SEM)**：结构方程模型，一种结合因子分析和路径分析的统计方法，用于检验理论模型中潜变量之间的因果关系。
**LIWC-22**：Linguistic Inquiry and Word Count，广泛应用于心理学和计算语言学的词典工具，将文本归类为数千个心理/语言类别。
**Narrative-reader Interaction Effects**：叙事-读者交互效应，指读者特征（如特质共情、相似经历）与叙事特征之间的交互作用对共情的影响。

## 可复现要素
- **数据集**：HEARTful Stories Dataset（874 篇个人叙事 + 共情标注 + 叙事风格标注），论文声明将公开发布（"We make our dataset publicly available"）。
- **基础语料**：EMPATHICSTORIES（1,500 篇）和 EMPATHICSTORIES++（500 篇）为公开数据集。
- **代码/权重**：论文未明确提供开源代码，但 Appendix C 提供了完整 prompt 模板和 codebook；使用 GPT-4 和 Llama 3 8B Instruct。
- **关键超参**：每篇故事至少 3 名独立读者评分；LLM 评分采用 codebook 定义的 Likert scale；统计检验使用 Mann-Whitney u-test + Benjamini-Hochberg 校正（9 项比较）。
