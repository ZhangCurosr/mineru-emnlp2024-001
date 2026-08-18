---
title: "Eyes-Don-t-Lie-Subjective-Hate-Annotation-and-Detection-with"
source: https://aclanthology.org/2024.emnlp-main.11.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:18:52"
---

# 论文速读：Eyes-Don-t-Lie-Subjective-Hate-Annotation-and-Detection-with

## 一句话总结
本文构建并开源了首个针对仇恨言论主观评估的眼动数据集 GAZE4HATE，系统验证 annotator 阅读时的眼动特征（如注视时长、瞳孔直径）与其主观仇恨评级之间的显著关联，并据此提出首个融合眼动信号的多模态仇恨言论检测基线模型 MEANION，证明眼动特征可为纯文本模型提供独立且可提升性能的互补信息。

## 研究问题与动机
1. **仇恨言论检测的高主观性与黑盒局限**：现有 HSD 模型多为神经网络黑盒，容易拟合数据集表层偏差，而人类标注本身高度依赖语境与个体背景，缺乏对主观变异的可解释建模机制。
2. **眼动数据在仇恨言论研究中的空白**：NLP 领域虽有利用眼动评估 Transformer 认知合理性的工作，但专门针对仇恨言论阅读过程的大规模眼动数据集尚属空白，难以支持细粒度主观性分析。
3. **人工 Rationale 与自然眼动的对齐关系不清**：现有可解释性研究主要依赖人工划词 rationale，但“显式标注的焦点”与“无任务干预下的自然阅读眼动”在预测主观评分及对齐模型内部机制方面的差异尚未被系统比较。
4. **多模态信号融入 HSD 的有效基线缺失**：如何将高维生理认知信号（眼动）与 (L)LM 表征有效结合，仍缺乏严谨的 baseline 与消融分析，限制了对比研究与工程落地。

## 核心贡献（创新点）
1. **发布 GAZE4HATE 数据集**：提供 90 句受控构造的德语性别相关语句（含显式/隐式仇恨及对照最小对偶）、43 名参与者的眼动轨迹、主观 Likert 评分及 token 级 rationale，共 3870 条主观标注样本；与已有工作本质区别在于首次在同一数据集中对齐“自然阅读眼动 + 显式 rationale + 主观连续评分”。
2. **系统量化眼动特征对主观仇恨评级的预测力**：通过 ANOVA + Tukey HSD 验证 6/13 项眼动特征显著区分主观仇恨类别，并发现瞳孔尺寸特征对情绪强度敏感而与非极性相关，区别于以往仅依赖句法/语义特征的主观建模工作。
3. **建立模型 Rationale 与人类行为信号的对齐诊断协议**：首次同时评估 InputXGradient、Saliency、Shapley 三种可解释方法生成的模型 rationale 与人工 rationale、眼动特征的相关性，揭示当前 HSD 模型缺失瞳孔相关的情绪唤醒信号。
4. **提出首个眼动融合 HSD 基线模型 MEANION**：将 CLS 文本嵌入、逐 token 眼动向量与 rationale BOW 拼接后输入 MLP，证明眼动特征可在多个基线 (L)LM 上稳定提升主观仇恨检测性能，且效果可与特定领域 fine-tuning 相媲美。

## 方法详解
1. **受控数据构造（Minimal Pairs）**：以德国 FEMHATE 数据集为基础，选取 20 句明确仇恨语句，通过替换关键词构造中性/积极对照句，形成显式（含仇恨词汇如 `stupid`）与隐式（词汇无害但语境冒犯，如 `Women belong in the kitchen`）两类条件，并加入中性/反男性/控制句共 90 句，严格隔离词汇与语境效应。
2. **眼动采集与主观标注流程**：43 名德语母语者使用 SR Eyelink 1000 Plus（27″ 显示器，2560×1440）按自定速度阅读（限 20 秒），记录原始眼动；阅读后依次完成：① 1-7 分 Likert 仇恨评分；② 1-5 分置信度评分；③ 点击标注支撑评分的 rationale token。顺序随机化，前 4 句为练习trial。
3. **眼动特征预处理**：提取 SR Eyelink 13 项特征（FIXATION-COUNT、DWELL-TIME、MAX/MIN/AVERAGE-FIX-PUPIL-SIZE、FIRST-RUN-FIXATION-COUNT 等）；为消除个体基线差异，按参与者独立做 min-max 归一化；缺失值按语义处理（跳过置 0、眨眼缺失取均值）。
4. **统计分析策略**：对 6 个显著特征分别执行多分类（positive/neutral/hate）与二分类（hate vs no-hate）ANOVA，并结合 Tukey HSD 进行两两比较，定位差异来源（如瞳孔特征源于 neutral vs 其余，注视特征源于 hate vs 其余）。
5. **MEANION 模型架构**：多模态输入 = CLS embedding（G-BERT/em-LLaMA2/em-Mistral） + 逐 token 眼动特征向量（补全至 max len=14） + rationale BOW 向量（sklearn CountVectorizer, N=248）。拼接后输入 MLP 分类器（grid search 调优 hidden layers、activation、solver、alpha、learning rate）。文本骨干模型 rott 在 German HateCheck 上 fine-tune 得到 rott-hc 作为强基线。

## 实验与结果
1. **数据集规模与分布**：GAZE4HATE 共 90 句 × 43 参与者 = 3870 条主观标注；按 70:10:20 进行 5-fold 分割，每 fold 含所有参与者样本但无同一句子重复。
2. **基线模型性能**：在 HuggingFace 开源 5 个德语/多语 HSD 模型（deepset、ortiz、aluru、rott、ml6）中，rott 表现最佳（hate F1=0.59），经 German HateCheck 微调后 rott-hc 达到 macro avg F1=0.68、hate F1=0.66，显著优于其他基线。
3. **眼动特征显著性**：6 个特征在多分类与二分类下均呈极显著差异（F-score 14.86~54.19, p<.001）；FIXATION-COUNT、DWELL-TIME、FIRST-RUN-FIXATION-COUNT 在二分类中 F-score 更高；瞳孔尺寸特征区分 neutral 与 hate/positive，验证其对情绪唤醒强度的敏感性。
4. **Rationale 对齐诊断**：InputXGradient 与人工 rationale 平均 Pearson r=0.335，与 DWELL-TIME/FIXATION-COUNT/FIRST-RUN-FIXATION-COUNT 相关性>0.2，而与三项瞳孔特征相关性接近 0；Shapley Value 整体对齐度最低。
5. **MEANION 消融与提升**：
   - 逐特征加入（EG）相对纯文本（E）的 macro F1 提升：BERT-base +0.03、rott-hc +0.06、em-LLaMA2 +0.02、em-Mistral +0.03。
   - Rationale BOW（ER）对 BERT-base 提升最大（+0.09），但对 LLM 有时产生负效应（最高 -0.04），组合（EGR/EGRP）性能波动。
   - McNemar 配对检验证实眼动引入的性能提升具有统计显著性；隐含仇恨检测仍是难点（rott-hc 显式 hate F1=0.68，隐式=0.65）。

## 相关工作脉络
1. **NLP 眼动可解释性（Das et al., 2016; Long et al., 2019; de Langis & Kang, 2023）**：现有工作多关注句法/语义注意力对齐，但数据集规模小且任务分散；本文填补仇恨言论专项空白，并提供更细粒度的多特征统计验证。
2. **Rationale 标注与模型诊断（Atanasova et al., 2020; DeYoung et al., 2020）**：主流依赖人工划词 rationale 评估模型；本文首次将“自然阅读眼动”与“显式 rationale”并列作为人类行为ground truth，并提出系统化的相关性对比协议。
3. **仇恨言论主观性与人口统计学建模（Waseem & Hovy, 2016; Röttger et al., 2021; Kanclerz et al., 2022）**：既往多通过多数投票或标注者人口特征缓解主观分歧；本文引入生理认知信号（眼动）作为个体偏差的动态代理，实现主观建模从静态属性向实时认知过程的迁移。
4. **认知引导的语言模型（Ding et al., 2022; Eberle et al., 2022）**：已有 CogBERT 等工作利用认知先验改进 LM；本文 MEANION 是首个将眼动特征直接注入 HSD 分类器的实证基线，并明确划定其与模型内部rationale的相关性边界。
5. **HSD 解释性方法评估（Simonyan et al., 2014; Sundararajan et al., 2017）**：本文在性别仇恨任务上验证 InputXGradient/Saliency/Shapley 三类方法的实际对齐表现，指出梯度类方法在情感/主观任务上更稳定，为后续可解释性选型提供实证依据。

## 局限性与未来方向
1. **生态效度受限**：数据为实验室受控最小对偶构造，未完全反映真实社交媒体中混杂、多模态、跨语言的仇恨言论场景。
2. **参与者池单一**：43 名德语大学生缺乏年龄、文化、语言背景的多样性，可能限制模型泛化与主观变异的覆盖范围。
3. **多模态融合较朴素**：仅采用 token 级特征拼接 + MLP，未探索时序眼动 Transformer、跨模态注意力或动态门控等先进架构。
4. **细粒度认知机制未深挖**：未系统分析“被操纵 token”在不同主观类别下的眼动轨迹差异，缺乏对显式/隐式仇恨认知负荷的对比。
5. **未来方向**：扩展参与者多样性与跨语言数据；开发时序/位置感知的眼动融合架构；将瞳孔唤醒信号作为独立监督目标以增强模型对隐性仇恨的敏感度；探索联邦学习/端侧部署以兼顾隐私与个性化内容审核落地。

## 研究启发与可借鉴点
1. **最小对偶实验设计可直接迁移**：通过控制单个词汇替换操控主观极性，是验证模型/信号对“语境依赖型”偏见敏感性的标准范式，适用于 toxic/sarcasm/bias 等高主观任务。
2. **眼动预
