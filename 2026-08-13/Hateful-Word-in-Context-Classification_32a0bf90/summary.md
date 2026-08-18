---
title: "Hateful-Word-in-Context-Classification"
source: https://aclanthology.org/2024.emnlp-main.10.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:20:42"
---

# 论文速读：Hateful-Word-in-Context-Classification

## 一句话总结
本文提出了 HateWiC 新任务与数据集，聚焦于特定语境中目标词是否表达仇恨的细粒度分类；通过联合建模词sense描述性定义与标注者人口统计特征，系统揭示了仇恨词义中客观指称与主观评价的交互机制，弥补了现有仇恨言论检测研究在词汇级主观消歧方向的空白。

## 研究问题与动机
- **研究粒度缺失**：现有 HSD 研究 overwhelmingly 集中于 utterance/句子级别分类，对同一词汇在不同语境中“仇恨 vs 非仇恨”的消歧建模严重不足。
- **通用LM的敏感度局限**：主流预训练语言模型擅长通用词义，但对领域特定、非标准或新兴仇恨表达缺乏敏感性，易忽略随着社会事件演化产生的 novel hate usages。
- **主客观语义交织**：仇恨词义并非纯描述性（descriptive），而是位于“描述-表达”连续体上，其仇恨属性高度依赖语境内容、交际意图与读者身份（即 hate-heterogeneous senses），传统单一多数视角标注会抹除这种主观差异性。
- **已有词级工作局限**：现有贬义词消歧（如 Dinu et al., 2021）或 dog whistle 检测研究多采用单一视角标注、规模较小，且未系统考察主观标注分歧与人口统计因素的影响。

## 核心贡献（创新点）
1. **提出 HateWiC 任务与新数据集**：将仇恨检测粒度下探至 word-in-context 级，构建约 4000 条实例、每条含 3 次独立主观评分的词级数据集；与既有 utterance 级 HSD 工作的本质区别在于聚焦“同一词义在不同语境下的仇恨属性动态变化”。
2. **解耦描述性与主观性输入通道**：首次在同一分类框架中并行引入 sense 定义嵌入（Wiktionary 原始定义与 FLAN-T5 生成定义）与标注者人口统计嵌入，揭示二者在不同主观强度场景下的互补机制。
3. **提出双向评估协议**：同时设计 Majority label prediction 与 Individual annotator label prediction 两个任务，并配合 Random split 与 OoV (unseen terms) split 划分策略，全面量化模型在常规泛化与零样本词汇泛化上的表现。
4. **发现 hate-heterogeneous 现象与定义失效边界**：实证证明词典定义整体有效，但在同一定义下同时存在仇恨与非仇恨多数标注的异质场景（占比约 8% 的 sense）中，定义嵌入会导致准确率暴跌近一半，而标注者人口统计信息可部分补偿该性能损失。

## 方法详解
- **Sense 表示构建**：使用三种编码器（BERT-base、HateBERT、WSD Biencoder）提取目标词在例句中的位置向量（WiC embeddings，取最后隐藏层对应子词位置后平均池化）。
- **定义嵌入**：
  - `Def`：直接编码 Wiktionary 提供的 sense 定义句子。
  - `T5Def`：使用 FLAN-T5 Base 按模板 `[SENTENCE] What is the definition of [TERM]?` 生成上下文相关定义，再同样池化编码。
- **主观性建模（Ann）**：为捕获个体差异，将标注者人口统计信息按模板 `“Reader is [AGE], [GENDER] and [ETHNICITY].”` 编码为 Ann embeddings，并与 sense embeddings 拼接（如 WiC+Def+Ann）。
- **分类器**：采用四层 MLP（隐藏层维度 300→200→100→50，learning rate 0.0005，max_iter 10，通过 GridSearchCV 在 dev 集调参）。
- **LLM 基线**：使用 7B LLaMA 2 进行 zero-shot 分类，提示词明确要求判断目标词在给定句中是否表达对某人群/个人的仇恨，max_new_tokens=10。
- **评估设置**：10-fold cross-validation，80/10/10 划分；支持按句子随机划分（Random）与按词汇划分（OoV）；同时在 DINU1/DINU2 外部数据集上做跨集验证。

## 实验与结果
- **数据集统计**：HateWiC 共 3845 个含明确多数标签的实例，二元分布均衡（仇恨 47.2%，非仇恨 52.8%）；二元标注一致率 60%（Krippendorff’s α = 0.45），三类一致率 51.3%（α = 0.33）；识别出 319 个 hate-heterogeneous sense 定义。
