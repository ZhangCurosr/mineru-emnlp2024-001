---
title: "DecorateLM-Data-Engineering-through-Corpus-Rating-Tagging-an"
source: https://aclanthology.org/2024.emnlp-main.83.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:32:20"
field: "大模型训练数据工程"
keywords: ["数据工程", "预训练语料优化", "语言模型蒸馏", "数据质量评分", "层次化标签"]
innovations: ["提出Rating-Tagging-Editing三阶段数据工程框架并蒸馏至1.2B小模型", "基于Bradley-Terry pairwise比较的8维质量评估体系", "三层层次化标签体系配合多级归一化采样策略实现领域均衡"]
benchmarks: ["C-Eval", "CMMLU", "MMLU", "AGI-Eval", "HumanEval", "MBPP", "GSM8K", "MATH", "BBH", "Domain Coverage"]
---

# 论文速读：DecorateLM-Data-Engineering-through-Corpus-Rating-Tagging-an

## 一句话总结
论文提出 DecorateLM，一种基于语言模型的数据工程方法，通过对预训练语料进行质量评分（Rating）、层次化标签（Tagging）和文本编辑（Editing）三个阶段，从 1000 亿 token 的原始语料中筛选并优化出 450 亿高质量 token，用于训练 1.2B 参数的小语言模型，显著提升下游任务性能。

## 研究问题与动机
1. **预训练语料质量难以保障**：LLM 依赖海量无监督数据，但数据规模庞大且缺乏样本级质量标注，难以确保数据质量。
2. **现有方法扩展性不足**：现有数据筛选方法缺乏细粒度标注能力，且难以在超大规模语料上保持或提升数据质量。
3. **中英文语料质量分布不均**：英文数据集（Dolma、The Pile）整体评分高且领域分布均衡，而中文数据集（BD-Wiki、CC-CN）评分低、交叉熵高，需针对性增强。
4. **小模型同样受益于高质量数据**：本文聚焦 1.2B 参数模型，验证了在较小模型上高质量数据工程的有效性和可扩展性。

## 核心贡献（创新点）
1. **提出 Rating-Tagging-Editing 三阶段数据工程框架**：首次将质量评分、层次化标签和文本编辑统一整合为可蒸馏的数据处理流水线，区别于仅关注数据筛选的 QuRating/DEITA 等工作。
2. **设计 8 维质量评估标准与 Bradley-Terry 偏好建模**：定义 Educational Value、Expertise、Fact&Trivia、Reasoning Level、Scarcity、Structural Format、Story-likeness、Subjectivity 八个维度，采用 pairwise 比较+B-T 模型生成稳定评分，解决 GPT-4 直接打分离散度高的问题。
3. **构建三层层次化标签体系（21→255→793）**：通过 GPT-4 迭代生成覆盖自然科学到社会事件的标签树，实现语料的领域多样性管理和定向增强。
4. **将 GPT-4 数据工程能力蒸馏至 1.2B SLM**：利用 MiniCPM-1.2B 分别训练 rating/tagging 模型和 editing 模型，在保持 GPT-4 近似性能的同时实现 1000 亿 token 级别的高效处理。
5. **在 100B 原始语料上验证装饰策略的有效性与经济性**：展示 DecorateLM 可将 perplexity 显著降低，且最佳组合策略（Rat. Agg.+Edit.）在多项基准上平均提升 4.1 分。

## 方法详解

**框架概览**：DecorateLM 由两个 1.2B 学生模型组成：一个联合处理 Rating 和 Tagging（因输入输出模式相似），另一个专门处理 Editing。教师模型为 GPT-4。

**Rating（评分阶段）**：
- 8 个质量维度的评估标准定义（见上文）
- 采用 Bradley-Terry 模型从 pairwise 比较中提取偏好概率，归一化到 [0, 100] 均匀评分尺度
- 训练数据：30,000 条 GPT-4 标注样本，学习率 0.00125，batch size 480，200 步收敛

**Tagging（标签阶段）**：
- 三级标签体系：21 个一级标签（如 Natural Sciences、Medical and Health）→ 255 个二级标签 → 793 个三级标签
- 通过 GPT-4 两阶段迭代对话生成完整标签树
- 层次化采样权重公式（Eq.3）：$W_{a,b,c} = \frac{N_{I=a}^{\alpha}}{\sum N_{I=i}^{\alpha}} \cdot \frac{N_{I=a,II=b}^{\beta}}{\sum N_{I=a,II=i}^{\beta}} \cdot \frac{N_{I=a,II=b,III=c}^{\gamma}}{\sum N_{I=a,II=b,III=i}^{\gamma}}$，通过指数 α,β,γ 调节分布平滑度

**Editing（编辑阶段）**：
- 目标：将网络文本改写为类似 Wikipedia 的正式表达，保留核心信息的同时提升清晰度和流畅度
- 训练数据：10,000 条 GPT-4 改写样本，600 步收敛（比 rating/tagging 更复杂）
- 评估指标：Enhanced Clarity、Text Fluency、Term Precision、Logical Coherence、Information Precision、Information Completeness

**采样策略**：
- Separate Criterion Sampling（Eq.1）：按维度优先级依次采样
- Aggregate Criterion Sampling（Eq.2）：加权聚合所有维度得分 $W_i = \sum_{t=1}^{8} k_t \cdot e^{\frac{score_{t,i}-\mu_t}{\sigma_t}}$
- 最终选取 58.5B token 中的 45B 高质量 token

## 实验与结果

**实验设置**：
- 骨干模型：MiniCPM-1.2B（预训练 800B tokens）
- 装饰语料来源：Dolma、CC-CN、C4、The Pile、BD-Wiki（共 100B tokens）
- 训练配置：20,000 steps，学习率 0.01，batch size 1200 tokens/iteration，10 机 × 8×A100-80GB

**评估基准**：C-Eval(0-shot)、CMMLU(5-shot)、AGI-Eval(5-shot)、MMLU(5-shot)、HumanEval(0-shot)、MBPP(0-shot)、GSM8K(0-shot)、MATH(4-shot)、BBH(0-shot)、ARC-E(0-shot)、ARC-C(0-shot)、TriviaQA(0-shot)，以及自定义 Domain Coverage (Avg.DC) 基准

**主要结果（Table 2）**：
- **Baseline**：平均 36.1
- **Rat.(Agg.)**：平均 38.5（↑2.4），几乎所有任务提升
- **Tag.**：平均 37.5（↑1.4），Domain Coverage +4.3
- **Edit.**：平均 38.4（↑2.3）
- **Rat.(Agg.)&Edit.**（最优）：平均 **40.2**（↑4.1），在 BBH(+4.2)、TriviaQA(+18.9)、HumanEval(+5.5) 上大幅提升
- **Rat.(Agg.)&Tag.&Edit.**：Domain Coverage 最高 45.0（↑7.5）

**关键发现**：
- English 数据集（Dolma、The Pile）初始质量高，Chinese 数据集（BD-Wiki、CC-CN）质量偏低，凸显 DecorateLM 对非英语语料的增强价值
- 编辑后文本 perplexity 显著下降，证明改写提升了模型可学习性
- 针对 MMLU 子任务定向打标的 MMLU-Tag 模型在 15/20 个子任务上优于通用 Tag.

## 相关工作脉络

1. **QuRating / DEITA / ALPAGASUS**：基于 LLM 的数据评分与筛选方法，本文与之区别在于引入多层次标签和文本编辑，且将能力蒸馏至小模型实现规模化处理。
2. **Phi-1 / MoDS**：使用 GPT-4/DeBERTa 改进教育数据和精确数据选择；本文扩展至通用预训练语料的多维度工程。
3. **INSTAG / Phi-1.5**：面向 SFT 数据的标签系统；本文标签体系面向 pretraining corpus，层级更深（3 级 793 标签），且与质量评分联动。
4. **WRAP / TinyStories**：用小数据+简单数据训练高效模型；本文强调"精选数据"而非"减少数据量"，通过装饰而非压缩提升效率。
5. **Phi-3**：两阶段训练（web + synthetic）；本文的 Editing 阶段可视为一种 synthetic data 增强，但侧重于对真实语料的改写而非纯合成。
6. **Rephrasing the Web (Maini et al., 2024)**：本文 Editing 方法的灵感来源，但以系统化框架（Rating+Tagging+Editing）和蒸馏 pipeline 为特色。

## 局限性与未来方向

1. **偏差继承风险**：GPT-4 的偏好和偏见可能通过标注数据传递给 DecorateLM。
2. **模型规模受限**：仅在 1.2B 模型上验证，需在大模型和更大规模语料（如 1.1T tokens）上重复验证泛化性。
3. **语言覆盖有限**：仅涉及英文和中文，未涵盖法语、俄语等其他语言。
4. **训练阶段局限**：仅在 decay 阶段使用装饰语料，从头训练（from scratch）的效果待验证。
5. **标签体系粒度不足**：三级标签难以完全捕捉现实内容的多样性，需更细粒度系统。
6. **采样策略简化**：当前策略未充分捕捉 rating 与 tagging 在不同任务间的 nuanced 关系，需探索更多采样策略。

## 研究启发与可借鉴点

1. **Bradley-Terry 偏好建模替代直接打分**：利用 pairwise comparison 比直接要求 LLM 输出连续分数更稳定，可作为数据质量评估的通用技巧。
2. **教师-学生双模型蒸馏架构**：将大模型的复杂数据工程能力（评分+标签 vs. 编辑）分派给不同小模型，兼顾效率与任务特异性，适用于其他数据管道构建。
3. **层次化标签的多级下采样策略**：公式(3)的逐级归一化采样思路可迁移至任何需要平衡多分类分布的数据筛选场景。
4. **Perplexity 下降作为编辑效果的间接验证**：在缺乏文本改写质量 gold standard 时，用目标模型 perplexity 变化评估编辑质量，是一个巧妙的替代指标。
5. **Domain Coverage 专项评估**：构建跨领域的 domain-specific 基准并报告 Avg.(DC)，为数据工程论文提供了可复用的评估范式。

## 关键术语表

**DecorateLM**：本文提出的数据工程框架，通过 Rating、Tagging、Editing 三个阶段对预训练语料进行质量评估、分类标注和文本改写。

**Bradley-Terry (B-T) 模型**：一种基于 pairwise 比较的偏好概率建模方法，本文用于从 GPT-4 的两两比较中生成稳定的质量评分。

**Domain Coverage (DC)**：本文自定义的跨领域覆盖评估基准，综合 SportQA、MedM-CQA、JECQA、SciQ、OpenFinData 五个领域任务，衡量模型领域均衡性。

**Separate Criterion Sampling vs. Aggregate Criterion Sampling**：前者按质量维度优先级依次采样，后者加权聚合所有维度得分采样，后者在实验中表现更优。

**MiniCPM-1.2B**：本文使用的骨干小语言模型（1.2B 参数），同时作为 DecorateLM 的蒸馏基座和训练验证对象。

**Decorated Corpus**：经 DecorateLM 处理后的语料，从 100B 原始 token 中精选出 45B 高质量 token 构成。

## 可复现要素
- **数据集**：Dolma、CC-CN、C4、The Pile、BD-Wiki（公开数据集）；装饰后语料 45B tokens 未在论文中声明公开
- **代码**：论文未提及代码开源
- **权重**：DecorateLM 模型权重未声明开源；MiniCPM-1.2B 为开源模型
- **关键超参**：学习率 0.00125，batch size 480（DecorateLM 训练）；学习率 0.01，batch size 1200 tokens/iteration（LM 训练）；AdamW 优化器；warmup 3 步，decay 每 120/5000 步；训练设备为多机 × 8×A100 GPU
