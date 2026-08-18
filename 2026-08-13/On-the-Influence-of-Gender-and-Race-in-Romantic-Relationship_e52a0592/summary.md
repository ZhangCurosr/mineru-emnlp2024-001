---
title: "On-the-Influence-of-Gender-and-Race-in-Romantic-Relationship"
source: https://aclanthology.org/2024.emnlp-main.29.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:13:35"
field: "NLP公平性与偏见评估"
keywords: ["bias", "relationship prediction", "gender bias", "racial bias", "name replacement", "heteronormativity", "LLM fairness"]
innovations: ["受控名字替换实验揭示LLM在浪漫关系预测中的异性恋规范偏见", "发现亚裔名字导致系统性低召回率并通过嵌入分析验证性别信息缺失机制", "提出性别vs种族信息影响的分离评估框架"]
benchmarks: ["DDRel"]
---

# 论文速读：On-the-Influence-of-Gender-and-Race-in-Romantic-Relationship

## 一句话总结
论文通过受控名字替换实验，系统评估大型语言模型（LLM）在浪漫关系预测任务中是否存在异性恋规范偏见（heteronormative bias）及种 racial 偏见，发现模型对同性别角色组合及涉及亚裔名字的配对预测为浪漫关系的召回率显著更低。

## 研究问题与动机
- LLM在进行双人关系推断时，是否会基于名字所暗示的性别、种族等人口统计属性产生系统性偏见？
- 现有偏差研究多聚焦个体刻板印象（如职业性别偏见），缺乏对LLM推断两人之间社会关系的偏见评估。
- 美国同性婚姻和跨种族婚姻直到2015年和1967年才全国合法化，LGBTQIA+群体和跨种族伴侣仍面临歧视，模型偏见可能加剧此类社会不平等。
- 电影剧本数据集DDRel的测试集中不存在同性别浪漫关系对话样本，限制了对此类偏见的直接评估能力。

## 核心贡献（创新点）
- 首次通过控制变量名字替换实验系统揭示LLM在关系预测中的异性恋规范偏见，证明同性别角色配对被预测为浪漫关系的可能性显著低于不同性别配对。
- 发现涉及亚裔名字的种内/种间配对在浪漫关系预测中表现显著更差（亚裔-亚裔召回率仅0.68），揭示模型对亚裔群体的系统性偏见。
- 通过上下文嵌入逻辑回归分析证明：亚裔名字在模型表征中编码的性别信息最少，导致模型难以利用性别身份进行关系推断，为偏见成因提供机制解释。
- 提出性别 vs 种族信息影响的分离评估框架（匿名化+部分匿名化实验），量化发现性别关联对预测的影响远强于种族/民族关联。
- 将偏见分析延伸至社会影响层面，讨论模型行为对LGBTQIA+群体代表性和资源分配（如广告推荐）的 representational 和 allocational harm。

## 方法详解
- **任务定义**：给定双人对话 $C = ((S_1, u_1), \ldots, (S_n, u_n))$，预测角色A和B的关系类型（13类预定义标签），将 Lovers、Spouse、Courtship 合并为"浪漫关系"，其余为"非浪漫关系"。
- **模型**：Llama2-7B/13B-chat（官方实现）和 Mistral-7B-Instruct（HuggingFace），zero-shot提示，temperature=0，max generation length=512，nucleus sampling。
- **数据集筛选**：使用 DDRel 测试集，筛选327个原角色为不同性别的对话（手动标注），其中271个不含显式性别线索（无 "sir"/"ma'am"/"father" 等词）。
- **名字替换实验（性别）**：为每种族收集30个名字，按 SSA 出生性别统计分为10个非线性等分箱（0–100%女性占比），覆盖中性名字；对所有不同名字对进行替换，分析保留/交换性别时的召回率变化。
- **名字替换实验（种族）**：收集80个强种族+强性别指示名字（4种族×2性别），90%阈值（白人70%），进行种内/种间配对替换分析。
- **匿名基线**：用 "X" 和 "Y" 替换名字，获取无名字信息的纯上下文基线。
- **嵌入分类分析**：从 Llama2-7B 提取每个名字在15个浪漫+15个非浪漫对话中的上下文嵌入（共209,800个），训练 logistic regression 分类性别（按种族）和种族（One-vs-All），5次随机划分取平均。
- **评估指标**：主要报告浪漫关系预测的平均召回率（recall），辅以 precision、F1、accuracy（附录）。

## 实验与结果
- **性别配对实验（Llama2-7B，Figure 2）**：同性别配对（男-男、女-女）召回率显著低于不同性别配对；女-女配对召回率高于男-男，表明对男性同性关系偏见更强；亚裔和 Hispanic 名字的性别辨别差距较小。
- **种族配对实验（Figure 3）**：涉及亚裔名字的配对召回率显著降低，亚裔-亚裔最低（0.68）；非亚裔的种内/种间配对差异不显著，未发现强烈跨种族偏见。
- **匿名基线（Table 2）**：Llama2-7B recall=0.6887，Llama2-13B recall=0.3019，Mistral-7B recall=0.2028；名字替换后recall显著偏离匿名基线，证明性别信息显著影响预测。
- **嵌入分类准确性（Table 1）**：性别分类——Asian 53.3%、Black 96.4%、Hispanic 80.5%、White 99.9%；种族分类——Asian 97.6%、Black 70.5%、Hispanic 89.5%、White 94.2%。
- **最强结果**：Llama2-7B在不同性别/种族设置下precision维持在0.78–0.84，但recall差异显著（0.68–0.86）。

## 相关工作脉络
- An et al. (2023, 2024) 研究名字偏见对社会常识推理和招聘决策的影响，本文将其框架扩展至关系预测的下游任务评估。
- Wang et al. (2022)、Sandoval et al. (2023)、Wan et al. (2023) 关注名字频率和人口统计属性对LLM输出的影响，本文聚焦关系推断场景中名字隐含信息的利用方式。
- Jia et al. (2021) 提出 DDRel 数据集用于双元对话关系分类，本文在其基础上进行系统性偏见评估。
- Maudslay et al. (2019)、Shwartz et al. (2020) 研究名字偏差和语境化表征中的名字 artifact，本文提供偏见在关系预测任务中的实证表现。
- Stewart & Mihalcea (2024) 研究机器翻译中对同性关系的偏见，与本文在NLU领域形成呼应。
- Blodgett et al. (2020) NLP偏见综述，本文在其 representational/allocational harm 框架下讨论社会影响。

## 局限性与未来方向
- Prompt敏感性和in-context学习：不同提示格式可能影响结果，未来需系统研究提示工程对偏见的影响。
- 名字覆盖不足：受限于数据源，难以覆盖所有种族和性别认同（尤其非二元性别）。
- 数据源偏差：DDRel测试集缺乏同性别浪漫关系对话，无法直接评估模型在此类场景的真实表现。
- 隐性性别线索：对话中可能存在难以检测的隐性性别暗示（如性取向暗示），可能混淆分析。
- 电影剧本语言风格：角色性别与语言风格的不一致可能引入统计噪声，但实验显示影响有限。

## 研究启发与可借鉴点
- 受控名字替换实验设计可通过系统化变量控制揭示LLM隐式偏见，此方法可直接迁移至情感分析、意图识别、关系抽取等其他NLU任务。
- 上下文嵌入逻辑回归分类为偏差溯源提供可量化的分析工具，可用于检验模型内部表征是否编码敏感属性。
- 匿名化+部分匿名化的对比实验设计可分离不同人口统计属性的影响权重，为多维度偏见分析提供范式。
- 可与团队在对话理解/关系抽取方向结合，评估模型对LGBTQIA+内容的理解能力，开发包容性评估基准。
- 中性名字箱的设计可扩展至非二元性别研究，推动更全面的公平性评估。

## 关键术语表
**Heteronormative bias（异性恋规范偏见）**：假设并偏好传统性别角色、异性恋关系和核心家庭的认知偏见，边缘化其他性别表达和性取向。
**DDRel**：Dyadic Dialogue Relationship数据集，用于从电影剧本对话中预测角色间关系类型（13类）。
**Contextualized embeddings（上下文嵌入）**：语言模型根据对话上下文为每个名字生成的动态向量表示，编码人口统计属性信息。
**Intra/Inter-racial pairing（种内/种间配对）**：指相同种族或不同种族的角色名字组合。
**Recall（召回率）**：在所有实际为浪漫关系的样本中，被模型正确预测为浪漫关系的比例；本文主要关注此指标以检测对少数群体的系统性遗漏。
**Name-replacement experiment（名字替换实验）**：通过替换对话中的角色名字来控制性别/种族变量，观察模型预测变化的因果推断方法。
**Representational harm（代表性伤害）**：模型对特定群体的错误表征导致的边缘化和社会可见性降低。
**Allocational harm（分配性伤害）**：模型偏见导致资源或机会的不平等分配（如广告推荐排除）。

## 可复现要素
- 数据集：DDRel（Jia et al., 2021），测试集约610条对话，需从 https://github.com/jiaqi-/DDRel 获取
- 代码：论文未明确提及代码开源
- 模型权重：Llama2-7B/13B-chat 和 Mistral-7B-Instruct 通过 HuggingFace 获取
- 关键超参：temperature=0，max generation length=512，nucleus sampling，5次随机划分
- 名字数据源：Rosenman et al. (2023) 用于种族分类，SSA Social Security Application dataset 用于性别分类
- 提示模板：Figure 4 in appendix
