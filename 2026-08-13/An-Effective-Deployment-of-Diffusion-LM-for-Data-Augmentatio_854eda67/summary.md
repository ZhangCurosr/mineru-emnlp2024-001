---
title: "An-Effective-Deployment-of-Diffusion-LM-for-Data-Augmentatio"
source: https://aclanthology.org/2024.emnlp-main.109.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:16:54"
field: "低资源自然语言处理"
keywords: ["数据增强", "扩散语言模型", "情感分类", "低资源NLP", "文本分类"]
innovations: ["基于扩散LM重构强标签相关token实现多样化且一致性的伪样本生成", "提出标记感知噪声调度与标记感知提示策略平衡多样性与一致性"]
benchmarks: ["SMP2020-EWECT", "India-COVID-X", "SenWave", "SST-2"]
---

# 论文速读：An-Effective-Deployment-of-Diffusion-LM-for-Data-Augmentatio

## 一句话总结
论文提出 DiffusionCLS 方法，利用扩散语言模型重构强标签相关 token 进行情感分类的数据增强，在低资源场景（域特定、标签不平衡、few-shot）下生成多样性与一致性兼具的伪样本，显著提升分类性能。

## 研究问题与动机
- **低资源情感分类的挑战**：域特定场景（如灾害、疫情）、标签分布不平衡、数据稀疏及 few-shot 条件下，PLM 性能显著下降。
- **现有文本数据增强方法的不足**：多数基于规则的 CTR 方法（如 GENIUS）只重述次要 token，依赖预训练知识引入域外噪声；基于标签生成方法（如 LAMBADA）仅以标签为条件，忽略原文上下文，导致一致性差。
- **关键情感 token 未被充分利用**：情感分类中强情感 token 对序列整体情感起关键作用，现有方法保留这些关键 token 不动，限制了样本多样性。
- **多样性与一致性难以兼顾**：增加 mask 比例可提高多样性但破坏一致性，需要找到最优平衡点。

## 核心贡献（创新点）
- **提出 DiffusionCLS 数据增强框架**：利用扩散 LM 重构强标签相关 token，与 GENIUS 等方法仅重述次要 token 形成本质区别，同时保障多样性与一致性。
- **设计 Label-Aware Noise Schedule**：基于 [CLS] token 注意力分数为 token 分配重要性权重，指导扩散模型优先恢复弱相关 token，再恢复强情感 token，与 DiffusionBERT 不同在于显式引入标签感知机制。
- **提出 Label-Aware Prompting 策略**：将标签拼接到 masked 序列中作为条件信息，解决关键 token mask 后条件生成一致性下降的问题。
- **设计 Noise-Resistant Training 方法**：结合对比损失与交叉熵损失，通过扩大不同标签样本在语义空间的间隔来缓解伪样本噪声的影响。

## 方法详解
- **Label-Aware Noise Schedule**：使用微调后的 TC 代理模型，计算每个 token 与 [CLS] token 的注意力分数 $w_i = \frac{1}{H}\sum_{h=1}^{H}s_i^h$ 作为重要性权重；token i 在第 t 步被 mask 的概率为 $q_t^i = 1 - \frac{t}{T} - \lambda \cdot S(t) \cdot w_i$，其中 $S(t) = \sin\frac{t\pi}{T}$，λ 为超参数；引入吸收态（absorbing state）机制，弱相关 token 先被 mask，强情感 token 后被 mask。
- **Label-Aware Prompting**：在 masked 序列后拼接对应标签作为条件输入，训练和推理阶段均使用，帮助模型在强 mask 条件下维持标签一致性。
- **Reflective Conditional Sample Generation**：在 TC 模型训练循环中动态生成不同 mask 程度的序列，集成标签注释和 TC 模型注意力分数，搜索最优 mask 条件。
- **Noise-Resistant Training**：对 k 个原始样本生成 B 个伪样本，对比损失 $L_c = \frac{1}{K}\log\sum_{i\in I}\sum_{j\in N_i}\exp(\frac{\text{sim}(h_i,h_j)}{\tau})$ 仅在原始样本间计算以扩大不同类别间隔；分类损失 $L_e$ 使用所有样本（原始+伪）的交叉熵；总损失 $L = L_c + L_e$。

## 实验与结果
- **数据集**：SMP2020-EWECT（中文疫情推文，6类，高度不平衡）、India-COVID-X（英文疫情推文，4类）、SenWave（多语言疫情推文，英/阿/法/西）、SST-2（英文电影评论，2类）。
- **基线**：Resample、Back Translation、EDA、AEDA、GENIUS、LAMBADA（SFT GPT-2）、mixup、AWD、SSMBA、ALP、SE。
- **主要结果**：
  - SMP2020-EWECT（G/E策略）：DiffusionCLS 达 Macro-F1 67.98%，提升 +2.11%，准确率 80.23%，提升 +1.06%，在所有数据增强方法中最优。
  - India-COVID-X（G/E策略）：Macro-F1 74.65%，提升 +3.66%，准确率 74.41%，提升 +3.78%，排名第二。
  - SenWave 部分数据实验中，50% 数据即可匹敌原始 PLM 全量性能（阿拉伯语）。
  - SST-2 few-shot：5-shot 达 65.30%（第二名），10-shot 达 68.29%（第二名）。
- **消融实验**：移除 D.A.（-2.6%/-3.59%）、L.A.P.（-1.27%/-1.67%）、N.R.T.（-1.03%/-1.04%），各模块均有正贡献。

## 相关工作脉络
- **DiffusionBERT (He et al., 2023)**：将扩散模型与 MLM 结合用于序列生成，本文在此基础上引入标签感知噪声调度与提示，实现条件生成增强。
- **GENIUS (Guo et al., 2022)**：基于 BART 的 CTR 方法，依赖预训练知识和 sketch 生成，本文通过扩散 LM 更好地捕获域内知识。
- **LAMBADA (Anaby-Tavor et al., 2020)**：微调 GPT-2 以标签为条件生成，忽略原文上下文导致一致性差，本文同时利用原文和标签条件。
- **EDA/AEDA**：基于规则的简单替换或插入，多样性有限且易引入域外噪声，本文生成质量更高。
- **SSMBA/ALP/SE**：few-shot 场景下的前沿方法，本文在 SST-2 上与之竞争，表现接近最优基线。
- **Mixup/AWD**：表征增强方法，通过插值或对抗扰动生成伪样本，本文聚焦于序列级文本增强。

## 局限性与未来方向
- 极端低资源场景下数据生成器性能受限，因模型仍需训练数据，数据不足会负面影响生成器。
- 扩散训练引入额外计算开销，需权衡生成质量与训练效率。
- 未来可将无标签数据引入扩散训练以缓解数据不足问题。

## 研究启发与可借鉴点
- **Label-Aware Noise Schedule 思路可迁移**：基于代理模型注意力分数的 token 重要性评估方法可应用于其他 NLP 任务的数据增强。
- **多样性-一致性权衡的量化分析**：通过 t-SNE 可视化和分组 mask 实验寻找最优 mask 比例的思路值得借鉴。
- **对比损失与分类损失的联合训练**：Noise-Resistant Training 中对比损失仅在原始样本间计算的设计，可有效减轻噪声伪样本的影响。
- **与团队方向结合机会**：该方法的扩散 LM 数据增强框架可扩展至跨语言情感分类、方面级情感分析等场景。

## 关键术语表
**Diffusion Language Model (扩散语言模型)**：将扩散模型思想应用于离散文本序列生成的语言模型，通过逐步添加/去除噪声实现文本生成。
**Label-Aware Noise Schedule (标记感知噪声调度)**：基于 token 与标签的相关性（注意力分数）设计非均匀的 mask 进度，使关键情感 token 后被 mask 的策略。
**CTR (Corrupt-Then-Reconstruct)**：先对原文本进行 corrupt 操作（如 mask），再通过模型重构的文本数据增强范式。
**Noise-Resistant Training (噪声抵抗训练)**：结合对比损失和分类损失，通过扩大不同类别样本间隔来降低伪样本噪声影响的方法。
**Proxy Model (代理模型)**：用于计算 token 重要性权重的辅助文本分类模型，基于其 [CLS] token 注意力分数估计 token 与标签的相关性。
**Macro-F1**：按各类别分别计算 F1 后取算术平均的评价指标，适用于标签不平衡场景。

## 可复现要素
- **数据集**：SMP2020-EWECT、India-COVID-X、SenWave、SST-2（均为公开数据集）
- **代码**：GitHub 开源（论文提及）
- **模型权重**：bert-base-uncased、chinese-roberta-wwm-ext 等来自 HuggingFace 平台
- **关键超参**：Diffusion Steps=32，Aug. Samples=4，Learning Rate=4e-06，Weight Decay=0.01，λ=0.5（图示），具体详见论文 Appendix Tables 5-6
