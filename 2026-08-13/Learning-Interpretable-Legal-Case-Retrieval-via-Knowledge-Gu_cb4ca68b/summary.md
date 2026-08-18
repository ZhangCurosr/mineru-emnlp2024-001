---
title: "Learning-Interpretable-Legal-Case-Retrieval-via-Knowledge-Gu"
source: https://aclanthology.org/2024.emnlp-main.73.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:10:19"
field: "法律信息检索"
keywords: ["legal case retrieval", "knowledge-guided reformulation", "large language model", "dual-level contrastive learning", "MaxSim", "interpretable retrieval", "long document retrieval"]
innovations: ["提出KGCR两步提示框架，利用罪名-法条法律知识引导LLM将法律案例重构为结构化子事实", "基于子事实的MaxSim聚合实现高效可解释的法律案例相关性建模", "设计双级对比学习与启发式子事实级标签派生策略缓解标注稀缺"]
benchmarks: ["LeCaRD", "LeCaRDv2"]
---

# 论文速读：Learning-Interpretable-Legal-Case-Retrieval-via-Knowledge-Guided-Case-Reformulation

## 一句话总结
本文提出 **KELLER**（Knowledge-guidEd case reformuLation approach for LEgal case Retrieval），利用大语言模型结合罪名与法条等专业知识，将冗长复杂的法律案例重构为结构化的"罪名-子事实"片段，再基于子事实间的 MaxSim 交互与双级对比学习实现高效且可解释的法律案例检索，在 LeCaRD 和 LeCaRDv2 上均取得 SOTA 效果。

## 研究问题与动机
1. **法律案例检索的核心挑战**：查询与文档均为结构化法律文书，平均长度数千 token，远超传统检索模型（如 BERT）512 token 的输入限制；且一个案例常包含多个独立的犯罪行为，关键信息分散于全文中。
2. **现有方法的不足**：
   - 扩展上下文窗口或段落分割（如 BERT-PLI、Lawformer）无法充分处理法律文本的高度专业性。
   - 直接文本摘要或嵌入级摘要（如 PromptCase）仅依赖启发式规则或模型自有知识，缺乏法律专家知识（如罪名、法条）引导，易遗漏重要细节。
3. **长文本理解与语义匹配的耦合难题**：单一检索器同时承担长文本理解与复杂语义匹配任务，在有限输入容量下难以兼顾。
4. **可解释性缺失**：法律检索场景要求结果可追溯、可解释，现有方法多为端到端黑盒相似度计算，无法说明匹配依据。

## 核心贡献（创新点）
1. **法律知识引导的案例重构（KGCR）**：提出两步提示策略——先提取罪名与法条并建立映射，再结合法条对罪名进行子事实摘要。与 PromptCase 等纯 LLM 摘要方法的本质区别在于：引入外部法律专家知识库（罪名-法条对应关系）作为高层抽象引导，显著降低子事实识别难度。
2. **基于子事实的 MaxSim 相关性建模**：将案例重构为结构化子事实后，直接对子事实嵌入矩阵应用 MaxSim + Sum 聚合得到最终相关性分数。与 kernel pooling 等软聚合方法的本质区别在于：法律案例中查询子事实与文档子事实通常为一对一匹配，MaxSim 避免噪声引入，且具备天然的可解释性（可追溯每个子事实的贡献）。
3. **双级对比学习（Dual-Level Contrastive Learning）**：在案例级对比损失之外，引入子事实级对比损失，通过启发式策略从案例级标注中派生高质量子事实级标签。与单一案例级训练的本质区别在于：利用跨粒度匹配信号增强子事实内容的理解能力。

## 方法详解
**1. 法律知识引导案例重构（KGCR）**
- **第一步：罪名与法条抽取**：用 LLM（Qwen-72B-Chat）从案例文本中提取所有罪名（crimes）和法条（law articles），输出格式为每条一行、多值以分号分隔。
- **第二步：后处理**：将法条标题扩展为完整条文内容（从 Web 获取），并参照法律专家数据库建立罪名-法条的双向映射。
- **第三步：子事实摘要**：以"罪名+法条"对为提示输入，引导 LLM 从原案例中生成该罪名的具体事实摘要（每部分不超过 100 词），最终形成以罪名为标题、事实为主体的子事实片段（sub-fact snippet）。

**2. 相关性建模（Relevance Modeling）**
- 对查询 $q$ 的子事实 $\{q_1, ..., q_m\}$ 和文档 $d$ 的子事实 $\{d_1, ..., d_n\}$，分别通过预训练文本编码器（SAILER）获取 [CLS] 嵌入 $E_{q_i}, E_{d_j}$。
- 计算相似度矩阵 $\mathbf{M}_{m \times n}$：$M_{i,j} = \text{Norm}(E_{q_i}) \cdot \text{Norm}(E_{d_j})^T$（L2 归一化点积）。
- 聚合得分：$s_{q,d} = \sum_{i=1}^{m} \max_{j=1}^{n} M_{i,j}$（MaxSim + Sum）。

**3. 双级对比学习（Dual-Level Contrastive Learning）**
- **案例级对比损失**：$\mathcal{L}_{\mathrm{R}} = -\log \frac{\exp(s_{q,d^+}/\tau)}{\exp(s_{q,d^+}/\tau) + \sum_{d^-} \exp(s_{q,d^-}/\tau)}$
- **子事实级对比损失**：$\mathcal{L}_{\mathrm{S}} = -\log \frac{\exp(s_{M_{i,j^+}}/\tau)}{\exp(s_{M_{i,j^+}}/\tau) + \sum_{J^-} \exp(s_{M_{i,j^-}}/\tau)}$
  - 标签生成策略（Appendix C）：若文档子事实与查询子事实共享同一罪名则视为正样本，否则基于相似度最高者选正；负文档中同罪名子事实不参与以避免假负样本。
- **总损失**：$\mathcal{L} = \mathcal{L}_{\mathrm{R}} + \alpha \mathcal{L}_{\mathrm{S}}$，其中 $\alpha = 0.9$，$\tau = 0.01$。

## 实验与结果
- **数据集**：LeCaRD（107 queries / 10,700 docs）和 LeCaRDv2（800 queries / 55,192 docs），以 label=3（LeCaRD）/ label=2（LeCaRDv2）为正样本。
- **评估指标**：MAP、P@3、NDCG@3/5/10。
- **主要结果（LeCaRDv2 微调设置）**：
  | 模型 | MAP | P@3 | NDCG@3 | NDCG@5 | NDCG@10 |
  |---|---|---|---|---|---|
  | SAILER（最佳基线） | 60.62 | 54.58 | 78.67 | 78.99 | 81.41 |
  | **KELLER** | **68.29** | **63.13** | **84.97** | **85.63** | **87.61** |
  - KELLER 相比 SAILER：MAP +7.67，NDCG@3 +6.30，NDCG@10 +6.20（相对提升约 7.6%）。
- **零样本设置（LeCaRDv2）**：KELLER MAP 65.87 vs. SAILER 62.80，提升 +3.07。
- **复杂查询表现**：在争议性查询（controversial queries）上，KELLER 相对次优方法的提升达 LeCaRD +24.04%、LeCaRDv2 +13.41%，显著优于简单查询场景。
- **消融结论**：KGCR 模块消融影响最大（MAP 从 68.29 降至 61.91）；MaxSim 优于 Mean/NC/KP 等聚合策略；双级对比学习两者均有益。

## 相关工作脉络
1. **SAILER（Li et al., 2023）**：面向法律案例检索的结构化预训练模型。本文在其之上通过 LLM 重构预处理输入，而非在编码端改进；SAILER 零样本效果强但微调后存在过拟合，KELLER 通过知识引导重构绕过长文本瓶颈。
2. **PromptCase（Tang et al., 2023）**：用 LLM 将案例摘要为 50 词。本文与其本质区别在于：PromptCase 无结构化输出且忽略多罪名场景，KELLER 通过法律知识引导生成多个结构化子事实，保留关键细节。
3. **BERT-PLI（Shao et al., 2020）与 Lawformer（Xiao et al., 2021）**：通过段落交互建模或扩展上下文窗口处理长文本。本文认为单一检索器同时处理理解与匹配存在瓶颈，通过 KGCR 将理解任务卸载至 LLM，检索器专注子事实级匹配。
4. **HyDE（Gao et al., 2023）与 Query2Doc（Wang et al., 2023）**：LLM 查询改写方法，生成伪文档辅助检索。本文的应用场景不同——查询和文档均为长法律文书，改写目标不是查询而是文档本身的结构化分解。
5. **BM25/TF-IDF**：传统词汇匹配基线，无法建模语义匹配，在复杂法律案例上性能明显落后。

## 局限性与未来方向
1. **外部知识库构建依赖**：需预先构建罪名-法条映射数据库，增加了工程步骤，不如开箱即用的稠密检索器便捷。
2. **LLM 推理开销**：对查询案例的处理需调用大语言模型（Qwen-72B-Chat），带来额外计算成本；虽然作者提及 vLLM 加速和低成本推理服务（如 Llama3-8B 超 800 tokens/s），但在线部署仍需权衡。
3. **现实场景的不确定性**：人工标注数据集上的优异表现不一定完全泛化到真实复杂查询；现有案例库未必能覆盖所有用户需求。
4. **未来方向**：引入更多专业知识、探索生成式模型以输出语言化解释、扩展至其他法域。

## 研究启发与可借鉴点
1. **知识引导的结构化重构范式**：对于长文档检索任务，可利用领域专家知识（如本工作中的罪名-法条映射）引导 LLM 将非结构化长文本分解为结构化的细粒度片段，再在片段级做交互匹配——该范式可迁移至医疗病历检索、专利检索等领域。
2. **双级对比学习的设计思路**：从粗粒度标注（案例级）出发，通过启发式规则派生细粒度标签（子事实级）进行联合训练，可有效缓解标注稀缺问题；此思路可应用于多粒度匹配的其他检索任务。
3. **MaxSim 聚合+可解释性的结合**：在法律等需要结果可追溯的场景中，MaxSim 不仅能提供高效的多向量近似最近邻检索，还能显式展示每个子事实的匹配贡献，这一设计值得在需可解释性的工业检索系统中借鉴。
4. **Zero-shot 鲁棒性验证**：在数据稀缺的垂直领域，零样本性能是重要的评估维度；本文同时报告了零样本和微调结果，并分析了 SAILER 微调后性能下降的原因（过拟合），这一分析视角值得参考。

## 关键术语表
**KELLER**：Knowledge-guidEd case reformuLation approach for LEgal case Retrieval，本文提出的法律知识引导案例重构检索方法。
**KGCR（Knowledge-Guided Case Reformulation）**：法律知识引导案例重构，通过两步 LLM 提示将法律案例分解为结构化的罪名-子事实片段的核心模块。
**MaxSim**：Late Interaction 聚合算子，对查询与文档子事实嵌入矩阵逐行取 max 后求和，兼顾效率、效果与可解释性。
**Dual-Level Contrastive Learning**：双级对比学习，同时在案例级（粗粒度）和子事实级（细粒度）施加对比损失。
**LeCaRD / LeCaRDv2**：两个中文法律案例检索基准数据集，由中国人民大学等机构构建。
**SAILER**：Structure-Aware Iterative Legal case Retrieval，面向法律案例检索的结构化预训练语言模型。
**PromptCase**：基于提示的法律案例摘要方法，将案例压缩为 50 词后做检索，是本文的主要对比基线。
**Sub-fact**：子事实，经 KGCR 重构后得到的以罪名为标题、犯罪事实为主体的一段结构化文本。

## 可复现要素
- **数据集**：LeCaRD 和 LeCaRDv2（论文附录 A.1 给出统计；需自行下载，非本文提供）
- **代码/权重**：论文未声明代码开源仓库；使用 Qwen-72B-Chat 进行案例重构（开源 LLM），检索编码器使用 SAILER（开源模型）
- **关键超参**：学习率 1e-5，batch size 128，温度参数 τ=0.01，子事实级损失权重 α=0.9，最大罪名数上限 4，子事实摘要每部分 ≤100 词
- **硬件**：4 × Nvidia Tesla A100-40G GPU
- **LLM 推理加速**：vLLM
