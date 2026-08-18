---
title: "An-Experimental-Analysis-on-Evaluating-Patent-Citations"
source: https://aclanthology.org/2024.emnlp-main.23.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:25:59"
field: "知识产权与专利信息挖掘"
keywords: ["patent citation prediction", "graph neural network", "patent analysis", "GNN explainability", "semantic graph", "binary classification"]
innovations: ["将专利引用预测形式化为二分类问题并构建纯语义图输入GNN，仅用文本达94%召回率", "提出基于GNNExplainer的专利质量可解释性分析，发现高引用专利解释子图具有更高聚类系数", "系统性验证了不同标签分层设置下GNN方法的稳健性，并揭示了训练数据时效性对预测性能的关键影响"]
benchmarks: ["USPTO Patent Dataset (A61/H04/G06 CPC classes)", "3/5/10-year citation history splits"]
---

# 论文速读：An-Experimental-Analysis-on-Evaluating-Patent-Citations

## 一句话总结
本文将近十年来的美国专利引用预测建模为二分类任务（高引用 vs 低引用），通过 Doc2Vec 构建专利语义图并结合 GNN（GCN / GraphSAGE / GTN）进行预测，实验表明仅用专利文本即可达到高达 **94% 的召回率**，并显著优于纯文本 MLP 基线；同时借助 GNNExplainer 对预测结果提供可解释性分析。

## 研究问题与动机
1. **专利引用数量是衡量专利质量的核心指标**，但影响专利获得高引用的因素至今缺乏系统理解，传统研究多聚焦于事后计量分析而非预测建模。
2. **纯文本方法的局限性**：先前基于文本（如 Hsu et al., 2020 的回归方法）未能捕捉专利间语义关联的网络结构，难以反映专利在技术景观中的"位置优势"。
3. **可解释性需求**：专利领域文本包含大量专业术语和律师撰写风格，传统 LIME 等文本解释器效果有限，需要结合图结构提供领域适配的解释。
4. **数据稀疏与时间动态**：专利引用随时间累积，短观察期（3 年）数据噪声大，长观察期（10 年）数据量减少，需在时效性与样本量之间权衡。

## 核心贡献（创新点）
1. **专利引用预测的二分类框架**：将引用预测从回归问题重新形式化为二分类问题（Top/Bottom x 百分位），通过分层标签策略（high vs rest / high vs middle / middle vs low）提供更细粒度的评估视角，区别于 Hsu et al. (2020) 的回归设定。
2. **基于 Doc2Vec 语义相似度的专利图构建方法**：使用约 20 万份专利文本训练 Doc2Vec 获取 100 维节点嵌入，以相似度阈值（0.6–0.8）构建无向语义图，使 GNN 可利用邻居节点的结构信息，而此前相关工作（如 Patent2vec）主要依赖标签/关键词图而非纯语义图。
3. **系统性 GNN 基线对比实验**：在三种 CPC 大类（A61/H04/G06）× 三种引用历史长度（3/5/10 年）的九组数据集上，全面比较 Doc2Vec/GPT-BERT 特征与 GCN/GraphSAGE/GTN 的组合，发现 GNN 方法在 Recall 上最高达 94%，远超 MLP 基线（~50%，接近随机）。
4. **面向专利质量解释的图可视化分析**：利用 GNNExplainer 提取解释子图并计算聚类系数等图拓扑指标，发现高引用专利的解释子图具有更高的平均聚类系数（CC），揭示了"高影响力专利的邻居也倾向于高引用"的结构模式。

## 方法详解

### 问题定义
给定 $m$ 个专利集合 $\mathbb{P} = \{P_1, P_2, \cdots, P_m\}$，令 $C_d^i$ 表示专利 $P_i$ 在授权后 $d$ 年获得的引用数（实验取 $d \in \{3, 5, 10\}$），标签函数定义为：

$$
y(C_d^i) = \begin{cases} 1, & \text{if } C_d^i \geq C_{x,h} \quad \text{(高引用)} \\ 0, & \text{if } C_d^i \leq C_{x,l} \quad \text{(低引用)} \end{cases}
$$

其中 $C_{x,h}$ 和 $C_{x,l}$ 分别为引用数分布的第 $x$ 百分位（上/下截断），实验取 $x=10$。此外还设计了三种成对分类设置：High vs Rest、High vs Middle、Middle vs Low。

### 图构建流程
1. **节点嵌入**：在约 20 万份专利（标题 + 摘要 + 权利要求）上训练 Doc2Vec，获得 100 维向量；另用 PatentBERT（1024 维）作为补充特征。
2. **边构建**：计算节点对之间的余弦相似度，当相似度 ≥ 阈值（0.6–0.8，对应图密度 5–25%）时建立无向边。
3. **图表示**：$G = (V, X, A)$，其中 $X \in \mathbb{R}^{n \times d}$ 为节点特征矩阵，$A \in \{0,1\}^{n \times n}$ 为邻接矩阵。

### GNN 模型
三种 GNN 变体均采用消息传递机制 $h_u^{(l)} = AGGR(h_u^{(l-1)}, \{h_i^{(l-1)} | i \in N(u)\})$：

- **GCN**：采用均值聚合，传播规则 $H^{(l)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l-1)}W^{(l-1)})$
- **GraphSAGE**：支持多种聚合器（mean/max-pooling/LSTM），并将当前节点表示与聚合后的邻居表示拼接
- **GTN（Graph Transformer Network）**：引入自注意力机制计算节点间 attention scores，再通过图卷积聚合邻居信息

所有模型以交叉熵为损失函数，Adam 优化，weight decay = 5e-4，最大 epoch = 500，学习率在 0.01–0.00001 范围内搜索。

### 可解释性分析
使用 **GNNExplainer**（Ying et al., 2019）提取节点分类的解释子图，并计算子图的三个拓扑指标：密度（density）、平均度（degree）、聚类系数（Clustering Coefficient, CC），对比高/低引用类别的差异。

## 实验与结果

### 数据集
- **来源**：USPTO 授权专利（2000–2022）
- **类别**：三个最大 CPC 类——A61（医学，269,364 件）、H04（电信，379,099 件）、G06（计算机，340,667 件）
- **划分**：九组数据集（3 CPC × 3 历史长度），每组保留最近 2 年数据作为测试集
- 训练集规模示例（Table 9）：A61-10y 为 (20,912, 5,981)，H04-3y 为 (59,713, 1,986)

### 基线方法
- **Doc2Vec-MLP**、**PatentBERT-MLP**（纯文本特征 + 多层感知机）
- **GCN / GraphSAGE / GTN**（分别搭配 Doc2Vec 和 PatentBERT 特征，共 6 种 GNN 配置）

### 核心结果（Table 2，正类召回率）
| 场景 | 最强模型 | 最高召回率 | 对比 MLP 基线提升 |
|---|---|---|---|
| A61-3y | PatentBERT-GTNP | 0.85 | vs PatentBERT-MLP (0.76) |
| A61-10y | PatentBERT-GTNP | **0.94** | vs PatentBERT-MLP (0.91) |
| H04-5y | Doc2Vec-GSAGE | **0.94** | vs Doc2Vec-MLP (0.68) |
| G06-5y | PatentBERT-GSAGE | **0.93** | vs PatentBERT-MLP (0.89) |

**关键结论**：
- GNN 方法在所有设置下均显著优于 MLP 基线，最高recall达 **94%**（>75% 的高引用专利被正确识别）
- GraphSAGE 和 GTN 普遍优于 GCN，说明更复杂的邻居聚合机制（广义聚合/自注意力）带来增益
- PatentBERT 特征在多数设置下优于 Doc2Vec，但 Doc2Vec 在部分场景（如 H04-5y）表现更强，表明两种嵌入各有优势

### 不同标签设置（Tables 3–5）
在更具挑战性的分类设置下（High vs Rest/Middle/Low vs Middle），GNN 方法仍保持稳定性能（F1 ≈ 0.70–0.74），而 MLP 基线退化为接近随机水平（F1 ≈ 0.50–0.52）。

### 时效性分析（Table 8）
使用更新近的训练数据（2010–2014 vs 2000–2004）预测 2016 年专利时，PatentBERT-GSAGE 的 Accuracy 从 0.67 提升至 0.73，Recall 从 0.64 提升至 0.83，印证了**近期专利文本对预测更有价值**。

### 可解释性结果（Table 6）
高引用专利的解释子图具有更高的平均聚类系数（CC）：A61 为 0.265 vs 0.228，H04 为 0.46 vs 0.331，G06 为 0.431 vs 0.284，揭示了高引用专利的局部邻域更紧密的结构特征。

## 相关工作脉络
1. **Hsu et al. (2020) Deep Learning, Text, and Patent Valuation**：使用深度学习从专利文本预测引用数量的回归模型；本文与其本质区别在于将问题重新定义为二分类、构建语义图并利用 GNN 捕获结构信息、增加可解释性模块。
2. **Patent2vec (Fang et al., 2021)**：基于多视图图（含标签/引用关系）的专利分类方法；本文与之差异在于完全基于纯语义相似度建图（无需外部引用结构），更适用于新授权专利的早期预测。
3. **PatentBERT (Lee & Hsiang, 2020)**：针对专利分类微调 BERT 的预训练模型；本文将其作为节点特征来源之一，并与 Doc2Vec 进行对比实验，验证了 PatentBERT 在大多数设置下的优越性。
4. **GNNExplainer (Ying et al., 2019)**：通用 GNN 解释器；本文首次将其应用于专利语义图的可解释性分析，发现解释子图的聚类系数可有效区分高/低引用类别。
5. **Hall et al. (2001, 2005)**：基于 NBER 专利引用数据库的经济计量学研究，关注引用与专利市场价值的关系；本文从 ML 预测角度补充了"仅凭文本能否预判引用质量"这一方法学问题。
6. **Kakkad et al. (2023) GNNX-Bench**：GNN 解释器的系统基准评测；本文在其方法论基础上展示了 GNN 解释在专利领域的具体应用效果。

## 局限性与未来方向
1. **未利用引用图结构**：本文的语义图完全基于文本相似度构建，未纳入实际的专利引用关系（bibliographic citations），而引用图本身蕴含丰富的技术传承信息。
2. **标签截断导致的信息损失**：仅使用分布两端的 x% 样本（丢弃中间部分），虽然提升了分类难度适中性，但损失了大量中等引用专利的有用信号。
3. **时效性局限**：数据截至 2022 年，且模型需定期重新训练以适应技术领域的快速变化（实验已验证近期数据的重要性）。
4. **解释的深度有限**：GNNExplainer 提供的是子图拓扑层面的解释，尚未深入到具体文本片段的语义归因。
5. **未来方向**：论文自述可进一步探索高引用专利的因果推理机制，并构建实际部署的原型系统。

## 研究启发与可借鉴点
1. **"文本+结构"双通道设计范式**：仅用文本 embedding 构建语义图再应用 GNN，是一种无需外部图结构即可发挥图模型优势的有效策略，可迁移至其他缺少显式关系图的领域（如学术论文影响力预测、代码仓库关联分析）。
2. **分层标签策略的价值**：除传统的 high vs low 二分类外，引入 middle 层设计 High vs Middle / Middle vs Low 的成对分类，为模型在难分类场景下的鲁棒性评估提供了更细粒度的视角，值得在类似质量预测任务中借鉴。
3. **图拓扑指标作为可解释性代理**：利用聚类系数等图论指标对比不同类别的解释子图差异，是一种无需额外训练即可验证模型行为合理性的轻量级解释方案，可复用于其他 GNN 应用。
4. **时效性实验设计**：按时间窗口切分训练/测试数据以验证模型的时序泛化能力，这一设计对任何具有时间依赖性的预测任务均有参考价值。
5. **Doc2Vec 在专利领域的持续有效性**：尽管 PatentBERT 整体更优，但 Doc2Vec 在部分设置下表现更强且计算效率更高，提示在资源受限场景下传统方法仍具竞争力。

## 关键术语表
**Patent Citation（专利引用）**：后续专利引用先前的专利，是衡量技术影响力和知识传递的核心指标。
**CPC Class（合作专利分类）**：Cooperative Patent Classification，国际通用的专利分类体系，本文选取 A61（医学）、H04（电信）、G06（计算机）三个最大类别。
**Doc2Vec（文档向量）**：Paragraph Vector 的扩展，将整篇文档映射为固定长度向量，本文用于构建专利语义图和节点特征。
**PatentBERT**：在超 1 亿份专利数据上预训练的 BERT-LARGE 模型，提供 1024 维专利专用文本嵌入。
**GNNExplainer**：通过学习节点级和边级 mask 来生成 GNN 预测的可解释子图的通用解释框架。
**Recall（召回率）**：正类（高引用专利）中被正确识别的比例，本文的核心评估指标，反映"发现高质量专利"的能力。
**Clustering Coefficient（聚类系数）**：衡量图中节点邻域紧密程度的指标，本文用于量化解释子图的结构特性。
**Binary Classification Formulation（二分类形式化）**：将引用预测从回归转化为二分类问题，通过百分位截断定义高/低引用标签。

## 可复现要素
- **数据集**：USPTO 授权专利（2000–2022），三大 CPC 类；论文已公开数据访问方式
- **代码**：开源于 https://github.com/robi56/patent_high_citation/
- **关键超参**：Doc2Vec 维度=100；PatentBERT 维度=1024；边相似度阈值=0.6–0.8；图密度=5–25%；Epoch=500；Optimizer=Adam；Weight decay=5e-4；Loss=Cross-Entropy；Learning rate=0.01–0.00001（网格搜索）
- **硬件环境**：Google Cloud Ubuntu VM，64GB RAM，8 vCPUs
