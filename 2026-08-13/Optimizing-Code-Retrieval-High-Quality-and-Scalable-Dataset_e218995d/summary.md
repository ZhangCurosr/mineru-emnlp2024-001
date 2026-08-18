---
title: "Optimizing-Code-Retrieval-High-Quality-and-Scalable-Dataset"
source: https://aclanthology.org/2024.emnlp-main.123.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:13:58"
field: "代码理解与检索"
keywords: ["代码检索", "数据集构建", "大语言模型", "函数调用关系", "拓扑排序", "对比学习"]
innovations: ["基于拓扑排序的多级函数调用解耦标注算法", "基于流行度的差异化第三方API文档收集策略"]
benchmarks: ["CoNaLa", "SO-DS", "StaQC", "CoSQA", "WebQueryTest"]
---

# 论文速读：Optimizing-Code-Retrieval-High-Quality-and-Scalable-Dataset

## 一句话总结
论文针对代码检索数据集构建中**质量与可扩展性难以兼顾**的问题，提出了一种基于大语言模型（LLM）的高质量、低成本自动标注方法，通过分析并增强函数调用上下文（库内调用关系与第三方API文档），构建了包含 **237.2K** 条 query-code 对的 **Query4Code** 数据集，并在多个真实场景基准上验证了其有效性。

## 研究问题与动机
1. **现有数据收集方法存在权衡困境**：从 GitHub 直接提取 docstring 生成的 query 与真实用户搜索词差距大；从编程社区（如 Stack Overflow）收集的用户提问往往伴随低质量代码片段；而基于专家手动标注的方法成本高、难以扩展。
2. **LLM 直接标注代码函数存在上下文缺失**：代码函数之间存在多级调用关系（intra-repository function calls）和第三方 API 调用，LLM 仅凭单个函数代码难以准确理解其功能，导致生成的 query 质量低。
3. **第三方可调用 API 的理解差异**：LLM 对高频流行的第三方 API 有较好理解，但对冷门 API（unpopular APIs）缺乏认知，直接影响标注质量。
4. **缺乏面向真实检索场景的高质量代码检索预训练数据**：现有数据集（如 CodeSearchNet）的 query 多为官方 docstring，与用户在搜索引擎中输入的 query 分布差异较大。

## 核心贡献（创新点）
1. **系统分析了影响 LLM 代码标注质量的两个关键因素**：库内函数调用数量和第三方 API 调用流行度，揭示了多级调用关系和低频 API 是标注质量下降的主因。
2. **提出了基于拓扑排序的多级函数调用解耦标注算法**：将多级调用关系分解为单级依赖，按无环有向图的拓扑序逐层标注，避免上下文膨胀和推理困难。
3. **设计了基于流行度的第三方 API 文档收集策略**：对超过阈值的热门 API 直接使用 LLM 理解，对冷门 API 则通过网页爬取补充其文档说明后注入标注上下文。
4. **构建了 Query4Code 大规模代码检索数据集**：包含 237.2K 条来自 12.3K 个 Python 仓库的高质量 query-code 对，并提供了两阶段标注生成的功能摘要（summary）。
5. **提出了基于 LLM 的反向验证与质量过滤机制**：通过 CLS 分类模型判断 code 是否能满足 query 需求，过滤掉相关性和匹配度低的样本。

## 方法详解
**整体流程**：解析仓库函数调用图 → 拓扑排序确定标注顺序 → 分层收集调用上下文 → 两阶段 LLM 标注（代码理解→query生成）→ 反向验证与过滤。

1. **任务分解（Task Decomposition）**：将标注过程拆分为两步：
   - 第一步：用 LLM 生成代码功能摘要 $s = \text{LLM}(c)$
   - 第二步：基于摘要和代码生成用户 query $q = \text{LLM}(s, c)$
   
2. **基于拓扑排序的标注算法（Algorithm 1）**：
   - 构建函数调用有向图 $G(V, E)$，节点为函数，边表示调用关系
   - 计算每个节点的入度，将入度为 0（无依赖）的函数先标注
   - 标注过程中递归调用时随机删除一条边以继续拓扑排序
   - 当前函数仅需包含其**直接调用函数**的信息，实现多级关系解耦

3. **第三方 API 文档收集**：
   - 以 API 在仓库中的调用频率作为流行度代理
   - 设定流行度阈值，低于阈值的冷门 API 通过 DuckDuckGo 搜索获取文档
   - 用 LLM 总结 API 功能并加入标注上下文

4. **反向验证与过滤（Data Filtering）**：
   - 用 LLM 对 (query, code) 对进行 CLS 分类，评分 0-3：
     - 3分：code 满足且超出 query 需求
     - 2分：code 满足部分 query 类别需求
     - 1分：code 仅满足 <50% 需求
     - 0分：几乎无关
   - 保留得分 1 和 2 的样本进入最终数据集

## 实验与结果
**数据集**：Query4Code，237.2K 条 query-code 对，来自 12.3K 个 Python 仓库，为 CodeSearchNet Python 子集的高质量子集。

**评估基准**：CoNaLa、SO-DS、StaQC（来自 Stack Overflow）、CoSQA、WebQueryTest（来自真实搜索引擎日志）。

**评估指标**：Mean Reciprocal Rank (MRR)。

**主要结果（Zero-shot，Table 3）**：
| 模型 | CoNaLa | SO-DS | StaQC | CoSQA | WebQueryTest |
|------|--------|-------|-------|-------|-------------|
| CodeBERT (Q4C vs CSN) | 25.45 vs 21.65 (+3.80) | 18.98 vs 18.42 (+0.56) | 15.74 vs 14.26 (+1.48) | **59.80 vs 56.34 (+3.46)** | **35.61 vs 32.43 (+3.18)** |
| GraphCodeBERT (Q4C vs CSN) | 28.88 vs 23.70 (+5.18) | 21.56 vs 19.01 (+2.55) | 18.72 vs 16.90 (+1.82) | **60.24 vs 56.83 (+3.41)** | **35.97 vs 31.83 (+4.14)** |
| UniXcoder (Q4C vs CSN) | 29.07 vs 25.47 (+3.60) | 19.85 vs 18.78 (+1.07) | 19.07 vs 16.45 (+2.62) | **58.87 vs 55.22 (+3.65)** | **34.42 vs 30.18 (+4.24)** |

**最强结果**：GraphCodeBERT 在 WebQueryTest 上达到 **35.97** MRR，相对 CodeSearchNet 预训练提升 **+4.14**；在 CoSQA 上提升 **+3.41**。CoSQA 和 WebQueryTest 提升最大，因二者 query 来自真实搜索引擎，与标注数据分布更匹配。

**Fine-tuning 结果**：Query4Code 预训练后在 CoNaLa、SO-DS、StaQC、CoSQA 四个基准上均全面优于 CodeSearchNet 预训练基线。

**人类评估**：三位专家打分，模型评分与专家评分的 Pearson r ≈ 0.63-0.65，专家间一致性 Krippendorff's α = 0.858，验证了过滤方法的有效性。

**成本分析**：GPT-3.5-turbo 单次标注成本 \$0.001-0.004，吞吐量约 3K 请求/分钟；相比人工标注 \$0.2/对，效率提升百倍。

## 相关工作脉络
1. **CodeSearchNet (Husain et al., 2019)**：从 GitHub 解析 docstring-code 对，规模大但 query 为官方文档风格，与用户真实搜索词差距大；本文方法生成更接近用户习惯的 query。
2. **Stack Overflow 系列数据集 (CoNaLa, SO-DS, StaQC)**：收集用户提问与代码对，但社区代码质量参差、存在语句级片段；本文方法聚焦仓库级高质量函数。
3. **CoSQA (Huang et al., 2021)**：基于搜索引擎日志+专家标注，质量高但规模有限、成本高昂；本文以 LLM 替代人工实现同等质量的可扩展方案。
4. **InPars / Promptagator (Bonifacio et al., 2022; Dai et al., 2022)**：用 LLM 生成 IR 查询的开创性工作，但面向文本文档；本文将其首次应用于代码检索领域，并针对代码特有的调用关系进行了适配。
5. **ContraCode / Corder / CodeRetriever**：基于对比学习的代码检索模型，使用语义 preserving 变换做数据增强；本文提供的高质量数据可与这些模型的训练流程结合。
6. **CodeBERT / GraphCodeBERT / UniXcoder / StarEncoder**：主流代码表征预训练模型，本文以其为基线验证 Query4Code 的数据价值，证明新数据集对已有模型的泛化增益。

## 局限性与未来方向
1. **仅针对 Python 语言**：受成本限制仅分析了 Python 仓库，方法虽可迁移到其他语言，但结论在其他语言中的适用性未经验证。
2. **LLM 幻觉风险**：合成数据可能包含错误或无关信息，影响检索准确性。
3. **数据偏差问题**：基于 LLM 合成的数据可能引入模型训练数据中的固有偏见。
4. **仅覆盖函数级别**：未涉及模块级、类级或文件级代码检索任务。
5. **未来方向**：探索其他编程语言的数据集构建、扩展到更大规模的仓库、结合人类反馈进一步迭代优化。

## 研究启发与可借鉴点
1. **多级调用关系的拓扑排序解耦策略**：将复杂的多级依赖分解为单层依赖的序列标注，可有效控制 LLM 上下文长度并提升理解准确性，该方法可迁移至任何存在调用链的代码理解任务。
2. **基于流行度的差异化上下文增强**：对高频知识直接用 LLM，对低频知识通过外部检索补充，这一"分级处理"思路可用于其他知识边界不清的 LLM 应用。
3. **两阶段标注（理解→生成）的解耦设计**：先让 LLM 理解代码语义生成 summary，再基于理解生成 query，有效分离了语义理解和意图对齐两个不同能力，提示质量更高。
4. **LLM 反向验证过滤机制**：用 LLM 自身对生成结果进行质量评估和分类过滤，为低成本合成数据质量保障提供了可复用的范式。
5. **数据集的跨任务潜力**：标注过程同时生成了功能摘要，可复用于代码摘要、代码补全等下游任务，体现了数据构建的"一次标注、多方复用"理念。

## 关键术语表
**Code Retrieval（代码检索）**：根据自然语言查询在代码库中检索语义最相关的代码片段的任务。
**Intra-repository Function Calls（库内函数调用）**：同一仓库内部不同函数之间的调用关系。
**Third-party API Calls（第三方 API 调用）**：函数对外部库/API 的调用，LLM 对其理解程度受 API 流行度影响。
**Topological Sorting（拓扑排序）**：对有向无环图节点进行线性排序，使每条边的起点排在终点之前，用于确定函数标注顺序。
**Query4Code（数据集）**：本文构建的包含 237.2K 条 query-code 对的大规模代码检索预训练数据集。
**MRR（Mean Reciprocal Rank）**：代码检索常用评估指标，即正确结果排名的倒数的均值。
**InfoNCE Loss**：对比学习常用的损失函数，通过最大化正样本对相似度、最小化负样本对相似度来学习表征。
**CLS Prompting（分类提示）**：用 LLM 对 (query, code) 对进行质量分类评分的提示方法。

## 可复现要素
- **数据集**：Query4Code，论文未明确声明公开状态，需查阅论文补充材料或作者主页确认。
- **代码**：论文未提及开源代码仓库。
- **模型**：使用 GPT-3.5-turbo (gpt-3.5-turbo-0613) 进行标注，GPT-4-turbo (gpt-4-1106-preview) 进行评分，CodeLlama-Instruct 7B 用于对比实验。
- **关键超参**：Temperature=0.2, top-p=0.95（CodeLlama 推断）；fine-tuning learning rate ∈ {1e-5, 2e-5, 5e-5}，batch size ∈ {32, 64, 128}，epochs=10，early stopping；三个随机种子 {0, 1, 2}。
- **硬件**：GeForce RTX 4090 GPU。
- **工具**：tree-sitter 用于代码解析，DuckDuckGo 用于 API 文档爬取，vLLM 用于加速 CodeLlama 推理。
