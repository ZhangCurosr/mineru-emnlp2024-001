---
title: "EfficientRAG-Efficient-Retriever-for-Multi-Hop-Question-Answ"
source: https://aclanthology.org/2024.emnlp-main.199.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:35:26"
field: "检索增强生成与多跳问答"
keywords: ["RAG", "Multi-hop QA", "Efficient Retrieval", "Small Model", "Query Generation", "DeBERTa"]
innovations: ["用小型分类器替代 LLM 完成迭代查询生成与信息筛选", "无 LLM 调用的多跳检索框架，效率提升 60-80%", "跨数据集强泛化的即插即用 RAG 增强模块"]
benchmarks: ["HotpotQA", "MuSiQue", "2WikiMQA"]
---

# 论文速读：EfficientRAG: Efficient Retriever for Multi-Hop Question Answering

## 一句话总结
论文提出 EfficientRAG，一个无需在每次迭代中调用 LLM 即可高效生成新查询并过滤无关信息的小型检索器，用于多跳问答；实验表明其在三个开放域多跳 QA 数据集上超越现有 RAG 方法，检索效率提升显著。

## 研究问题与动机
- 现有迭代式 RAG 方法（如 Iter-RetGen、Self-Ask）依赖多次 LLM 调用进行查询重写/分解，导致延迟高、成本低。
- 需要大量专用 prompt 和 few-shot 示例，跨场景泛化能力受限。
- 多跳问题的关系类型有限但实体数量庞大，小模型足以完成关系识别与实体抽取。
- 现有方法在检索过程中引入大量无关 chunk，影响 LLM 生成质量。

## 核心贡献（创新点）
1. **轻量级双模块检索框架**：提出 Labeler & Tagger 与 Filter 两个小模型组件协同工作，替代 LLM 完成查询迭代生成与信息筛选。
2. **无 LLM 调用的迭代检索**：通过 token 级分类与 chunk 级标注实现多跳推理链构建，避免多轮 LLM 推理开销。
3. **合成数据自动构建流程**：利用 Llama-3-70B 自动生成多跳分解、token 标注、下跳查询与负样本训练数据。
4. **即插即用的 RAG 增强器**：EfficientRAG 可接入任意迭代式 RAG 系统，无需修改生成器部分。
5. **跨数据集强泛化能力**：在不同多跳 QA 数据集间训练/测试迁移，性能稳定甚至超越原域训练模型。

## 方法详解
**整体框架**：EfficientRAG 包含两个核心组件——Labeler & Tagger（负责标注有用 token 并判断 chunk 是否继续检索）和 Filter（负责从已标注信息生成下一跳查询）。

**Labeler & Tagger 设计**：
- 使用 DeBERTa-v3-large 作为编码器，对 query + chunk 序列进行 token embedding。
- Token 分类头：将每个 token 映射到 "useful" 或 "useless" 两类，识别可用于回答子问题的关键信息。
- Chunk 分类头：对序列平均池化后分类为 `<Continue>` 或 `<Terminate>`，判断是否需要继续检索。

**Filter 设计**：
- 输入为 query 与 Labeler 标注的有用 token 拼接序列。
- 同样基于 DeBERTa-v3-large，通过 token 级分类提取下一跳查询关键词。
- 最终输出为可检索的自然语言查询。

**合成数据构建流程**（使用 Llama-3-70B-Instruct）：
1. **多跳分解**：将复杂问题拆分为单跳子问题，建立依赖关系图。
2. **Token 标注**：对每个子问题对应 chunk 进行二值标注（重要/不重要）。
3. **下跳查询生成**：基于子问题与依赖标注生成下一跳查询。
4. **负采样**：检索与过滤后查询最相似但不相关的 chunk 作为硬负样本，标记为 `<Terminate>`。

**训练细节**：
- 基础模型：DeBERTa-v3-large（304M 参数，24 层）
- 优化器：AdamW，学习率 5e-6
- 训练硬件：4× Nvidia A100，约 10 GPU-hours
- 检索器：Contriever-MSMARCO

## 实验与结果
**数据集**：HotpotQA、MuSiQue、2WikiMQA（均为开放域多跳 QA 基准）

**检索性能**（Table 2）：
- HotpotQA：Recall@6.41 = **81.84**（仅需检索 6.41 个 chunk）
- 2WikiMQA：Recall@3.69 = **84.08**（仅需检索 3.69 个 chunk）
- MuSiQue：Recall@6.09 = 49.51（复杂度更高，仍保持高效）
- 对比基线：在相近召回率下，EfficientRAG 检索 chunk 数量仅为 Direct-R@30 的 1/5~1/8。

**端到端 QA 性能**（Table 3，Llama-3-8B-Instruct 生成器）：
- HotpotQA：Acc = **57.86**（仅次于 Iter-RetGen iter3 的 59.47，远超 Direct-R 的 27.99）
- 2WikiMQA：Acc = **53.41**（与 Iter-RetGen iter3 的 59.29 接近）
- MuSiQue：Acc = 26.97（虽 recall 较低，但仍优于多数基线）

**效率对比**（Table 4，200 样本 MuSiQue）：
- LLM 调用次数：EfficientRAG = **1.00** vs Self-Ask = 7.18
- 延迟：EfficientRAG = **3.62s** vs Self-Ask = 27.47s（快约 7.6 倍）
- 相比 Iter-RetGen iter3（9.68s）提升约 **60-80%** 时间效率

**不同生成器性能**（Table 5，2WikiMQA）：
- 使用 GPT-3.5-turbo-1106 时，EfficientRAG 达到 Acc = **61.88**，超越所有基线（包括 Iter-RetGen iter3 的 60.60）。

**迁移能力**（Table 6）：
- HotpotQA 训练 → 2WikiMQA 测试：Acc = **56.59**（超越 2WikiMQA 训练模型的 53.41）
- 证明模型不依赖领域特定知识，泛化能力强。

## 相关工作脉络
- **Direct-R / Naive RAG**：仅单次检索用户查询，无法处理多跳依赖；本文通过迭代检索显著超越。
- **Iter-RetGen**（Shao et al., 2023）：基于 LLM 迭代检索-生成协同，需要 3 次 LLM 调用；本文用小型分类器替代，效率提升 3 倍以上。
- **Self-Ask**（Press et al., 2023）：通过 LLM 生成中间问答链，需 7+ 次 LLM 调用；本文完全避免 LLM 调用完成同等功能。
- **Query Decomposition 方法**：一次性分解复杂问题为子问题；本文通过迭代方式动态生成下跳查询，在相同 chunk 预算下召回率相当。
- **ATLAS / REPLUG**： Few-shot RAG 方法；本文聚焦多跳场景的效率优化，方法论不同。

## 局限性与未来方向
- 最终生成器仅使用 Llama-3-8B，未验证更大 LLM（如 70B）下的性能上限。
- 主要在开放域数据集上验证，领域特定多跳 QA 场景尚未充分探索。
- MuSiQue 等极高复杂度数据集上 recall 仍有提升空间。
- 未来可扩展至更多 QA 范式（如事实验证、表格问答）及领域定制场景。

## 研究启发与可借鉴点
1. **小型分类器替代 LLM 推理**：对于结构化任务（token 标注、chunk 分类），小模型可高效完成，大幅降低延迟与成本。
2. **合成数据自动化构建**：利用大模型生成训练数据（分解、标注、负样本）的流程值得复用，可迁移至其他检索增强任务。
3. **端到端效率-效果权衡**：在 6-7 个 chunk 内实现 80%+ recall，为资源受限场景提供可行方案。
4. **跨数据集泛化设计**：不依赖领域特定知识的设计思路，适用于需要快速部署的新领域场景。
5. **即插即用增强器架构**：EfficientRAG 作为独立模块可无缝接入现有 RAG 流水线，工程友好性强。

## 关键术语表
**Multi-hop Question Answering**：需要多个推理步骤才能回答的问题，通常涉及跨多个文档的信息整合。
**Retrieval-Augmented Generation (RAG)**：将外部知识库检索结果作为上下文输入 LLM，以提升生成准确性与事实性。
**Query Decomposition**：将复杂多跳问题分解为多个简单子问题的过程，便于逐跳检索。
**Labeler**：EfficientRAG 中的 token 级分类器，识别 chunk 中对回答子问题有用的关键信息。
**Filter**：EfficientRAG 中的查询生成器，基于已标注信息构建下一跳检索查询。
**Recall@K**：在检索 Top-K 个 chunk 中包含正确答案所需 chunk 的比例。
**Contriever-MSMARCO**：基于对比学习的无监督 dense retriever，用作本文的底层检索模型。
**DeBERTa-v3-large**：解码增强型 BERT 模型，304M 参数，作为 Labeler 和 Filter 的基础编码器。

## 可复现要素
- **数据集**：HotpotQA、MuSiQue、2WikiMQA（公开可用）
- **代码**：https://aka.ms/efficientrag（已开源）
- **模型权重**：基于 DeBERTa-v3-large 微调，论文提供训练配置
- **关键超参**：学习率 5e-6，AdamW 优化器，4× A100 GPU，约 10 GPU-hours
- **检索器**：Contriever-MSMARCO
- **生成器**：Llama-3-8B-Instruct / GPT-3.5-turbo-1106
