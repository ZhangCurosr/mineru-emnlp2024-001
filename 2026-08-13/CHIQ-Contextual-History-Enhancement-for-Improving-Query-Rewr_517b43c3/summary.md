---
title: "CHIQ-Contextual-History-Enhancement-for-Improving-Query-Rewr"
source: https://aclanthology.org/2024.emnlp-main.135.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:28:41"
field: "对话信息检索"
keywords: ["对话搜索", "查询重写", "历史增强", "开源LLM", "伪监督", "TopiOCQA", "QReCC"]
innovations: ["两步法：先利用开源LLM增强对话历史（消歧/扩展/伪响应/主题切换/摘要），再基于增强历史重写查询", "设计五种历史增强提示策略并系统评估各自贡献", "提出即席重写、搜索导向微调与结果级融合三种集成方式，在多数设置下达SOTA"]
benchmarks: ["TopiOCQA", "QReCC", "CAsT‑19", "CAsT‑20", "CAsT‑21"]
---

# 论文速读：CHIQ-Contextual-History-Enhancement-for-Improving-Query-Rewr

## 一句话总结
论文提出了 CHIQ，一种利用开源 LLM（LLaMA‑2‑7B/Mistral‑2‑7B）的两步查询重写方法：先通过五种 NLP 能力增强对话历史（消除歧义、扩展响应、伪响应生成、主题切换检测、历史摘要），再基于增强后的历史生成搜索查询。该方法在五个对话搜索基准上取得了 most settings 下的 state‑of‑the‑art 性能，并以开源模型竞争性超越了依赖闭源 LLM 的现有系统。

## 研究问题与动机
- **核心问题**：对话搜索中，多轮历史包含指代模糊、缩写、冗余信息，导致直接基于原始历史进行查询重写（CQR）时检索效果下降。
- **现有方法不足**：
  1. 主流 CQR 工作依赖闭源 LLM（如 ChatGPT）直接生成查询，成本高且难以自定义部署。
  2. 开源 LLM（如 LLaMA‑2‑7B）虽具备基础 NLP 能力，但未经过专门设计难以充分释放其对话搜索潜力。
  3. 直接增强历史或微调小型生成器（如 T5）的方式未系统考虑历史歧义的多维度解耦。

## 核心贡献（创新点）
1. **提出两步历史增强框架**：先利用开源 LLM 的 NLP 能力对对话历史进行多角度增强，再基于增强历史重写查询。与以往直接用 LLM 生成查询的本质区别在于“先清洗上下文，后生成查询”，更贴合小模型的推理边界。
2. **设计五种历史增强提示策略**（QD、RE、PR、TS、HS），分别对应消歧、响应扩展、伪响应生成、主题切换检测和历史摘要。与以往单一去噪或扩充方法相比，多维度解耦可覆盖历史中不同类型的噪声与缺失信息。
3. **提供三种查询生成集成方式**：即席重写（CHIQ‑AD）、搜索导向微调（CHIQ‑FT）以及两者的结果级融合（CHIQ‑Fusion）。与仅依赖单一重写路径的工作不同，Fusion 能兼顾不同方法生成的互补查询分布。
4. **在五个基准（TopiOCQA、QReCC、CAsT‑19/20/21）上全面验证**：CHIQ 使用开源 LLaMA‑2‑7B 在多数设置下达到 SOTA，且在多个设置上超越依赖闭源 LLM 的基线，证明了开源模型经历史增强后可达更高性价比。

## 方法详解
- **任务形式化**：给定对话历史 $\mathcal{H}=\{u_k, r_k\}_{k=1}^n$ 与新用户问题 $u_{n+1}$，目标是生成一个自包含的搜索查询 $q_{n+1}$，供检索器召回相关 passage。
- **历史增强五步法**：
  - **Question Disambiguation (QD)**：将当前轮次 $u_{n+1}$ 重写为无歧义、自包含的版本 $u'_{n+1}$。
  - **Response Expansion (RE)**：利用前文历史扩展最后一轮系统响应 $r_n$ 为更丰富的 $r'_n$。
  - **Pseudo Response (PR)**：基于历史与当前问题生成一段伪响应 $r'_{n+1}$，补充潜在相关术语。
  - **Topic Switch (TS)**：判断是否发生话题切换，若是则仅保留最近一轮作为有效历史，避免无关早期对话干扰。
  - **History Summary (HS)**：将增强后的历史压缩为一段简洁摘要，保留关键信息。
- **查询生成三种方式**：
  - **CHIQ‑AD**：将增强后的历史 $\mathcal{H}'$ 与 $u_{n+1}$ 直接输入开源 LLM，生成查询 $q'_{n+1}$。
  - **CHIQ‑FT**：离线利用增强历史+ gold passage $p^*_{n+1}$ 调用 LLM 生成多个伪查询 $Q'_{n+1}$，选取与 $p^*_{n+1}$ 检索得分最高的查询作为监督信号，微调小型 T5 模型。
  - **CHIQ‑Fusion**：将 CHIQ‑AD 与 CHIQ‑FT 的检索结果列表按 result‑level fusion（$\alpha=1$）合并。
- **关键公式**：
  1. 原始重写：$q_{n+1} \leftarrow \mathcal{LLM}(\mathcal{T}^{CQR} \oplus \mathcal{H} \oplus u_{n+1})$
  2. 伪监督生成：$Q'_{n+1} \leftarrow \mathcal{LLM}(\hat{\mathcal{T}}^{CQR} \oplus \mathcal{H}' \oplus u_{n+1} \oplus p^*_{n+1})$
  3. 监督选择：$q'_{n+1} \leftarrow \arg\max_{q'} S(Q'_{n+1}, p^*_{n+1}), q'\in Q'_{n+1}$

## 实验与结果
- **数据集**：TopiOCQA（训练 3,509 会话/2,514 轮，测试集），QReCC（训练 10,823 会话/8,124 轮，测试集），CAsT‑19/20/21（仅测试集）。
- **评估指标**：MRR、NDCG@3、Recall@10（dense 用 ANCE，sparse 用 BM25）。
- **主要结果**（LLaMA‑2‑7B backbone）：
  - **TopiOCQA dense**：CHIQ‑Fusion MRR 38.0，较最优闭源基线 LLM4CS（ChatGPT‑3.5）高出约 1.0 MRR；CHIQ‑AD 较 LLM4CS（LLaMA）提升 5.5% MRR。
  - **QReCC dense**：CHIQ‑AD MRR 47.0，超越 LLM4CS（LLaMA）约 2.2%。
  - **CAsT‑19 dense**：CHIQ‑Fusion MRR 73.3，超越 LLM4CS（LLaMA）约 4.9%；CHIQ‑AD 超越 LLM4CS（LLaMA）约 2.4%。
  - **稀疏检索（BM25）**：CHIQ‑Fusion 在 TopiOCQA 与 QReCC 上均显著优于多数基线（如 TopiOCQA MRR 25.6 vs LLM4CS‑LLaMA 18.9）。
- **开源 vs 闭源**：在原始历史上 LLaMA 与 ChatGPT‑3.5 的 MRR 差距为 1.9%–12.1%，使用增强历史后缩小至 0.9%–4.7%。
- **消融**：TS 对 TopiOCQA 贡献最大（dense 提升 13.2% MRR），PR 与 RE 对两者均有稳定增益；伪监督中引入历史增强、多查询生成、gold passage 条件均显著提升 FT 性能。

## 相关工作脉络
1. **传统 CQR（T5QR、QuReTeC、CONQRR 等）**：依赖小型模型微调或弱监督，未充分利用 LLM 的上下文理解能力；CHIQ 以开源 LLM 为骨干并引入历史增强。
2. **LLM 直接重写（LLM4CS、LLM‑Aided）**：直接调用闭源 LLM 生成查询；CHIQ 改用开源模型，并通过历史预处理降低歧义，降低对强大推理能力的依赖。
3. **Reranking/Embedding 微调（RepLLaMA、E5‑Mistral、LLM‑Embedder）**：针对 ad‑hoc 检索微调嵌入模型；CHIQ 聚焦对话场景下的查询重写，且无需重新训练嵌入器。
4. **对话密集检索（ConvDR、InstructorR、RETPO）**：端到端建模历史与文档匹配；CHIQ 仍基于 CQR 范式，强调与现成检索器/排序器的即插即用。
5. **历史去噪/摘要（EDIRCS、Mo et al. 2023b/c）**：侧重上下文过滤或对比学习；CHIQ 的系统性多维度增强（QD+RE+PR+TS+HS）覆盖更多噪声类型。

## 局限性与未来方向
- **未实验更大规模开源模型**（如 56B Mixtral、70B LLaMA）与更多闭源模型（GPT‑4、Claude），性能上限未探明。
- **计算与财务资源限制**，导致消融与跨模型比较不够充分。
- **未来方向**：
  - 增加回退过滤策略，检测并屏蔽 LLM 产生的噪声输出。
  - 探索混合监督信号（人工重写查询与伪查询插值）以改善 CHIQ‑FT 的训练质量。
  - 扩展更多历史增强策略（如实体链接、事实核查）以进一步降低歧义。

## 研究启发与可借鉴点
1. **历史增强范式可迁移**：将对话历史视为可优化的中间表示，通过多维度 NLP 任务（消歧、摘要、伪生成）提升下游任务输入质量，这一思路可推广至多轮对话问答、对话式推荐等场景。
2. **伪监督信号生成方法**：利用 gold passage 条件引导 LLM 生成多个候选查询，并选取检索得分最高者作为监督，兼顾了多样性与检索友好性，可作为轻量级数据合成模板。
3. **消融实验设计严谨**：逐一关闭 QD、RE、PR、TS、HS，定量评估各组件贡献，并对比 dense/sparse 两种检索设置，为后续研究提供了清晰的对照基线。
4. **开源模型性价比论证**：系统展示开源 7B 模型经历史增强后可接近甚至超越闭源 3.5 模型，为资源受限团队提供了可复现的低成本替代方案。

## 关键术语表
- **CHIQ**：本文提出的两阶段方法，先增强对话历史再重写查询。
- **CQR（Conversational Query Rewriting）**：对话搜索中将上下文相关 utterance 转换为独立搜索查询的任务。
- **Question Disambiguation (QD)**：将模糊的用户问题重写为自包含、无歧义的形式。
- **Pseudo Response (PR)**：利用 LLM 基于历史生成一段伪响应，以补充潜在相关术语。
- **Topic Switch (TS)**：检测对话是否发生话题跳转，若是则截断历史仅保留最近一轮。
- **History Summary (HS)**：将增强后的历史压缩为简洁段落，保留关键信息。
- **CHIQ‑FT**：基于增强历史与伪监督信号微调小型 T5 模型的查询重写器。
- **CHIQ‑Fusion**：将 CHIQ‑AD 与 CHIQ‑FT 的检索结果列表按线性组合融合。

## 可复现要素
- **数据集**：TopiOCQA、QReCC、CAsT‑19/20/21（均公开）。
- **代码**：已开源，见 https://github.com/fengranMark/CHIQ。
- **权重**：使用 LLaMA‑2‑7B‑instruct 与 Mistral‑2‑7B‑instruct 的官方 instruct‑tuning 变体；T5‑base/large 为公开模型。
- **关键超参**：fine‑tuning 10 epochs，lr=1e‑5，batch size=8；生成时 temperature=0.7，max length=32 tokens；BM25 参数 $k_1=0.9, b=0.4$（TopiOCQA）与 $k_1=0.82, b=0.68$（QReCC）；检索器截断长度 query=32、concatenated input=512、passage=384 tokens。
