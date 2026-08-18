---
title: "Tokenization Is More Than Compression"
source: https://aclanthology.org/2024.emnlp-main.40.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:10:11"
---

# 论文速读：Tokenization Is More Than Compression

## 一句话总结
本文通过提出基于最短路径的最小词元分段器 PATHPIECE 并训练 64 个控制变量的语言模型，系统验证了“CTC 越低下游性能越好”的假设，结果证实其并不成立；分词器的实际性能更多取决于预处理规则、词表初始化策略与分段方法的协同设计，而非单纯的文本压缩率。

## 研究问题与动机
- **核心假设存疑**：Gallé (2019) 与 Goldman et al. (2024) 等研究认为 BPE 的优势源于其能将文本压缩为较短的 token 序列，但缺乏对分词器内部各阶段的控制变量验证。
- **结论相互矛盾**：Ali et al. (2024) 发现 CTC 与下游性能无强相关，而部分早期工作却强调压缩率的重要性，亟需大规模统一基准澄清。
- **分词器设计黑盒**：现有研究多将 tokenization 视为单一模块，实际上它包含预处理（Pre-tokenization）、词表构建（Vocabulary Construction）与分段（Segmentation）三个阶段，各阶段的交互效应尚未被充分解耦。
- **缺乏开源可复现基准**：现有对比实验多使用闭源词表或不同架构，难以公平比较；本文旨在提供一套完整开源的实验设置与模型权重。

## 核心贡献（创新点）
- **提出 PATHPIECE 分词器**，通过 DAG 最短路径动态规划在给定词表下实现理论最小词元分段。与 BPE/Unigram 等贪心合并或概率剪枝策略不同，PATHPIECE 直接将 CTC 作为优化目标，为验证“纯压缩假说”提供了理想对照工具。
- **构建四维度控制变量实验框架**，将分词过程解耦为预处理、初始化、词表构建与分段，并在统一 MPT 架构下对比 18 种变体。不同于以往仅对比单一分词器的研究，本文首次量化了各阶段的独立贡献与跨阶段交互效应。
- **实证反驳“CTC 越低性能越好”的假设**，并提炼出可复用的工程指南。与基于经验观察的结论不同，本文通过 64 个模型与严格统计检验证明，分词质量更多取决于 FirstSpace 预处理与 BPE 词表初始化，而非单纯压缩率。
- **大规模开源资源**，公开 PATHPIECE 实现、64 个语言模型权重（350M/1.3B/2.4B）及全部词表，为社区提供了低成本、高可控性的分词器评估基准。

## 方法详解
- **PATHPIECE 分段（Segmentation）**：将文档视为有向无环图（DAG），每个字节位置为节点，若字节序列 $[j,i]$ 属于词表 $\mathcal{V}$ 则添加有向边。通过动态规划求解最短路径，记录每个位置的最短路径长度 $pl[i]$ 与对应 token 宽度 $wid[i]$，反向回溯构造最优分段。算法复杂度为 $O(nL)$，其中 $L=16$ 为最大 token 字节宽度。
- **PATHPIECE 词表构建（Vocabulary Construction）**：采用自顶向下剪枝策略，初始词表 $\mathcal{V}_0$ 可为高频 n-gram 或 BPE/Unigram 大词表（本文取 $2^{18}=262{,}144$）。对当前词表中每个 token $t$，计算其被移除后 CTC 的最小增量 $MI_t$，优先移除增量最小的 token。为避免 $O(nL|\mathcal{V}|)$ 重算，利用正向/反向最短路径 $pl[\cdot]$ 与 $bpl[\cdot]$ 高效推导两种替换情形（内部断裂或超集覆盖）的最小增量，单次迭代复杂度 $O(nL^2)$。
- **与 Unigram 的理论联系**：Unigram 的最大似然分段等价于带权 PATHPIECE（权重为 $-\log p(t_k)$）；反之令所有 token 概率相等即退化为最小化词元数目标，二者在算法层面可相互转化。
- **实验设计控制**：固定 MPT decoder-only 架构，预训练语料为 The Pile，分词训练语料为 MiniPile（6GB）。预处理方案包括 None、FirstSpace、Space、Digit 及组合；词表构建涵盖 BPE、WordPiece、Unigram、SaGe、PATHPIECE 及随机采样基线 RandTrain；分段策略涵盖 Merge、Greedy、Likelihood 与 PATHPIECE，并交叉组合以隔离各阶段效应。

## 实验与结果
