---
title: "Rethinking-Token-Reduction-for-State-Space-Models"
source: https://aclanthology.org/2024.emnlp-main.100.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:16:38"
field: "高效大模型推理"
keywords: ["State Space Model", "Token Reduction", "Mamba", "Post-training Optimization", "Token Pruning", "Token Merging", "Efficient Inference"]
innovations: ["首次揭示SSM上token reduction失效机理并提出UTRC统一框架", "基于clipped hidden state sum的token重要性度量适配SSM高维通道空间", "hybrid pruning-merging混合策略配合intra-layer跨分支对齐设计"]
benchmarks: ["LAMBADA", "HellaSwag", "PIQA", "Arc-E", "Arc-C", "WinoGrade"]
---

# 论文速读：Rethinking-Token-Reduction-for-State-Space-Models

## 一句话总结
本文发现现有Transformer的token reduction方法直接迁移到State Space Models（如Mamba）会因忽视token重要性而导致严重性能下降，因此提出一种面向SSM的统一post-training token reduction方法（UTRC），通过融合pruning与merging并引入基于hidden states的token重要性度量，在降低FLOPS与显存占用的同时保持模型精度。

## 研究问题与动机
- **核心问题**：如何将token reduction技术有效应用于SSM架构（如Mamba），以提升其推理效率并降低内存需求？
- **现有方法不足**：
  - **Pruning失效**：直接剪枝虽移除"低重要性"token，但在SSM的序列递推计算中，被剪除token的信息损失会逐级累积放大，导致精度大幅下降（如图1所示，Mamba-2-8B剪枝20% FLOPS后平均准确率暴跌）。
  - **Merging失效**：现有合并方法（如ToMe、PuMer）均匀分区后合并token，未考虑token的内在重要性，可能将高价值token合并到无关token中，造成更大性能损失。
- **动机来源**：SSM缺少attention机制，无法直接复用基于注意力权重的token筛选策略；同时SSM的高维通道空间（channel space）蕴含更细粒度的token间交互信息，需重新设计重要性评估方式。

## 核心贡献（创新点）
- **首次揭示SSM上token reduction失效机理**：系统分析了pruning与merging在SSM上的失败原因，指出pruning的信息累积损失问题与merging对token重要性的忽视是两大核心缺陷。
- **提出统一post-training token reduction框架（UTRC）**：首次针对SSM设计融合pruning与merging的混合策略，通过token重要性分类+相似度匹配实现细粒度intra-layer reduction。
- **重新设计token重要性度量**：提出基于SSM hidden states clipped sum的指标（而非直接套用Transformer的attention或L1/L2 norm），充分利用SSM高维通道空间捕捉token间细粒度交互。
- **层级化reduction位置设计**：提出在每个第10层后及每5层应用reduction的层级策略，在早层高效压缩与充分表征重要性之间取得平衡。
- **零样本泛化验证**：在Mamba-2（1.3B/2.7B）与Mamba（1.4B/2.8B）四个模型上验证，平均准确率较最优基线提升5.7%–13.1%，GPU峰值内存最高降低40%。

## 方法详解
- **Token重要性度量（Equation 5）**：对SSM层输出的hidden states $\mathbf{y} \in \mathbb{R}^{B \times N \times D'}$，按特征维度做max-clipping后求和再归一化：$\mathcal{S} = \frac{\sum_{d=1}^{D'} \max(0, [\mathbf{y}]_{::d})}{D'}$，得到每个token的重要性分数。消融表明clip操作优于l1/l2 norm与无clip版本。
- **Token重要性分类**：将所有token按重要性分数排序，较低一半归入集合 $M_A$（低重要性），较高一半归入 $M_B$（高重要性）。
- **相似性匹配（Equations 6–7）**：对每个 $a_i \in M_A$，在其余token中找到余弦相似度最高的 $f_i \in M_B$，记录相似度得分 $g_i$。
- **保留Top-$p\%$连接**：按 $g_i$ 排序后仅保留最相似的 $p\%$ 连接，用于后续reduction操作。
- **Unified Token Reduction（UTR）**：对保留的连接执行混合策略——其中 $(p \times q)\%$ 执行pruning（删除 $a_i$），剩余 $[p \times (1-q)]\%$ 执行merging（将 $a_i$ 与 $f_i$ 特征平均后删除 $a_i$），消融得 $q=0.5$ 最优。
- **Intra-layer设计差异**：hidden states分支采用hybrid reduction（pruning+merging），residual connection分支仅采用merging，以对齐两分支的token index避免错位。
- **层级化reduction位置**：Mamba-2-2.7B/2.8B在层 [12, 17, 22, 27, 32, 37, 42] 执行；Mamba-2-1.3B/Mamba-1.4B在层 [10, 15, 20, 25, 30, 35] 执行，固定每层压缩比。

## 实验与结果
- **模型与数据集**：Mamba-2-1.3B、Mamba-2-2.7B、Mamba-1.4B、Mamba-2.8B；六个zero-shot基准：LAMBADA（PPL↓）、HellaSwag、PIQA、Arc-E、Arc-C、WinoGrade（Acc↑）。
- **基线**：PuMer（面向VLM的pruning+merging）与EViT（vision Transformer剪枝）。
- **主要结果（Table 1）**：Mamba-2-1.3B，20% FLOPS reduction下，平均准确率54.6%（EViT仅44.2%，差距10.4%）；Mamba-2-2.7B，30% FLOPS reduction下，平均准确率58.7%（EViT 41.6%，差距17.1%）。
- **主要结果（Table 2）**：Mamba-1.4B，20% FLOPS reduction下平均准确率53.4%（EViT 41.1%）；Mamba-2.8B，20% FLOPS reduction下平均准确率57.6%（EViT 43.6%）。
- **最强结果**：Mamba-2-2.7B + 30% FLOPS reduction，平均准确率58.7%，LAMBADA PPL 17.96，GPU峰值内存降低40.0%；Mamba-2.8B在30% FLOPS reduction下吞吐量提升1.29×。
- **消融（Table 3–5）**：Clip重要性度量最优；reduction位置 [12,17,22,27,32,37,42] 最佳；hidden states q=0.5 + residual merging-only 组合表现最好（40.61 PPL / 54.7% 平均准确率，30% FLOPS reduction）。

## 相关工作脉络
- **SSM架构（Mamba系列）**：本文与Mamba（Gu & Dao, 2023）及Mamba-2（Dao & Gu, 2024）形成对照——前者关注模型结构本身，本文聚焦已有预训练大SSM的post-training效率优化。
- **Token Pruning（EViT / DynamicViT / SPViT）**：Transformer剪枝方法依赖attention权重或[CLS] token评分；本文指出SSM无attention层，该类方法直接迁移失效。
- **Token Merging（ToMe / PuMer）**：基于token key相似度进行二分图合并；本文指出其忽视token内在重要性，易将高价值token错误合并。
- **SSM视觉应用（S4ND / ViM）**：本文扩展了SSM效率优化的研究版图，从单纯的模型架构设计延伸至推理加速策略。
- **激活感知剪枝（Agile-Quant）**：同类工作探索LLM推理优化，但针对量化而非token长度维度；本文填补了SSM在token级压缩方向上的空白。

## 局限性与未来方向
- **未涉及微调**：实验仅在zero-shot设置下验证，未评估fine-tuning阶段进一步适配token reduction是否可提升性能。
- **未在Transformer LLM上验证**：方法虽声称可泛化到Transformer，但缺乏对其他Transformer大模型的实证（如LLaMA等）。
- **统一超参的假设**：固定 $p$、$q=0.5$ 等超参在不同任务与序列长度下可能非最优，缺乏自适应选择机制。
- **未来方向**：探索fine-tuning阶段联合训练reduction策略、泛化至更多架构（Transformer LLM、多模态SSM）、设计任务自适应的压缩比与层次选择。

## 研究启发与可借鉴点
- **CLIP重要性度量的设计思路**：将hidden states按通道做max-clipping后求和，既能过滤负值噪声又保留正激活信息，该思路可迁移到其他非attention架构的token评估。
- **Hybrid reduction策略**：在同一框架内统一pruning与merging并通过超参 $q$ 控制比例，兼顾信息保留与冗余压缩，可作为通用token压缩范式。
- **Intra-layer跨分支对齐设计**：针对hidden state与residual两条分支分别设计不同reduction方式并强制index对齐，解决了多分支reduction中的错位问题，对多路径架构优化有借鉴价值。
- **层级化reduction位置策略**：通过消融发现早期层+固定间隔的reduction优于深层或逐层reduction，为后续工作的位置搜索提供了可复用的实验基准。
- **Post-training通用性**：无需重新训练即可直接应用于不同规模的SSM，部署成本低，适合大规模模型的快速迭代优化。

## 关键术语表
- **State Space Model (SSM)**：一类基于状态方程建模序列数据的架构，通过隐藏状态的递归更新实现长程依赖建模，代表工作包括Mamba。
- **Selective SSM (Mamba)**：引入时变参数（$\Delta$）与选择性机制的SSM变体，可动态决定遗忘或保留信息，支持并行训练与推理。
- **Token Pruning**：根据token重要性评分剔除低价值token，减少序列长度以降低计算开销。
- **Token Merging**：将多个相似token的特征融合（如平均）替换为单个token，以降低token数量。
- **UTRC（Unified Token Reduction by token Importance Classification）**：本文提出的方法，通过token重要性分类+相似度匹配实现pruning与merging的统一融合。
- **FLOPS Reduction**：通过减少token数量实现的计算量压缩比例，用于衡量推理加速效果。
- **Clipped Hidden State Sum**：本文提出的token重要性指标，对SSM输出的hidden state按特征维度做max(0,·)后求和并归一化。

## 可复现要素
- **数据集**：LAMBADA、HellaSwag、PIQA、Arc-E、Arc-C、WinoGrade（均为公开benchmark）。
- **代码/权重**：论文未提及开源代码与权重；模型基于公开Mamba仓库（HuggingFace）。
- **关键超参**：剪枝比例 $p$（对应10%/20%/30% FLOPS reduction）、hybrid混合系数 $q=0.5$、reduction层级间隔（每5层，起始层视模型大小而定）、相似度保留阈值。
- **训练环境**：NVIDIA A100 80GB GPU，PyTorch + HuggingFace Transformers。
- **评估调整**：输出token被缩减后，label logits同步截断至前 $(1-m\%)$ 部分计算PPL与准确率。
