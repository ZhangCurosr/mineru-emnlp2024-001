---
title: "In-context-Contrastive-Learning-for-Event-Causality-Identifi"
source: https://aclanthology.org/2024.emnlp-main.51.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:22:24"
field: "事件因果推理"
keywords: ["Event Causality Identification", "In-context Learning", "Contrastive Learning", "Prompt Learning", "ECI", "Natural Language Understanding"]
innovations: ["提出ICCL框架，结合上下文学习与对比学习增强ECI任务", "利用事件偏移向量构建对比损失，显式区分正负示例语义", "简单cloze模板+虚拟词替代复杂多提示设计，降低工程复杂度"]
benchmarks: ["EventStoryLine Corpus (ESC)", "Causal-TimeBank Corpus (CTB)"]
---

# 论文速读：In-context-Contrastive-Learning-for-Event-Causality-Identification

## 一句话总结
本文提出 **ICCL**（In-Context Contrastive Learning）模型，将**上下文学习**与**对比学习**结合用于事件因果关系识别（ECI）任务，通过对比损失增强正/负示例的类比效果，在 ESC 和 CTB 数据集上达到最新性能。

## 研究问题与动机
- **ECI 任务挑战**：判断文档中两个事件提及之间是否存在因果关系，是问答、阅读理解、事件预测等应用的基础。
- **现有 Prompt 学习方法局限**：DPJL 等方法依赖复杂的多提示设计与主/衍生任务间的正相关假设，工程成本高且泛化受限。
- **In-Context Learning 不足**：单纯拼接正/负示例进行类比预测，未显式区分两者语义差异，类比信号较弱。
- **核心动机**：利用对比学习显式拉拢相似（正）事件对、推远相异（负）事件对，从而强化上下文示例的判别性表示，提升 ECI 性能。

## 核心贡献（创新点）
1. **提出 ICCL 框架**：首次将 in-context learning 与 contrastive learning 联合用于 ECI，通过对比损失优化事件对表示，而非仅依赖 prompt 模板设计。
2. **基于事件偏移向量的对比学习**：利用两个事件 hidden state 的差值（$\mathbf{h}_{e_1}-\mathbf{h}_{e_2}$）作为关系向量，使正/负示例在嵌入空间中明确分离。
3. **简单的 cloze 提示模板**：放弃复杂的衍生任务，仅用 `[start] E1 [MASK] E2 [end]` 结构配合正负示例，降低工程复杂度并提升可解释性。
4. **系统验证与消融**：在 ESC 和 CTB 上全面对比 10+ 基线，证明 ICCL 在句内/跨句因果识别上均显著优于 SOTA，并深入分析示例数量、$\beta$ 超参、少样本场景的影响。

## 方法详解
ICCL 包含三个模块：

1. **Prompt Learning Module**  
   - 将输入事件对 $\{E_1, E_2, S_1, S_2\}$ 与 $K$ 个检索到的示例（$M$ 个正例 $d^+$，$N$ 个负例 $d^-$）拼接为统一 prompt。
   - 查询模板：$T_p(q)=S_1^q+S_2^q+[\texttt{start}]+E_1^q+[\texttt{MASK}]+E_2^q+[\texttt{end}]$  
   - 示例模板：$T_a(d_k)=S_1^k+S_2^k+[\texttt{start}]+E_1^k+y^k+E_2^k+[\texttt{end}]$  
   - 输入序列：$[CLS]+T_a(d_1^+)[SEP]+\dots[T_a(d_M^+)]+[SEP]+T_a(d_1^-)[SEP]+\dots[T_a(d_N^-)]+[SEP]+T_p(q)[SEP]$

2. **In-context Contrastive Module**  
   - 取 PLM 输出的事件提及 hidden state，构造偏移向量：$\mathbf{z}^q=\mathbf{h}_{e_1}^q-\mathbf{h}_{e_2}^q$，$\mathbf{z}_m^+=\mathbf{h}_{e_1}^{m^+}-\mathbf{h}_{e_2}^{m^+}$，$\mathbf{z}_n^-=\mathbf{h}_{e_1}^{n^-}-\mathbf{h}_{e_2}^{n^-}$。
   - 采用监督对比损失：  
     $L_{con}=-\log\sum_{m=1}^{M}\frac{\exp(\sin(\mathbf{z}^q,\mathbf{z}_m^+)/\tau)}{\sum_{d\in\mathcal{D}}\exp(\sin(\mathbf{z}^q,d)/\tau)}$，其中 $\mathcal{D}=\{\mathbf{z}_m^+\}\cup\{\mathbf{z}_n^-\}$，$\tau=1.0$ 为温度系数。

3. **Causality Prediction Module**  
   - 取 `[MASK]` 位置 hidden state $\mathbf{h}_{mask}$，对新增虚拟词 $\mathcal{V}_a=\{<\texttt{causal}>, <\texttt{none}>\}$ 做 softmax 分类。
   - 预测损失：$L_{pre}=-\frac{1}{L}\sum_l \mathbf{y}^{(l)}\log(\hat{\mathbf{y}}^{(l)})+\lambda\|\theta\|^2$（L2 正则）。
   - 总损失：$L_{total}=L_{pre}+\beta*L_{con}$，$\beta=0.5$ 为平衡系数。

训练时使用 AdamW 优化器，学习率 $1e{-5}$，batch size=16，RoBERTa-base（768 维）。

## 实验与结果
- **数据集**：EventStoryLine Corpus (ESC)（5,334 事件提及，5,625 因果对，5-fold CV）；Causal-TimeBank (CTB)（7,608 标注对，仅句内因果，10-fold CV）。
- **基线**：ILP、KnowMMR、RichGCN、CauSeRL、LSIN、LearnDA、GESI、ERGO、DPJL、SemSln。
- **最佳结果（RoBERTa）**：
  - **ESC 句内**：P=67.5%，R=73.7%，F1=70.4%（超越 DPJL 的 67.9%）
  - **ESC 跨句**：P=60.3%，R=62.7%，F1=61.3%
  - **ESC 总体**：P=62.6%，R=66.1%，F1=64.2%
  - **CTB 句内**：P=63.7%，R=68.8%，F1=65.4%（超越 DPJL 的 64.6%）
- **提升幅度**：在 ESC 句内 F1 上较前一 SOTA（DPJL）提升 **+2.5 个百分点**，CTB 提升 **+0.8 个百分点**。
- **消融验证**：完整 ICCL > In-context > Prompt；对比模块显著提升性能；示例数增加时 ICCL 持续改善而 In-context 下降；少样本（20% 数据）下 ICCL F1=51.9% 仍优于全量 ERGO（50.9%）。

## 相关工作脉络
1. **Graph-based ECI**（RichGCN、GESI、ERGO、SemSln）：构建文档级事件图进行节点/边分类，依赖图神经网络捕获全局结构；ICCL 放弃显式图结构，改用 prompt+对比学习挖掘 PLM 隐式知识。
2. **Prompt-based ECI**（DPJL、KnowMMR）：DPJL 设计主 cloze 任务 + 两个衍生任务；ICCL 仅用单一 cloze 模板 + 对比损失，避免多任务依赖。
3. **In-context Learning for NLP**（Dong et al., 2022）：通用 IC LLM 范式；本文将其适配到 ECI 并引入对比学习区分正负示例。
4. **Supervised Contrastive Learning**（Khosla et al., 2020）：标准对比损失；本文将其应用于事件对偏移向量，并融入 prompt 上下文。
5. **Low-resource / Few-shot ECI**：ERGO 等在全量数据上表现较强；本文证明 ICCL 在 20% 数据下仍能超越全量基线，体现小样本鲁棒性。

## 局限性与未来方向
- **示例数量有限**：受 PLM 输入长度限制，正/负示例各仅 2 个，对比学习样本多样性不足，可能削弱损失有效性。
- **PLM 敏感性**：不同 PLM 性能差异明显（RoBERTa 最优，T5 生成式模型表现差），需针对任务选择编码器架构。
- **零样本场景不佳**：GPT-3.5/4 等 LLM 零样本性能远低于微调方法，说明因果推理仍依赖领域微调。
- **未来方向**：作者计划将 ICCL 框架迁移至其他 NLP 任务，验证泛化能力；可探索更大规模示例检索、自动示例选择、跨句因果的长序列建模等。

## 研究启发与可借鉴点
1. **对比学习用于上下文示例增强**：将正负示例视为对比正负样本，可有效提升 few-shot 类比学习效率，可迁移至关系抽取、事件抽取等序列标注任务。
2. **事件偏移向量表征关系**：$\mathbf{h}_{e_1}-\mathbf{h}_{e_2}$ 简单且有效，避免了复杂的交互模块，可作为通用事件对表示方案。
3. **简单 cloze 模板+虚拟词**：放弃复杂提示工程，通过新增 `<causal>/<none>` 虚拟词直接对齐预训练目标，降低提示设计成本。
4. **少样本鲁棒性验证**：在 20% 数据下仍超越全量基线，为低资源场景下的 ECI 研究提供可行路径。
5. **PLM 选型敏感性分析**：系统对比 BERT/RoBERTa/ERNIE/DeBERTa/T5，证明编码器优于生成器适合因果推理，可为类似任务提供选型参考。

## 关键术语表
- **Event Causality Identification (ECI)**：判断文档中两个事件提及之间是否存在因果关系的信息抽取任务。
- **In-context Learning (ICL)**：在 prompt 中直接提供若干带标签示例，引导模型类比预测的学习范式。
- **Contrastive Learning**：通过对比损失拉近正样本对、推远负样本对，优化表示空间判别性的自监督/监督学习方法。
- **Cloze Task**：将下游任务转化为填空形式，利用预训练语言模型的掩码预测能力进行推理。
- **Event Mention Offset Vector**：两个事件 hidden state 的差值，用于表征事件对之间的语义关系。
- **Supervised Contrastive Loss**：在对比学习中引入样本标签监督信号，以类别一致性优化嵌入分布的损失函数。
- **RoBERTa**：Facebook 提出的优化版 BERT，移除 NSP 任务、更大规模训练，本文作为主干编码器取得最佳效果。
- **EventStoryLine Corpus (ESC)**：包含 22 主题新闻文档的事件因果标注数据集，用于评估句内/跨句因果识别。

## 可复现要素
- **数据集**：EventStoryLine 0.9 Corpus (ESC)、Causal-TimeBank Corpus (CTB)（公开可用，遵循标准划分：ESC 5-fold CV，CTB 10-fold CV 句内子集）
- **代码/权重**：论文未提供开源代码与预训练权重（仅注明基于 PyTorch 与 HuggingFace transformers）
- **关键超参**：
  - 学习率：$1e{-5}$
  - Batch size：16
  - $\beta$（对比损失权重）：0.5
  - 温度系数 $\tau$：1.0
  - 示例数量 $(M,N)$：(2,2)
  - 优化器：AdamW + L2 正则
  - 主干模型：RoBERTa-base（768 维）
