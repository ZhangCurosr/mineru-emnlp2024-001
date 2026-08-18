---
title: "How-do-Large-Language-Models-Learn-In-Context-Query-and-Key"
source: https://aclanthology.org/2024.emnlp-main.192.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:21:12"
field: "大语言模型可解释性与上下文学习机制"
keywords: ["in-context learning", "mechanistic interpretability", "query-key two towers", "majority label bias", "recency bias", "causal tracing"]
innovations: ["识别仅占1%的in-context heads并证明其因果决定性", "提出value-output提取标签特征、query-key双塔计算相似度的ICL机制假设", "基于头级机制解释并缓解多数标签偏差与近因偏差"]
benchmarks: ["Financ", "AGnews", "Amazon", "ETHOS", "SST2"]
---

# 论文速读：How-do-Large-Language-Models-Learn-In-Context-Query-and-Key

## 一句话总结
论文通过因果追踪与梯度重要性方法识别出仅占约 1% 的 in-context heads，并提出 ICL 的“双塔度量”假设：在深层 heads 中，value-output 矩阵负责提取标签特征，query-key 矩阵作为双塔计算末位 token 与各标签位置特征间的相似度以控制标签信息流动；基于该假设，论文解释了 ICL 的多数标签偏差与近因偏差，并提出两种轻量干预，分别将两类偏差降低约 22% 与 17%。

## 研究问题与动机
- ICL 的核心机制仍不清楚，已有工作多对各层 attention head 取平均观察信息流，难以定位精确起作用的 head。
- 每个 head 由 Q/K/V/O 四个矩阵构成，其在 ICL 中各自扮演的角色缺乏细粒度实证分析。
- ICL 长期受多数标签偏差（majority label bias）与近因偏差（recency bias）困扰，现有研究缺乏基于头级机制的统一解释与可操作的缓解方案。
- 语义无关标签（foo/bar）已被用于分离任务学习（TL）与预训练先验，但 TL 的内部计算路径尚未被清晰刻画。

## 核心贡献（创新点）
- **精确定位 1% 关键头**：提出因果追踪与 logit 增益双重筛选，识别出 12 个 in-context heads（6 个 fooheads、6 个 barheads），干预后五数据集平均 ICL 准确率从 87.6% 降至 24.4%。  
  与已有工作的区别：不同于此前对各层注意力取平均的做法，本文在 head 粒度上给出可直接干预的因果证据。
- **提出 ICL 双塔度量假设**：认为 in-context heads 中，value-output 矩阵提取标签特征，query-key 矩阵充当双塔计算末位 token 与每个演示标签位置的语义相似度，相似度越高对应标签概率越大。  
  与已有工作的区别：将此前偏信息流描述的机制升级为可量化的“度量学习”视角，并给出头内矩阵分工的证据链。
- **从机制出发解释两类经典偏差**：多数标签偏差源于同标签演示越多、对应相似度之和越大；近因偏差源于位置编码在 Q/K 相似度计算中引入 position term，导致晚期演示被放大。  
  与已有工作的区别：将经验性偏差转化为 query-key 注意力分配与位置编码影响的直接推论。
- **提出两种可执行的偏差缓解方法**：对少数标签位置的加权 value-output 进行注意力放大以降低多数标签偏差 22%；在 in-context heads 中移除位置编码影响以降低近因偏差 17%（注意力标准差下降约 40%）。  
  与已有工作的区别：方法直接作用于已定位的 in-context heads，而非全局提示工程或重采样。

## 方法详解
- **关键头识别**：
  - 方法一（因果追踪）：将深层层各 head 参数置零，重算五数据集准确率下降幅度。
  - 方法二（logit 增益）：计算每个 head 对预测标签 b 的概率增益  
    $S_l^h = \log p(b \mid o_l^h + \text{Lin}_l) - \log p(b \mid \text{Lin}_l)$，其中 $o_l^h$ 为 head 输出向量，$\text{Lin}_l$ 为第 l 层输入，最终通过 $D_v = \text{softmax}(E_u v)$ 映射到词表空间。
  - 两方法各自 top-10 交叉取并，得到 12 个 in-context heads（6 fooheads、6 barheads）。
- **双塔假设的证据链**：
  - 浅层：演示特征被提取并汇聚到标签位置与末位位置（呼应 Wang et al., 2023）。
  - 深层 in-context heads 的 value-output 矩阵：在对应标签位置上的加权 value-output 向量在词表空间中具有高标签相关词（如 foo-heads 的 foo 位置 top tokens 含 foo/Foo/FO 等），logit-minus $M = \log p(\text{foo} \mid \alpha^p \cdot vo^p) - \log p(\text{bar} \mid \alpha^p \cdot vo^p)$ 在匹配位置显著为正（表 4，约 0.26~0.32），在冲突位置为负（约 -0.02~-0.14）。
  - 深层 in-context heads 的 query-key 矩阵：通过 $\alpha^p = \text{softmax}(q_{\text{last}} \cdot k_p^T)$ 计算末位与每标签位置的相似度；预测由 foo 翻转为 bar 时（表 6），fooheads 在 foo 位置的注意力显著下降（-21.8%~-43.0%），barheads 在 bar 位置显著上升（+86.0%~+237.8%），是预测翻转的主要驱动。
- **多数标签偏差机制与干预**：
  - 机制：softmax 约束下，某标签演示数越多，对应 key 位置的注意力权重之和越大，最终概率偏向该标签。
  - 干预：对少数标签位置加权 value-output 向量乘放大系数 $a = a_c \cdot a_v$（$a_v$ 为多数/少数演示数比，论文取 $a_c = 0.03$），再叠加到最终嵌入，平衡相似度之和。
- **近因偏差机制与干预**：
  - 机制：位置编码通过 layer input 持续进入 Q/K 向量的构造，导致相似晚期位置在浅层特征提取与深层注意力中均被放大；改变演示顺序会改变末位 token 中不同位置信息的混合比例（如 X% 末位 + Y% 中位 + Z% 远位），从而改变预测。
  - 干预：在 in-context heads 中移除位置编码对 attention score 的影响，等效于为 each in-context head 加一条 shortcut adapter，重新计算输出向量。

## 实验与结果
- **数据集**：Financ、AGnews、Amazon、ETHOS、SST2 五个句子分类数据集，均使用语义无关标签 foo/bar；每数据集随机采样 1,000 句。
- **模型**：Llama-7B（32 层 × 32 heads）与 GPT-J（28 层 × 16 heads）。
- **关键头有效性**：对 fooheads 干预后，正确标签为 foo 的准确率显著下降，正确标签为 bar 时反而上升；barheads 干预结果相反（表 3），印证了头级特异性。
- **机制验证**：
  - 表 4：fooheads 在 foo 位置的 logit-minus 显著为正（~0.26~0.32），barheads 在 bar 位置显著为负（~-0.11~-0.20），表明 value-output 向量在匹配标签位置携带强标签信息。
  - 表 5/6：fooheads 在 foo 位置的 attention score 远高于 bar 位置，且预测翻转时注意力变化幅度（-20%~-43% / +86%~+238%）显著大于 value-output 向量的变化，证实 query-key 矩阵是预测迁移的主因。
- **最强结果与提升幅度**：
  - 基线：五数据集平均 ICL 准确率 87.6%；对 12 个 in-context heads 干预后降至 24.4%。
  - 多数标签偏差缓解：平衡 vs. 少一个演示数据的准确率波动平均减少 22%（Llama 29.1%、GPT-J 14.9%，表 7）。
  - 近因偏差缓解：不同演示顺序下的准确率标准差平均下降 17%（Llama 23.4%、GPT-J 10.6%，表 8）；in-context heads 内注意力标准差下降约 40%（Llama 40.1%、GPT-J 37.7%）。

## 相关工作脉络
- **Pan et al. (2023)**：提出 ICL 可分解为任务识别（TR）与任务学习（TL）；本文在此基础上进一步剖析 TL 阶段 in-context heads 内部 Q/K/V/O 的分工，并给出度量学习形式化。
- **Wang et al. (2023)**：指出 label words 在浅层充当锚点、在深层向末位聚合信息；本文验证该流向，并以 head 粒度与双塔相似度机制给出更细粒度的解释。
- **Wei et al. (2023)**：使用 foo/bar 语义无关标签证明 ICL 依赖演示-标签映射；本文沿用该设定并在此之上定位具体负责该映射的关键 heads。
- **Xie et al. (2021)、Garg et al. (2022)、Akyürek et al. (2022)、Von Oswald et al. (2023)**：从理论角度将 ICL 解释为隐式贝叶斯推断或梯度下降；本文提供与之相容的实证机制路径（双塔度量），将抽象算法落实为可观测的 head 内部操作。
- **Zhao et al. (2021)、Lu et al. (2021)**：经验性揭示多数标签偏差与近因偏差；本文首次以 query-key 注意力分配与位置编码影响给出机制层面的解释并提出针对性缓解。

## 局限性与未来方向
- 聚焦于深层 in-context heads 的机制，浅层如何将演示特征投影到标签位置与末位位置的系统性分析仍待补充。
- 假设主要经句子分类任务验证，尚未扩展到 chain-of-thought 推理、机器翻译、信息抽取等其他 ICL 任务。
- 关键头的识别依赖因果追踪与 saliency-based logit 增益两类方法，而重要性归因目前缺乏统一标准，结果的跨模型/跨规模稳定性有待检验。
- 近因偏差缓解仅针对深层 in-context heads 移除 position term，浅层特征提取阶段同样受位置影响（图 3 残差差异），尚未一并处理。
- 放大系数 $a_c = 0.03$ 等超参需实验调优，跨数据集与不同模型体量的迁移性仍需更多验证。

## 研究启发与可借鉴点
- **语义无关标签设计**：以 foo/bar 剥离预训练语义先验，可稳定隔离 IC 任务学习能力，适合用于机制研究与偏差消融实验。
- **head 粒度因果+梯度联合定位**：将 causal tracing 与 logit gain 两路证据交叉，能更稳健地定位功能头，可迁移到其它模型的模块重要性分析。
- **双塔度量视角**：将 Q/K 视为双塔、V/O 视为特征投影，为理解与改进 IC 提供了可扩展的建模语言；可在下游任务中尝试显式引入双塔相似度头作为正则或辅助任务。
- **轻量干预范式**：放大少数标签位置注意力权重、移除特定 head 的 position term 均为低成本的接口级干预，便于在基准上快速验证机制假设。
- **偏差评测指标**：本文用准确率波动幅度与注意力标准差共同刻画偏差，形成可复用的评测框架，便于横向比较不同缓解策略。

## 关键术语表
- **In-Context Learning (ICL)**：在 prompt 中提供若干示例对后，大模型无需更新参数即可执行目标任务的能力。
- **In-Context Heads**：对 ICL 预测影响最显著的少量 attention heads；本文识别到 12 个。
- **Fooheads / Barheads**：分别专门增强预测 "foo" 或 "bar" 概率的 in-context heads 子集。
- **Value-Output 矩阵 (VO)**：从各位置输入线性变换生成 value-output 向量，负责在标签位置提取并保留标签语义信息。
- **Query-Key 双塔**：Q 与 K 矩阵分别在末位 token 与各标签位置编码特征，并计算两者语义相似度，作为 ICL 信息路由的度量器。
- **Majority Label Bias**：ICL 中模型倾向于预测出现次数更多的标签的偏差。
- **Recency Bias**：ICL 中模型对演示序列中靠后的示例赋予过高注意力的偏差。
- **Causal Tracing**：通过置零/注入扰动头或向量并观测输出变化，定位对最终预测具因果影响的计算路径。

## 可复现要素
- 数据集：Financ、AGnews、Amazon、ETHOS、SST2（公开数据集）；每个数据集随机采样 1,000 句。
- 模型：Llama-7B、GPT-J（开源权重）。
- 代码与数据：论文声明将于 https://github.com/zepingyu0512/in-context-mechanism 开源。
- 关键超参：重要性筛选取两类方法各自 top-10；多数偏差放大系数 $a_c = 0.03$；演示数量 2~4 条/标签。
- 其他：因果追踪（置零 head 参数）、logit 增益公式、词表空间投影 $D_v = \text{softmax}(E_u v)$；近因偏差实验中对比原始与 reverse 顺序并移除 in-context heads 的 position embedding 影响。
