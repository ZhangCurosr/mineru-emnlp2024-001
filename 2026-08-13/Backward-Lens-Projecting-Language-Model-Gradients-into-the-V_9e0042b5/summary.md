---
title: "Backward-Lens-Projecting-Language-Model-Gradients-into-the-V"
source: https://aclanthology.org/2024.emnlp-main.142.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:26:54"
field: "语言模型可解释性"
keywords: ["梯度可解释性", "Logit Lens", "知识编辑", "反向传播", "低秩梯度", "imprint-and-shift"]
innovations: ["首次将 LM 梯度矩阵投影至词表空间，揭示反向传播中的 token 级信息流", "证明 n-token 提示下 MLP 梯度秩≤n，并发现最后一层梯度恒为秩1", "提出'imprint-and-shift'双阶段知识存储机制及仅需单次前向的传播近似编辑方法"]
benchmarks: ["CounterFact", "WikiText-2"]
---

# 论文速读：Backward-Lens-Projecting-Language-Model-Gradients-into-the-V

## 一句话总结
本文首次将 Transformer 语言模型的梯度矩阵投影到词表空间，揭示梯度可通过低秩分解为由前向输入 $x_i$ 和后向向量雅可比积（VJP）$\delta_i$ 张成的线性组合，并据此发现 MLP 知识存储的"印刻-偏移"（imprint-and-shift）两阶段机制。

## 研究问题与动机
- **现有解读方法仅覆盖前向过程**：Logit Lens 等解释方法将权重和隐藏状态投影到词表空间以追踪信息流，但从未涉足反向传播与梯度。
- **梯度矩阵维度高、难解释**：MLP 每层含数千神经元，直接分析整个梯度矩阵不可行；需利用其内在低秩结构压缩至少量向量。
- **知识存储机制尚不清楚**：Transformer 的 FFN/MLP 层是否以及如何在学习/编辑时编码新信息，缺乏从反向视角的细致刻画。
- **模型编辑依赖反向传播**：现有编辑方法（如 ROME、MEMIT）需完整反向传播或多步迭代；理解梯度结构有望催生更高效的单次前向编辑方案。

## 核心贡献（创新点）
1. **首次实现梯度词表投影**：提出将 LM 梯度矩阵分解为前向输入 $x_i$ 与后向 VJP $\delta_i$ 的线性组合（spanning set），再通过 Logit Lens 将每个生成向量映射为词表分布——此前所有工作均只处理前向权重/激活。
2. **梯度低秩性的形式化证明与预测**：给出引理（Lemma 4.1）严格证明 n-token 提示的梯度矩阵秩 $\leq n$，并指出最后一层 MLP 梯度恒为秩 1；这是首次将低秩性用于可解释性解释而非仅用于参数高效微调（如 LoRA）。
3. **发现"印刻-偏移"知识存储机制**：通过理论推导（Lemma 5.2）证明单次反向更新时，$FF_1$ 将前向输入 $x_i$ 加/减进神经元权重（imprint），$FF_2$ 将 VJP $\delta_i$ 减去进神经元权重（shift）——这是首个从反向视角描述的 MLP 双阶段学习机制。
4. **单次前向编辑方法（Forward Pass Shift）**：基于 shift 机制，用目标 token 的解码矩阵嵌入 $D^\top[t]$ 近似 VJP，仅通过一次前向传播完成单条知识的编辑，无需任何反向传播；在 CounterFact 上达到与 ROME 相当的 Efficacy（99.4%），同时显著降低运行时复杂度。

## 方法详解
- **梯度秩的分解**：对于含 n 个 token 的 prompt，MLP 层梯度矩阵可写为：
  $$\frac{\partial L}{\partial W} = \sum_{i=1}^{n} x_i^\top \cdot \delta_i$$
  其中 $x_i \in \mathbb{R}^d$ 是第 i 个 token 的前向中间输入，$\delta_i \in \mathbb{R}^d$ 是该层输出的 VJP（反向传播中的"隐藏状态"）。每个外积项为秩 1 矩阵，故总矩阵秩 $\leq n$。
- **FF₁ 与 FF₂ 的不同 spanning set**：对 $FF_1$（$d \times d_m$），由于 $\delta_i$ 维度为 $d_m$ 不适合直接词表投影，选择前向输入 $x_i$（维度为 d）作为 spanning set；对 $FF_2$（$d_m \times d$），选择 VJP $\delta_i$（维度为 d）作为 spanning set。
- **Logit Lens 应用于梯度向量**：沿用原版公式 $\text{LL}(v) = \text{Softmax}(ln_f(v) \cdot D)$，将每个 spanning set 向量投影为词表概率分布。对 $FF_2$ 的 VJP 关注概率最低的 token（因更新方向为减去 $\delta_i$）。
- **"印刻-偏移"机制的理论推导**：Lemma 5.2 证明，对单个 token 进行梯度更新后再重复前向，$FF_1$ 的变化使对应神经元对 $x_i$ 的响应被放大（imprint），$FF_2$ 的变化则将 $\delta_i$ 方向加到输出中（shift）。
- **Forward Pass Shift 编辑**：仅收集最后一层输入 $x_n$ 和目标 token 的嵌入 $D^\top[t]$，直接更新 $FF_2 \leftarrow FF_2 + \eta \cdot x_n^\top \cdot D^\top[t]$，模拟真实梯度的 shift 部分，避免计算 VJP。

## 实验与结果
- **数据集**：CounterFact（100 条 prompt-target 对，单编辑；1000 条用于编辑基准测试），WikiText-2 用于评估编辑后困惑度。
- **模型**：GPT2-small/medium/xl、Llama2-7B。
- **梯度秩实验**：在所有层（除最后一层）中，梯度矩阵秩几乎总是恰好等于 prompt 长度 n（>98.5% 的情况），验证 Lemma 4.1；最后一层 $FF$ 矩阵恒为秩 1。
- **Logit Lens 梯度可视化**：编辑"Obama grew up in → Paris"时，$FF_2$ 的 VJP 在多层中一致指向目标词"Paris"及其关联词；编辑"Jack Dorsey founded → IBM"时同理。
- **Prompt 各 token 的贡献分析**（GPT2-xl）：主导更新的 VJP 来自两类区域：①初始层中主语 token（如"Messi"）；②约第二四分之一层中最后一个 prompt token（"for"）。其余 token 的 $\delta_i$ 范数接近零。
- **编辑效果（CounterFact, GPT2-xl）**：

| 方法 | Efficacy ↑ | Paraphrase ↑ | Neighborhood ↑ | N-gram ↑ | Perplexity ↓ |
|------|-----------|-------------|---------------|---------|-------------|
| ROME | 99.4 | 71.9 | 10.91 | 622.78 | 137.38 |
| MEMIT | 79.4 | 40.7 | 10.98 | 627.18 | 93.6 |
| **Forward Pass Shift** | **99.4** | **41.6** | **6.02** | **622.45** | **93.66** |
| Finetune (MLP 35) | 100.0 | 46.1 | 5.59 | 618.5 | 103.42 |

Forward Pass Shift 在 Efficacy 上与 ROME 持平（99.4%），在 perplexity 和 n-gram 保持上显著优于 ROME；Paraphrase 和 Neighborhood 略逊于 ROME。

## 相关工作脉络
- **Logit Lens（nostalgebraist, 2020）**：将前向隐藏状态投影到词表；本文首次将其应用于反向梯度，本质区别是从"模型此刻在想什么"转向"模型需要修改什么"。
- **知识神经元（Geva et al., 2021; Dai et al., 2022）**：发现 MLP 神经元编码特定事实；本文进一步刻画这些知识在反向传播中如何被写入权重的动态机制。
- **ROME（Meng et al., 2022）**与**MEMIT（Meng et al., 2023）**：通过 edit location 定位并更新 $FF_2$；本文从理论上解释了这些方法有效的内在原因（shift 机制），并提出无需反向传播的单步替代。
- **MEND（Mitchell et al., 2021）**：利用元学习近似梯度；本文方法在编辑效果相近的同时不依赖元训练，且复杂度大幅降低。
- **LoRA（Hu et al., 2022）**：利用梯度低秩性做高效微调；本文首次在可解释性语境下形式化证明该低秩性，并基于此驱动向量级分析而非直接优化。
- **Saliency Map 与梯度聚类（Simonyan et al., 2014; Ilharco et al., 2022）**：关注梯度对预测的影响或与训练数据的关系；本文核心区别是将梯度向量本身投影为可读 token 分布。

## 局限性与未来方向
- **Logit Lens 在浅层可解释性不足**：早期层（<10）的 VJP 投影质量下降，可能因 LL 在浅层存在系统性 gap。
- **未覆盖注意力层**：本文聚焦 MLP 层，注意力层可能同样存储知识，未来需扩展至 attention gradients。
- **单 token 嵌入近似 VJP**：Forward Pass Shift 用目标 token 嵌入 $D^\top[t]$ 代替真实 VJP $\delta_n$，在 paraphrase 泛化和 neighborhood 特异性上有一定损失。
- **未探索多 prompt 编辑/全量微调**：全文基于单次反向更新，多 prompt/多 step 场景下的低秩性假设失效。
- **忽略优化器缩放（如 Adam）**：实际 fine-tuning 中优化器会改变梯度权重，但本文未讨论该影响。
- **未来方向**：利用 SVD 分解完整梯度提取主导子空间；探索不同编辑任务间共享的子空间结构（superposition 视角）；引入 Normalized Logit Lens 或 sparse autoencoder 提升浅层可解释性。

## 研究启发与可借鉴点
1. **梯度分解视角**：将梯度矩阵拆解为前向输入与 VJP 的外积之和，以 spanning set 替代逐神经元分析，可将分析复杂度从 $O(d \times d_m)$ 降至 $O(n)$（n 为 prompt 长度），适用于任何基于 Transformer 的可解释性研究。
2. **"imprint-and-shift" 机制可用于指导编辑策略**：若需保留某侧知识而修改另一侧，可针对性地编辑 $FF_1$ 或 $FF_2$，或通过控制 spanning set 中不同 token 的 $\delta_i$ 范数来精细调节修改方向。
3. **Forward Pass Shift 的低复杂度优势**：单次前向编辑方法为资源受限场景（边缘部署、在线编辑）提供了实用基线；其将目标嵌入直接注入权重的思想可推广至其他模态或架构。
4. **Normalized Logit Lens（Appendix H）**：对低范数 VJP 向量进行归一化后再做 LL 投影，可有效消除噪声，使目标 token 信号更清晰；该方法可直接复用于其他向量投影研究。
5. **梯度 VJP 跨层余弦相似度分析**：Figure 23 展示的跨层 VJP 对齐模式（同层块内相似度 >0.7）可作为诊断工具，用于评估不同编辑方法的梯度传播一致性。

## 关键术语表
- **Vector-Jacobian Product (VJP)**：反向传播中误差从后续层回传至当前层输出的加权向量，等价于反向过程中的"隐藏状态"。
- **Logit Lens (LL)**：将模型内部任意 d 维向量通过解码矩阵 D 投影为词表概率分布的可解释方法。
- **Imprint-and-Shift 机制**：MLP 知识存储的双阶段过程——$FF_1$ 印刻前向输入 $x_i$，$FF_2$ 偏移目标 token 嵌入 $\delta_i$。
- **Spanning Set（生成集）**：梯度矩阵可表示为其向量集合的线性组合；本文指 $x_i$（对 $FF_1$）或 $\delta_i$（对 $FF_2$）。
- **Forward Pass Shift**：本文提出的单次前向编辑方法，用目标 token 嵌入近似 VJP，直接更新 $FF_2$。
- **CounterFact**：常用于知识编辑评测的事实性对抗数据集，包含需被编辑的真-假命题对。
- **Rank-n 梯度矩阵**：n-token prompt 下 MLP 梯度的秩最多为 n，源于 n 个秩-1 外积之和的结构。

## 可复现要素
- **数据集**：CounterFact（公开，https://github.com/kmeng01/rome/tree/main/counterfact）；WikiText-2（公开）。
- **代码**：已开源，https://github.com/shacharKZ/BackwardLens。
- **模型**：GPT2-small/medium/xl、Llama2-7B（均通过 HuggingFace transformers 加载）。
- **关键超参**：编辑层数（Layer 35 表现最优）；学习率 η=0.24；Single backward pass，SGD 优化器，无 batching，无 Adam 缩放。
- **实验设置**：100 条样本用于分析实验，1000 条样本用于 CounterFact 基准测试；每样本单次反向传播编辑；Efficacy/Paraphrase/Neighborhood/N-gram/Perplexity 五项指标。
