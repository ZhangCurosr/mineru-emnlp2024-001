---
title: "TinyChart-Efficient-Chart-Understanding-with-Program-of-Thou"
source: https://aclanthology.org/2024.emnlp-main.112.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:33:28"
field: "多模态图表理解"
keywords: ["图表理解", "多模态大模型", "Program-of-Thoughts", "Token Merging", "视觉语言模型", "数值推理"]
innovations: ["提出仅3B参数的TinyChart模型，通过PoT学习和Visual Token Merging在多个图表基准上达到SOTA", "将Program-of-Thoughts学习策略引入图表理解领域，训练模型生成Python程序求解数值问题", "在每层ViT中引入无参数Visual Token Merging模块，实现高分辨率图表的高效编码"]
benchmarks: ["ChartQA", "Chart-to-Text (Pew)", "Chart-to-Table", "OpenCQA", "ChartX-Cognition"]
---

# 论文速读：TinyChart: Efficient Chart Understanding with Program-of-Thoughts Learning and Visual Token Merging

## 一句话总结
TinyChart 是一个仅需 **3B 参数**的多模态图表理解模型，通过 **Program-of-Thoughts (PoT) 学习策略**（训练模型生成 Python 程序求解数值问题）和 **Visual Token Merging**（在每层 ViT 内合并相似视觉 token 以缩短序列长度），在 ChartQA、Chart-to-Text、Chart-to-Table、OpenCQA 和 ChartX 等多个基准上达到 SOTA，并以更小的模型规模实现更高效的推理吞吐。

---

## 研究问题与动机

1. **模型规模过大限制部署**：现有图表理解模型参数规模庞大，如 ChartLlama 拥有 13B 参数，难以在显存小于 26GB 的单卡 GPU 上训练或部署。
2. **数值计算容易出错**：现有 MLLM 在面对需要数值计算的问题时极易产生错误，这类问题无法通过直接生成短答案解决，需要推理步骤。
3. **高分辨率图像编码效率低**：图表中的关键信息（如 OCR 文字）在低分辨率下难以识别，但标准 ViT 处理高分辨率图像时会产生极长的视觉特征序列，带来巨大的计算开销。

---

## 核心贡献（创新点）

1. **TinyChart 高效图表理解模型**：提出一个仅 3B 参数的多模态图表理解模型，在多个基准上超越 13B 参数模型并达到 SOTA 性能；与已有工作的区别在于证明了小模型通过高效视觉编码和程序式学习也能取得顶尖效果，而非单纯依赖大参数规模。

2. **Program-of-Thoughts (PoT) 学习策略**：训练模型逐步生成 Python 程序来计算数值答案，而非直接生成短答案；与已有工作（如 ChartAst 使用受限模板生成的可执行命令）的本质区别在于使用更通用的 Python 语言，且结合 GPT 生成的多样化数据，覆盖范围更广。

3. **Visual Token Merging 高效视觉编码**：在每层 Vision Transformer 中插入无参数的 Token 合并模块，通过二分图匹配合并最相似的 token 对，并配合 Proportional Attention 补偿合并带来的注意力权重损失；与已有工作（如固定分辨率处理或使用额外高分辨率编码器）的区别在于无需额外编码器，直接在标准 ViT 内部实现序列可控压缩。

4. **ChartQA-PoT 数据集构建**：基于 ChartQA 训练集构建包含 140,584 个 (问题, PoT 答案) 对的数据集，其中 119,281 个来自模板生成、21,303 个来自 GPT-3.5-turbo 生成；与已有数据集的区别在于专门为支持程序式学习而生成的结构化代码-注释对。

---

## 方法详解

### 模型架构
TinyChart 由三部分组成：**Vision Transformer Encoder** → **Vision-Language Connector**（单层 MLP + GeLU）→ **Large Language Model**（Phi-2，3B 参数）。

### Visual Token Merging（视觉 Token 合并）
- 核心观察：图表图像中大量颜色块和空白区域的 patch 在视觉上高度相似。
- 在每个 ViT 层内，计算 self-attention 的 Keys 之间的余弦距离，通过**二分图匹配**找到 top-r 最相似的 token 对，对每对 token 进行**平均池化**合并，使视觉特征序列每层缩短 r 个 token。
- 引入 **Proportional Attention** 公式补偿合并导致的 token 权重损失：
  $$\text{Attention} = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} + \log s\right)V$$
  其中 s 表示每个 token 代表的原始 patch 数量，通过加 $\log s$ 使合并后的 token 在注意力计算中被"复制" s 次。

### Program-of-Thoughts (PoT) 学习
- 模型被训练生成包含注释的 Python 代码，代码执行结果即为答案。
- 答案变量统一命名为 "Answer"，代码包含多步推理过程（如提取数据 → 计算 → 存储结果）。
- 推理时可设置五种模式：
  1. **Direct**：直接生成短答案。
  2. **PoT**：生成 Python 代码并通过解释器求解。
  3. **Combine**：对含计算关键词的问题使用 PoT，其余使用 Direct。
  4. **Auto**：训练模型自主判断使用哪种方式。
  5. **Oracle**：两者均生成并选取正确答案。

### 训练数据
- 主训练集共 **1.36M** 样本，涵盖 ChartQA（QA）、Pew/Statista（Chart-to-Text）、ChartQA/PlotQA（Chart-to-Table）、OpenCQA、ChartLlama 指令数据等。
- ChartQA-PoT 训练子集：140,584 个 (question, PoT answer) 对。

### 损失函数
采用标准语言建模损失，仅对响应部分计算：
$$\mathcal{L} = \frac{1}{T} \sum_{i=1}^{T} \text{LLM}(R_i | V, L, R_{<i})$$

---

## 实验与结果

### 评测基准
- **ChartQA**：relaxed accuracy（允许 5% 数值误差），含 Augmented 和 Human 两类问题。
- **Chart-to-Text**：Pew 数据集，BLEU4 指标。
- **Chart-to-Table**：ChartQA 数据表标注，RMS_F1 指标。
- **OpenCQA**：开放式问答，BLEU4 指标。
- **ChartX-Cognition**：涵盖 QA、Summarization、Description、Redrawing，使用 GPT-Acc / GPT-Score。

### 主要结果（Table 1）

| 模型 | 参数量 | 分辨率 | ChartQA Avg. | ChartQA-Hum. | Chart-to-Text (BLEU4) | Chart-to-Table (RMS_F1) | OpenCQA (BLEU4) | 吞吐量 (it/s) |
|------|--------|--------|-------------|--------------|-----------------------|------------------------|-----------------|--------------|
| GPT-4V | - | - | 78.50 | - | - | - | - | - |
| Gemini-Ultra | - | - | 80.80 | - | - | - | - | - |
| Qwen-VL-Max | - | - | 79.80 | - | - | - | - | - |
| ChartAst | 13B | 448×448 | 79.90 | 65.90 | 15.50 | 91.60 | 15.50 | 1.47 |
| **TinyChart@768** | **3B** | **768×768** | **83.60** | **73.34** | **17.18** | **93.78** | **20.39** | **3.14** |

- TinyChart@768 在 ChartQA 平均准确率上达 **83.60%**，超越 GPT-4V（78.50%）、Gemini-Ultra（80.80%）和所有开源模型。
- 在更具挑战性的 ChartQA-Human（人类标注的数值计算问题）上达 **73.34%**，较 ChartAst 提升 **7.44%**。
- 推理吞吐量为 **3.14 it/s**，显著高于 13B 模型（ChartAst 1.47 it/s、ChartLlama 1.94 it/s）。

### ChartX 泛化评测（Table 4，未额外微调）
- TinyChart@768 在 QA 任务上达 **33.35 GPT-Acc**，超越 GPT-4V（33.04），在所有开源图表 MLLM 中最佳。

### PoT 设置对比（Table 2）
- TinyChart@768 在 Direct 设置下 76.36%，PoT 设置下 80.84%，Combine 设置下 83.60%，Oracle 设置下 89.12%。
- 数值计算问题上 PoT 较 Direct 提升显著（78.98 vs. 56.64）。

### 消融实验结论（Table 5）
- **分辨率提升**：从 384→512 带来全基准显著提升，但吞吐量从 3.73 降至 2.38 it/s。
- **Token Merging**：r=20 时吞吐量恢复至 3.65 it/s，同时保持高分辨率收益。
- **768 分辨率训练**：无 Token Merging 时序列长度 2,916，在 32GB V100 上 OOM；启用 r=84 后序列压缩至 732，正常训练且吞吐量达 3.14 it/s。
- **GPT-based PoT**：加入 GPT 生成数据后，PoT 答案（76.88）超过 Direct 答案（72.44），Combine 达 79.48。

---

## 相关工作脉络

1. **ChartLlama (Han et al., 2023)**：13B 开源图表 MLLM，仅支持 336 分辨率，无法处理高分辨率图表中的细粒度 OCR 信息；TinyChart 以 3B 规模通过高分辨率 + Token 合并实现更好效果。

2. **ChartAst (Meng et al., 2024)**：13B 开源 SOTA，使用受限模板生成可执行 JSON 代码；TinyChart 使用更通用的 Python PoT 学习，并结合 GPT 生成数据提升覆盖度。

3. **DePlot + Codex (Liu et al., 2023)**：1.3B + 175B 双模型方案，依赖外部大模型进行 plot-to-table 转换；TinyChart 仅 3B 单模型内完成理解与计算。

4. **ChartInstruct (Masry et al., 2024)**：7B 指令微调图表模型，但未专门针对数值计算优化；TinyChart 通过 PoT 学习专门强化数值推理能力。

5. **Token Merging (Bolya et al., 2023)**：通用 ViT 加速方法，本文将其引入多模态大模型内部，适配多模态场景并配合 Proportional Attention 修正注意力偏差。

6. **Program of Thoughts (Chen et al., 2023)**：在数学推理中提出 PoT，本文首次将其应用于**图表理解**的数值计算场景，并构建专用数据集。

---

## 局限性与未来方向

1. **幻觉问题**：在图表摘要生成中仍存在 MLLM 常见的幻觉现象，可能生成与图表内容不符的信息。
2. **无 OCR 文字的数据点估计困难**：即使提高分辨率，模型在缺少周围文字标注的情况下仍难以准确估计数据点的数值，反映了对大跨度空间关系理解能力的不足。
3. **未来方向**：（1）探索缓解图表理解中幻觉的方法；（2）提升模型对无 OCR 辅助数据的数值估算精度；（3）增强对不同图表类型（如 3D 柱状图）的泛化能力。

---

## 研究启发与可借鉴点

1. **PoT 学习迁移**：将程序式思维引入多模态理解任务（而不仅限于纯文本推理）是一条有效路径；可借鉴其"代码生成 + 执行验证"的训练范式应用于其他需要数值推理的视觉任务。

2. **Token Merging 的多模态适配**：在 Vision Transformer 内部每层插入 Token 合并模块是高效处理高分辨率输入的有效手段；配合 Proportional Attention 修正权重损失的设计可复用至其他高分辨率视觉-语言任务。

3. **双源数据集构建策略**：模板生成（保证覆盖度和正确性）+ 大模型生成（提升多样性）的组合方式是构建高质量训练数据的可复用范式。

4. **多设置评测框架**：Direct / PoT / Combine / Auto / Oracle 五模式评测体系可细致揭示模型在不同推理策略下的能力边界，值得在其他需要多步推理的视觉任务中采用。

5. **小模型高效部署路径**：证明了 3B 参数模型通过结构化学习策略和高效编码可在垂直领域达到甚至超越 13B+ 闭源模型，为资源受限场景下的多模态部署提供了可行路线。

---

## 关键术语表

**Program-of-Thoughts (PoT)**：一种训练大模型生成逐步推导代码（而非直接输出答案）以解决数值计算问题的学习策略，代码经解释器执行后得到最终答案。

**Visual Token Merging**：在 Vision Transformer 每层中通过将最相似的视觉 token 对合并来缩短特征序列长度的无参数加速模块。

**Proportional Attention**：在 Token 合并后，通过在注意力计算中引入 $\log s$ 项（s 为被合并 token 对应的原始 patch 数量）来补偿合并带来的注意力权重偏差的修正机制。

**ChartQA-PoT**：基于 ChartQA 训练集构建的包含 140,584 个 (问题, Python 程序) 对的专用数据集，用于支持 PoT 学习。

**Relaxed Accuracy**：允许答案与标准答案存在 5% 以内数值误差的宽松准确率评估指标。

**Combine Setting**：根据问题是否包含计算关键词，动态选择 Direct 或 PoT 生成方式的答案组合评测设置。

**Oracle Setting**：同时生成 Direct 和 PoT 两种答案，并选取正确的那个作为最终结果的理想化评测设置。

---

## 可复现要素

- **数据集**：ChartQA（公开）、PlotQA（CC-BY-4.0）、DVQA（Tencent）、OpenCQA（GPL-3.0）、Pew/Statista（GPL-3.0）、ChartX（公开）；ChartQA-PoT 数据集论文未声明开源。
- **代码/权重**：论文未明确声明代码与模型权重是否开源（以论文原文为准）。
- **关键超参**：模型基于 TinyLlava 初始化（SigLIP 视觉编码器 + Phi-2 LLM）；分辨率支持 384/512/768；Token Merging 每层合并数 r=12/15/20/84；训练 3 epochs，batch size=512，learning rate=1e-4，warmup 3%；训练设备 32×Tesla V100 32GB，耗时 3 天。

---
