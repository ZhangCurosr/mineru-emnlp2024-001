---
title: "Autoregressive-Pre-Training-on-Pixels-and-Texts"
source: https://aclanthology.org/2024.emnlp-main.182.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:27:49"
field: "多模态语言模型"
keywords: ["pixel-based language modeling", "autoregressive pre-training", "multimodal pre-training", "cross-lingual understanding", "RGB text rendering", "vocabulary bottleneck"]
innovations: ["提出 PixelGPT：首次在自回归 decoder-only 架构上对 24-bit RGB 文本图像进行纯视觉预训练", "提出 DualGPT：通过图像-文本配对数据实现视觉与文本双模态联合预训练，缓解模态冲突", "系统验证 RGB 渲染在保留颜色语义（如 emoji）及跨字体迁移鲁棒性方面的优势"]
benchmarks: ["GLUE", "XNLI", "HatemojiBuild"]
---

# 论文速读：Autoregressive Pre-Training on Pixels and Texts

## 一句话总结
本文提出 PixelGPT 和 DualGPT 模型，在自回归 Transformer 解码器上探索对渲染的 RGB 文本图像（而非灰度/二值图）进行纯视觉预训练，并进一步融合文本与视觉双模态数据，有效突破了传统 tokenization 带来的词汇表瓶颈，在多语言语言理解与跨语言泛化任务上展现出与 SOTA 双向像素模型相当甚至更优的性能。

## 研究问题与动机
1. **词汇表瓶颈**：传统语言模型依赖 WordPiece/BPE 等子词分词，在多语言和视觉复杂文档（PDF）场景下受限；像素级建模可绕过分词，直接从图像像素学习。
2. **自回归像素模型的空白**：现有像素语言模型（如 PIXEL、CLIPPO）主要为双向编码器架构，自回归 decoder-only 架构在像素预训练领域尚处于起步阶段。
3. **视觉信息损失**：现有方法多使用 8-bit 灰度或 2-bit 二值图像，丢失了彩色信息（如 emoji、高亮文本）；使用全彩 RGB（24-bit）渲染能保留更丰富的视觉细节。
4. **多模态协同潜力**：视觉与文本双模态是否在预训练阶段产生协同效应，对语言理解与跨语言泛化的增益尚待系统探究。

## 核心贡献（创新点）
1. **提出 PixelGPT**：首次系统验证在解码器-only Transformer 上直接对 24-bit RGB 文本图像进行自回归预训练的可行性，在 GLUE 上达到与双向 PIXEL 模型相当的性能。
2. **引入 DualGPT 双模态预训练策略**：将视觉图像与纯文本数据联合预训练，并通过图像-文本配对数据缓解单模态训练时的模态冲突，显著提升了多语言理解能力。
3. **证明 RGB 渲染相比灰度/二值渲染的多方面优势**：在 HatemojiBuild 等需要颜色信息的数据集上，RGB 渲染显著优于灰度；同时在 XNLI 上对不同渲染模式的鲁棒性进行了系统分析。
4. **开源代码、数据与模型权重**：发布 Fine-tuning 数据集和模型 checkpoint，促进该新兴领域的后续研究。

## 方法详解
- **文本渲染为图像**：使用文本渲染器将文本转换为 $x \in \mathbb{R}^{H \times W \times C}$，其中 $H=16$ 像素，$W=16{,}384$ 像素，24-bit RGB 三通道（$C=3$），形成 $1024$ 个 $16 \times 16$ 的 patch；使用 $16 \times 16$ 黑色 patch 作为 EOS 标记，不参与注意力与损失计算。
- **输入表示**：图像输入经 Patch Embedding（类似 ViT）线性投影到 $D$ 维空间后按文本顺序排列；文本输入使用 Llama 2 的 BPE tokenizer，词汇表大小 32k。
- **预训练目标**：
  - **Next Patch Prediction（视觉）**：对视觉 patch 序列采用因果分解 $p(x_p^1, \cdots, x_p^N) = \prod_{t=1}^N p(x_p^t | x_p^1, \cdots, x_p^{t-1})$，使用归一化 MSE Loss，排除 EOS patch。
  - **Next Token Prediction（文本）**：标准交叉熵 loss，teacher-forcing 方式预测下一个 token。
- **模型变体**：
  - **TextGPT**：仅文本预训练。
  - **PixelGPT**：仅 RGB 图像预训练（MSE loss）。
  - **MonoGPT**：文本与图像独立流并行预训练（无配对）。
  - **DualGPT**：文本、图像及图像-文本配对数据联合预训练（配对数据中图像序列拼接于文本序列之前）。
- **架构**：基于 Llama 2 Decoder，24 层，hidden size=1024，intermediate size=2816，16 个 attention heads（GQA，8 KV heads），SwiGLU 激活，RoPE 位置编码，RMSNorm。

## 实验与结果
- **数据集**：GLUE（9 项 NLU 任务）、XNLI（15 种语言跨语言推理）、HatemojiBuild（emoji 仇恨检测）。
- **预训练数据**：视觉端使用 peS2o、English Wikipedia、C4 渲染图像；文本端使用 Dolma（3T tokens）、Common Crawl、C4、peS2o、The Stack v1、Project Gutenberg、Wikipedia。
- **GLUE 结果**（Table 2）：PixelGPT（317M 参数）在 GLUE 平均分为 **74.2**，超过 PIXEL（74.1）和 PIXAR（74.0），在 STS-B（+5.4）、MRPC（+13.1）、RTE（+11.9）、WNLI（+5.4）上显著优于 GPT-2；在 QQP（+1.5）、RTE（+3.4）、WNLI（+5.4）上超过双向 PIXEL 模型。
- **XNLI Translate-Train-All**（Table 3）：PixelGPT 平均分 **63.2**，超过 PIXEL（62.8）与 BERT（63.9 相近）；在泰语（+11.3）和中文（+4.3）上大幅超越 BERT。
- **多模态消融**（Table 4/5）：DualGPT（text+pixel+pair）在 GLUE 上得 **76.9**，优于 TextGPT（76.3）和 MonoGPT（75.4）；XNLI 上 DualGPT pixel 端也明显优于 MonoGPT。
- **Scaling 趋势**（Figure 3/4）：在约 200B tokens/patches 时 PixelGPT 超越 PIXEL，且仍呈上升趋势；早期加入配对双模态数据对 pixel-based 模型增益尤其显著。
- **大 Batch 提升**（Figure 5）：像素模态 fine-tune 使用更大 batch size（如 512）显著改善训练稳定性与性能。
- **字体迁移鲁棒性**（Figure 6）：使用与预训练不同的字体（含符号化字体 JournalDingbats1）fine-tune，性能基本保持稳定。
- **RGB vs 灰度**（Table 6）：在 HatemojiBuild 上 RGB 渲染（61.4）显著优于灰度（58.7，+2.7）。

## 相关工作脉络
1. **PIXEL（Rust et al., 2023）**：基于 MAE 风格的双向编码器像素语言模型，处理灰度图、MSE loss；本文 PixelGPT 可视为其自回归 decoder 版本，但采用 RGB 渲染。
2. **CLIPPO（Tschannen et al., 2023）**：统一 encoder 处理图像与文本，对比学习；本文走自回归生成路线，关注 next patch/token prediction。
3. **PIXAR（Tai et al., 2024）**：自回归像素建模，但使用二值图像与 BCE loss；本文使用连续 RGB 空间与 MSE loss，信息更丰富。
4. **DONUT（Kim et al., 2022）**：OCR-free 视觉文档理解模型；本文聚焦于纯渲染文本的自回归预训练而非文档结构化抽取。
5. **Llama 2（Touvron et al., 2023）**：本文以 Llama 2 Decoder 为基座架构，验证像素预训练的可行性。
6. **GPT-2 / BERT**：传统文本-based 自回归与双向基线，本文证明纯视觉自回归模型可在部分任务上与之竞争。

## 局限性与未来方向
1. **模型规模受限**：当前仅 24 层、~317M 参数，未探索 7B/13B/70B+ 的大规模扩展。
2. **训练计算量有限**：预训练仅在 100-200B tokens/patches，未达 1T+ 量级，scaling 潜力未完全释放。
3. **生成任务存在障碍**：像素模型的输出是图像 patch，需额外 OCR 后处理才能得到文本，增加复杂度与误差来源。
4. **初步探索性质**：作为像素自回归预训练的初步研究，需在更多任务与设置下验证鲁棒性。

## 研究启发与可借鉴点
1. **RGB 渲染的可视化增益**：对包含颜色语义的任务（如 emoji 理解、高亮文本），保留 24-bit RGB 比灰度/二值更有价值，可迁移到任何视觉文本理解场景。
2. **大 batch size 对像素模态稳定性的关键作用**：fine-tune 像素模型时需使用更大 batch（如 256/512）以缓解训练方差，这一经验可推广至其他新型视觉模态的自回归训练。
3. **配对双模态数据的早期介入效益**：DualGPT 在 10B tokens 时就展现出对 MonoGPT 的显著优势，表明配对数据在训练早期就能引导像素模态更好地对齐文本语义。
4. **跨字体鲁棒性验证方法**：使用差异极大的符号化字体（JournalDingbats1）进行迁移测试，可作为评估像素预训练鲁棒性的有效手段。
5. **突破分词瓶颈的思路**：在多语言/低资源语言场景下，可考虑将文本渲染为图像进行预训练，绕过 BPE/WordPiece 的词汇表限制。

## 关键术语表
- **PixelGPT**：仅基于 24-bit RGB 文本图像进行自回归预训练的 decoder-only Transformer 模型。
- **DualGPT**：同时利用纯文本、渲染图像及图像-文本配对数据进行多模态预训练的模型变体。
- **Next Patch Prediction**：像素预训练目标，对视觉 patch 序列逐个预测下一个 patch，使用归一化 MSE loss。
- **词汇表瓶颈（Vocabulary Bottleneck）**：传统子词分词在多语言/稀有脚本场景中无法覆盖所有字符的固有限制。
- **Translate-Train-All**：XNLI 跨语言评估设置，模型在英语与机器翻译的 14 种语言数据上联合 fine-tune。
- **MonoGPT**：文本与图像独立（非配对）双流预训练的模型变体。
- **GQA（Grouped Query Attention）**： grouped query attention，减少 KV head 数量以加速推理的注意力优化技术。
- **HatemojiBuild**：基于 adversarial perturbations 构建的 emoji 仇恨言论检测 benchmark。

## 可复现要素
- **数据集**：GLUE、XNLI、HatemojiBuild 为公开基准；预训练数据 peS2o、English Wikipedia、C4、Common Crawl、Dolma、The Stack v1、Project Gutenberg、Wikipedia 均有公开版本或部分开源。
- **代码/权重**：代码、数据与模型 checkpoint 已开源，GitHub：https://github.com/ernie-research/pixelgpt（论文声明）。
- **关键超参**：patch size=16×16，max seq length=1024，hidden size=1024，24 层，learning rate 5e-4（预训练）/1e-5~1e-4（fine-tune），batch size=4M~10M tokens/patches，optimizer=AdamW，warmup=200 steps，mixed precision=bfloat16，RMSNorm eps=1e-5，RoPE theta=10000。
