---
title: "By-My-Eyes-Grounding-Multimodal-Large-Language-Models-with-S"
source: https://aclanthology.org/2024.emnlp-main.133.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:28:02"
field: "多模态大语言模型应用"
keywords: ["视觉提示", "多模态大语言模型", "传感器数据", "可视化生成", "少样本学习", "MaaS"]
innovations: ["将传感器数据以可视化图像形式输入MLLM的视觉提示方法，平均提升10%准确率并降低15.8倍token成本", "设计两阶段自动化可视化生成器（工具过滤+图像评估选择），无需领域先验知识"]
benchmarks: ["HHAR", "UTD-MHAD", "PTB-XL", "WESAD", "Swim"]
---

# 论文速读：By-My-Eyes-Grounding-Multimodal-Large-Language-Models-with-S

## 一句话总结
本文提出将传感器数据可视化后作为图像输入多模态大语言模型（MLLM）的**视觉提示（Visual Prompt）**方法，并通过自动化可视化生成器在9个传感任务上验证，平均比文本提示提升10%准确率，同时token成本降低15.8倍。

## 研究问题与动机
1. **文本提示处理长序列传感器数据性能骤降**：GPT-4o在数值均值预测和波形分类任务上，当序列长度超过100个元素时性能显著下降，500元素时接近随机分类器水平。
2. **文本表示传感器数据token成本极高**：以100Hz采样、采集1分钟的加速度计数据为例，1-shot文本提示约需90K token，按GPT-4o API计每小时成本约$450，无法实用。
3. **特征工程依赖领域知识、泛化性差**：已有方法需提取任务特定特征（如ECG的R-R间期），对非专家用户不友好。
4. **MLLM视觉能力未被充分利用于传感数据**：现有工作主要将传感器数据转为文本，忽略了MLLM日益增强的图像理解能力。

## 核心贡献（创新点）
1. **提出视觉提示框架**：将传感器数值序列转换为可视化图像（如波形图、频谱图），结合任务描述文本输入MLLM；本质区别在于用"视觉压缩"替代"文本展开"，使长序列信息以固定token成本呈现。
2. **设计自动化可视化生成器**：包含工具过滤（filtering）和可视化选择（selection）两阶段，MLLM可自主从公共库（Matplotlib、Scipy、Neurokit2）中选出最适合当前任务和数据特征的可视化方法，无需人工先验知识。
3. **系统性评估9个跨模态传感任务**：覆盖加速度计（HHAR/UTD-MHAD/Swim）、ECG（PTB-XL ×4）、EMG（Gesture）、呼吸（WESAD），证明方法的广泛适用性，并在同token预算下展示视觉提示可容纳更多示例的优势。
4. **揭示CoT与文本摘要对传感数据效果有限**：实证表明zero-shot CoT对文本提示有负面影响，文本摘要也显著劣于视觉提示，凸显视觉表征的独特价值。

## 方法详解
### 视觉提示设计
- 将传感器数值序列通过可视化方法生成图像（如raw waveform、spectrogram、ECG individual heartbeats等）。
- Few-shot示例图像上方附带标签（如`{{Label of example X}}`），目标数据标题为"target data"。
- 附带文本指令说明数据收集方式和任务目标，引导MLLM有效利用图像。

### 可视化生成器（两阶段）
**阶段一：可视化工具过滤（Filtering）**
- 向MLLM提供公共库中16种可视化工具的名称、描述及任务/数据说明，要求MLLM以JSON格式输出适合该任务的工具列表，利用in-context learning提供的few-shot演示提升质量。

**阶段二：可视化选择（Selection）**
- 用所有过滤后的工具分别生成可视化图像，再让MLLM观察各图后选出最优一个。
- 关键提示设计：明确要求MLLM避免依赖先验知识，聚焦于图像本身内容，以减少偏差。
- 最终将选中的可视化图像与任务指令组合为视觉提示送入MLLM求解。

### Token成本计算
- GPT-4o图像token按 $85 + 170 \times N$ 计算（N为覆盖图像的$512 \times 512$像素块数），所有可视化图为单张512×512图像，token数与序列长度无关。
- 相同预算下视觉提示可放入更多示例（如500 token预算：文本≈2000字符/2KB，视觉≈2张图/1.57MB，信息密度高785倍）。

## 实验与结果
**数据集**：HHAR、UTD-MHAD、Swim（加速度计）；PTB-XL（ECG，4种心律失常检测）；Gesture（EMG）；WESAD（呼吸，压力检测）。每类30个测试样本（UTD-MHAD为10个）。

**基线**：Text-only prompt（文本提示）、Task-specific model（1D CNN / XResNet-101全监督上限）。

**核心结果（1-shot，GPT-4o）**：

| 数据集 | Text-only | Visual Prompt（ours） | 提升 |
|--------|-----------|----------------------|------|
| HHAR | 0.66 | 0.67 | +0.01 |
| UTD-MHAD | 0.10 | 0.43 | **+0.33** |
| Swim | 0.51 | 0.73 | +0.22 |
| PTB-XL (CD) | 0.73 | 0.80 | +0.07 |
| PTB-XL (MI) | 0.62 | 0.68 | +0.06 |
| PTB-XL (HYP) | 0.47 | 0.55 | +0.08 |
| PTB-XL (STTC) | 0.53 | 0.57 | +0.04 |
| Gesture | 0.27 | 0.30 | +0.03 |
| WESAD | 0.48 | 0.61 | +0.13 |
| **平均** | **0.49** | **0.59** | **+10%** |

**Token成本**：视觉提示平均token数仅为文本提示的1/15.8（HHAR：2020 vs 52910，降26.2×；WESAD：1211 vs 60253，降49.8×）。

**消融**：可视化生成器 vs Fixed waveform（ECG任务↓20%）vs Desc.-based（UTD-MHAD降至0.05），证明自适应选择至关重要。

**Multi-shot**：同token预算下，5-shot视觉提示在PTB-XL MI/HYP任务上显著优于1-shot文本提示；但更多shots不总带来提升（"lost in the middle"现象）。

**小模型**：LLaVa-7B在视觉/文本提示下均接近随机（~0.33），凸显方法对强MLLM的依赖。

## 相关工作脉络
1. **PromptCast / LLMTime**：将时间序列转为文本prompt做预测；本文关注更广泛的传感分类任务，且揭示长文本序列的固有缺陷。
2. **Penetrative AI / Health-LLM**：直接将原始传感器数值文本化输入LLM；本文证明此方式在长序列场景下性能与成本双劣。
3. **Zhang et al. (2023) 编码器+embedding方案**：需额外预训练专用编码器；本文无需微调或额外训练，零样本即可迁移。
4. **Toolformer / HuggingGPT / Data Interpreter**：将LLM/MLLM作为工具调度器；本文反其道而行，让MLLM作为主求解器，工具仅辅助生成可视化。
5. **医疗图像诊断中的MLLM应用**（Wu et al., 2023）：证明MLLM可读医学图像；本文将该思路推广至通用传感器数据可视化。

## 局限性与未来方向
1. **数值精确检索任务不适配**：视觉表示省略了具体数值，对需精确数字的任务（如均值计算）不如文本提示；需探索图文信息的最优配比。
2. **高密度多通道数据可视化困难**：256通道EEG等数据在单图中无法清晰呈现，分subplots又导致性能下降12%，待改进。
3. **CoT提示效果不稳定**：zero-shot CoT对文本提示有负面影响，对视觉提示也无明显增益，需设计传感数据专用的推理链。
4. **小规模MLLM不适用**：LLaVa-7B表现接近随机，需针对小模型优化视觉表征或增强其多图理解能力。
5. **样本量受限**：因文本prompt成本高昂，每类仅30样本，扩大规模有待资源支持。

## 研究启发与可借鉴点
1. **"视觉压缩替代文本展开"**：对任意长序列/数值数据，优先尝试可视化降维+MLLM理解，而非直接文本化，可同步提升性能与成本效益。
2. **自动化工具选择的两阶段框架**（filter→assess）：可迁移至其他需要"选择最优数据呈现方式"的场景（如金融时序、日志数据）。
3. **同token预算下比较多shot能力**：视觉prompt因固定token开销，可在相同预算下塞入更多示例，这一设计思路值得在通用VLM应用中复用。
4. **揭示CoT在感知型任务上的局限性**：提示后续研究在涉及模式识别/信号理解的NLP任务中，需谨慎使用CoT，避免干扰模型的直观判断。
5. **可拓展方向**：将本框架与自有团队的方向（如医疗信号分析、行为识别）结合，验证跨领域泛化性。

## 关键术语表
**Visual Prompt**：将数据转化为图像形式并嵌入提示中供MLLM读取的提示策略，区别于纯文本prompt。
**MLLM（Multimodal Large Language Model）**：支持多模态输入（文本+图像/音频）的大语言模型，如GPT-4o、Gemini。
**Visualization Generator**：自动为给定传感数据和任务从公共库中选择并生成最优可视化图表的模块。
**In-context Learning**：通过在prompt中提供few-shot示例，让MLLM在不更新参数情况下适应新任务的能力。
**PTB-XL**：大规模公开12导联心电图数据集（10秒/记录，100Hz），用于心律失常检测基准评测。
**Token Cost**：MLLM API调用按输入/输出token计费；图像token按$512\times512$像素块计数，与数据长度无关。
**UTD-MHAD**：多模态人类活动识别数据集，含加速度计与深度相机数据，支持21类细粒度动作分类。
**WESAD**：可穿戴压力与情感检测数据集，含呼吸、皮电、心率等多模态生理信号。

## 可复现要素
- **数据集**：HHAR、UTD-MHAD、Swim、PTB-XL、WESAD均为公开数据集；手势EMG数据集（Ozdemir et al., 2022）公开。
- **代码**：开源，地址 https://github.com/diamond264/ByMyEyes。
- **模型**：主要实验使用GPT-4o（OpenAI API）；小模型实验使用LLaVa-7B。
- **关键超参**：window size与sampling rate按原数据集规范（见Appendix F）；测试集每类30样本（UTD-MHAD每类10）；1-shot为主，另测3/5-shot；图像统一512×512像素。
- **可视化库**：Matplotlib、Scipy、Neurokit2，共16种可视化工具。
