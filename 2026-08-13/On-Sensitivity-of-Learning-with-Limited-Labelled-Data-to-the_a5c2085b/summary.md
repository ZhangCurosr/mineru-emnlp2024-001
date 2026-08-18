---
title: "On-Sensitivity-of-Learning-with-Limited-Labelled-Data-to-the"
source: https://aclanthology.org/2024.emnlp-main.32.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:13:24"
field: "机器学习可重复性与稳定性分析"
keywords: ["随机性敏感性", "有限标签数据学习", "in-context learning", "随机性因子交互", "重要性评分"]
innovations: ["提出考虑交互作用的随机性因子解耦方法", "定义归一化重要性评分支持跨实验比较", "纠正in-context learning对样本顺序敏感性的先前结论"]
benchmarks: ["GLUE benchmark (SST2, CoLA, MRPC)", "AG News, TREC, SNIPS, DB-Pedia"]
---

# 论文速读：On Sensitivity of Learning with Limited Labelled Data to the Effects of Randomness

## 一句话总结
论文提出一种系统研究方法，显式建模随机性因子间的交互作用，量化各因子对有限标签数据学习方法（in-context learning、fine-tuning、meta-learning）性能偏差的贡献度，并揭示样本选择、类别数量、提示格式等系统性因素如何调节随机性因子的相对重要性。

## 研究问题与动机
- **核心问题**：有限标签数据学习方法（如in-context learning）对训练过程中的非确定性决策（随机性因子）高度敏感，且不同因子间存在交互作用，导致现有研究结论相互矛盾。
- **现有方法不足**：
  1. **Random策略**：同时变化所有随机因子，无法分离单个因子的独立效应，导致重要性被高估或低估。
  2. **Fixed策略**：固定其他因子为单一配置，结果严重依赖随机选取的配置点，仍无法充分缓解交互作用影响。
  3. **缺乏量化指标**：现有工作仅用二值判断（重要/不重要）评估因子，无法跨模型、数据集、实验设置进行深度比较。
- **关键矛盾案例**：此前研究发现in-context learning对样本顺序敏感，但后续研究使用更复杂的样本选择策略后该敏感性消失，暗示交互作用导致错误归因。

## 核心贡献（创新点）
- **系统化解耦方法**：提出基于"调查运行+缓解运行"的框架，通过多次随机固定其他因子配置并取平均，显式缓解交互作用，而非简单固定或同时变化所有因子。
- **重要性评分机制**：定义重要性分数 = (贡献标准差 - 缓解标准差) / 金模型标准差，实现因子重要性的归一化量化，支持跨实验设置的横向比较。
- **纠正先前结论**：证明in-context learning对样本顺序的敏感性并非持续存在，而是与样本选择、类别数量、提示格式等因素存在交互作用。
- **系统性选择的影响分析**：首次量化类别数量、每类样本数、提示格式等系统性超参数如何调节随机性因子的重要性分布。

## 方法详解
### 1. 方法框架（Algorithm 1）
- **随机性因子集** RF = {Label Selection, Data Split, Data Order, Sample Choice, Model Initialisation}，每个因子有K个配置。
- **调查配置集** IFC_i = C_i：待研究的因子i的所有可能状态。
- **缓解配置集** MFC_i = C_1 × ... × C_{i-1} × C_{i+1} × ... × C_K：其余因子的联合配置空间（笛卡尔积）。

### 2. 关键公式
- **部分标准差** p_std_m = std(r_{m,*})：固定其他因子配置m时，变化因子i的配置n得到的性能标准差。
- **贡献标准差** c_std = mean(p_std_*)：多次缓解运行的部分标准差平均值，代表因子i的最终效应。
- **缓解标准差** m_std = std(p_mean_*)：部分均值的标准差，代表其他因子的联合效应。
- **金模型标准差** gm_std = std(r_g)：所有因子配置组合下的性能标准差，作为基准。
- **重要性分数** importance = (c_std - m_std) / gm_std：因子i对总偏差的净贡献比例。

### 3. 参数选择启发式
- N（调查运行数）、M（缓解运行数）均远大于1，建议M ≥ N以更好缓解交互作用。
- L（金模型运行数）= N × M，确保重要性分数计算基于相同规模的分布。
- 消融实验表明：20次缓解运行 + 10次调查运行已足够稳定（贡献标准差变化<0.01）。

## 实验与结果
### 数据集与模型
- **7个文本分类数据集**：SST2、CoLA、MRPC（二分类），AG News、TREC、SNIPS、DB-Pedia（多分类）。
- **In-context learning模型**：Flan-T5 base、LLaMA-2 13B、Mistral-7B、Zephyr-7B（4-bit量化）。
- **Fine-tuning模型**：BERT base、RoBERTa base。
- **Meta-learning方法**：MAML、Reptile、Prototypical Networks。

### 关键结果
- **Sample Choice是in-context learning最重要的因子**：平均重要性分数0.25（跨模型/数据集），Flan-T5在多分类数据集上达0.39。
- **Data Order的重要性具有数据集和模型依赖性**：
  - 二分类数据集上大多数in-context learning模型对Data Order不敏感（平均重要性0.28）。
  - 多分类数据集上Data Order重要性显著上升（最高达0.29，SNIPS数据集Zephyr-7B）。
- **类别数量的影响**：
  - In-context learning：类别数增加 → Data Order重要性上升，Sample Choice重要性下降。
  - Fine-tuning：类别数增加 → Data Order和Model Initialisation重要性均下降。
- **样本数的影响**：增加每类样本数（2-shot→10-shot）→ Sample Choice重要性显著下降（Zephyr从0.39降至0.02），但对Data Order无一致影响。
- **提示格式的影响**：最小化格式（Format B/C/D）比优化格式（Format A）导致更高的Data Order重要性；大模型（Mistral、Zephyr）对提示格式变化更鲁棒。
- **最强结果**：Flan-T5在SST2上Data Order重要性为-0.47（不显著），而Zephyr-7B在SNIPS上Data Order重要性为0.29（显著），验证交互作用的存在。

## 相关工作脉络
- **Lu et al. (2022)、Zhao et al. (2021b)**：发现in-context learning对样本顺序敏感，但未考虑与其他因子（如样本选择）的交互作用，导致结论不可复现。
- **Zhang et al. (2022)、Chang & Jia (2023)、Li & Qiu (2023)**：使用复杂样本选择策略后观察到Data Order敏感性消失，本文证明这是交互作用导致的错误归因，而非策略本身消除敏感性。
- **Dodge et al. (2020)、McCoy et al. (2020)**：研究fine-tuning的随机种子影响，但未量化各因子的相对重要性，也未考虑因子间交互。
- **Mosbach et al. (2021)、Zhao et al. (2021a)**：关注训练顺序和权重初始化的影响，但同样忽视交互作用。
- **Weber et al. (2023)、Sclar et al. (2023)**：研究提示格式敏感性，但未将其与随机性因子重要性关联分析。
- **本文定位差异**：相比现有工作的孤立分析或二值判断，本文提供系统化解耦框架和归一化重要性评分，支持跨设置深度比较。

## 局限性与未来方向
- **模型规模限制**：实验使用较小模型（base版本BERT/RoBERTa/Flan-T5，4-bit量化大模型），较大模型可能表现出更高的随机性敏感性，本文结果可能低估因子重要性。
- **运行次数权衡**：虽经消融验证20次缓解运行已足够，但在更大数据集或更复杂任务上可能需更多运行次数以充分缓解交互作用。
- **测试样本数限制**：每轮评估仅用1000个测试样本，可能影响重要性分数的精确度，尤其对推理成本高昂的大模型。
- **未来方向**：
  1. 将方法扩展至问答、多模态等NLP子任务。
  2. 探索各随机性因子的有效缓解策略（如样本选择策略、集成方法）。
  3. 研究更大模型、更深层次交互作用的系统性影响。

## 研究启发与可借鉴点
- **重要性评分机制可复用**：提出的(c_std - m_std) / gm_std归一化框架可迁移至任何需量化多因子贡献度的机器学习稳定性分析场景。
- **实验设计借鉴**：采用"调查运行+缓解运行"交替策略平衡计算成本与分析精度，为类似研究提供可复现的参数选择指南（N=10, M=20/100）。
- **系统性超参数的调节作用**：揭示类别数量、样本数、提示格式等常被忽视的系统性选择如何调节随机性因子重要性，启发未来工作应将这些因素纳入实验设计。
- **交互作用的重要性**：证明忽略因子交互会导致错误归因，建议在有限数据学习研究中采用系统化解耦方法而非孤立分析单个因子。
- **可结合本团队方向**：若团队关注prompt tuning或少样本学习，本文提供的提示格式敏感性分析与样本选择策略对比可直接应用于优化实验设计。

## 关键术语表
- **Randomness Factor（随机性因子）**：训练过程中引入非确定性决策的因素，包括标签选择、数据划分、数据顺序、样本选择、模型初始化。
- **Interactions（交互作用）**：不同随机性因子对性能偏差的联合影响，一个因子的重要性可能因其他因子的配置而显著变化。
- **Importance Score（重要性评分）**：归一化指标，衡量单个随机性因子对总性能偏差的净贡献比例，范围可正可负。
- **Golden Model（金模型）**：固定所有随机性因子配置的基准模型，其性能标准差作为重要性评分的分母基准。
- **Mitigation Run（缓解运行）**：固定其他因子为不同配置、重复调查过程以平均化交互作用的实验运行。
- **Contributed Standard Deviation（贡献标准差）**：多次缓解运行的部分标准差平均值，代表单因子的最终效应估计。

## 可复现要素
- **数据集**：GLUEbenchmark（SST2、CoLA、MRPC）、AG News、TREC、SNIPS、DB-Pedia，均为公开数据集。
- **代码开源**：论文未提及代码开源状态，需联系作者获取。
- **模型权重**：使用HuggingFace公开模型（Flan-T5、LLaMA-2、Mistral-7B、Zephyr-7B、BERT、RoBERTa），均已公开。
- **关键超参**：
  - In-context learning：N=10（调查运行）、M=20（缓解运行）、L=200（金模型运行）。
  - Fine-tuning/Meta-learning：N=10、M=100、L=1000。
  - 测试样本：每轮评估使用1000个测试样本。
  - In-context学习shots：默认2-shot per class。
  - 量化设置：LLaMA-2、Mistral、Zephyr使用4-bit量化。
