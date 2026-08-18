---
title: "Tracking-the-perspectives-of-interacting-language-models"
source: https://aclanthology.org/2024.emnlp-main.90.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:34:55"
field: "多Agent语言模型系统"
keywords: ["LLM交互系统", "视角空间", "信息扩散", "模型演化", "对抗性视角", "极化", "图神经网络", "surrogate data kernel"]
innovations: ["提出视角空间方法定量追踪多LLM交互系统的信息扩散", "将LLM系统建模为通信图网络并研究拓扑结构对演化的影响", "设计iso-mirror和polarization等系统级动力学分析指标"]
benchmarks: ["Yahoo! Answers", "Databricks Dolly 15k"]
---

# 论文速读：Tracking the perspectives of interacting language models

## 一句话总结
本文提出将大规模语言模型交互系统建模为通信图网络，并引入**视角空间（perspective space）**这一基于嵌入的量化方法，用于系统性追踪和分析多模型系统中信息扩散的动态演化过程。

## 研究问题与动机
- **核心问题**：LLM产生的内容正日益广泛地传播于互联网并进入其他模型的训练/检索数据，如何量化分析这种"模型间相互影响"系统中的信息扩散行为？
- **现有方法不足一**：当前研究仅依赖系统提示词注入差异化人格/观点来构造多样性，但缺乏对模型初始化差异及演化过程的**定量评估**。
- **现有方法不足二**：对响应变化的分析局限于人工检查或将其分类为少数几个类别，无法捕捉连续、高维的视角演化轨迹。
- **现有方法不足三**：缺乏形式化框架来研究不同通信拓扑结构对信息扩散的因果影响。

## 核心贡献（创新点）
1. **将LLM交互系统形式化为有向图**：以模型和数据库为节点、以信息流动为边，建立了可系统化干预的研究框架。
2. **提出视角空间（perspective space）**：通过surrogate data kernel + CMDS将多模型响应差异映射到低维欧氏空间，实现对模型视角的定量表征。
3. **设计iso-mirror与polarization等系统级动力学指标**：分别从整体演化稳定性与群体分化程度刻画系统行为。
4. **三项模拟案例研究**：系统揭示通信拓扑（全局/局部/对抗/回声室）对信息扩散、极化和模型崩溃的深层影响。

## 方法详解
- **通信图建模**：设系统包含模型集合 $\mathcal{F} = \{f_1, \ldots, f_n\}$ 和数据库集合 $\mathcal{D} = \{D_1, \ldots, D_{n'}\}$，构建图 $G = (V, E)$，其中 $V = \mathcal{F} \cup \mathcal{D}$，有向边 $(v, v') \in E$ 表示 $v$ 可影响 $v'$（如模型可从数据库检索，或模型输出可更新数据库）。
- **Surrogate Data Kernel**：对每个模型 $f_i$，用固定 prompt 集 $\mathbf{X}$ 生成响应 $f_i(\mathbf{X})$，再通过**替代嵌入函数** $\tilde{g}$（不依赖于任何具体LLM）将响应映射为矩阵 $\tilde{g}(f_i(\mathbf{X})) \in \mathbb{R}^{m \times p}$。
- **视角空间构建**：计算所有模型对的 Frobenius 距离 $\tilde{M}_{ij} = \|\tilde{g}(f_i(\mathbf{X})) - \tilde{g}(f_j(\mathbf{X}))\|_F$，再对距离矩阵做**经典多维缩放（CMDS）**，得到各模型在 $d$ 维欧氏空间中的坐标 $Z_i$，这些坐标构成视角空间。
- **动力学追踪**：通过时间序列 $t = 1, \ldots, T$ 重复上述过程，可构建包含完整历史的视角空间，进而计算 iso-mirror（系统动力学总结）、聚类稳定性（ARI）、极化度等指标。

## 实验与结果
- **基础模型**：Pythia-410M，经 Dolly 15k 指令微调后，再在 Yahoo! Answers 不同主题上微调产生多样性。
- **嵌入函数**：all-MiniLM-L6-v2。
- **案例研究1（通信网络中断）**：25个模型，从完全连通突变为局部连通（$t^*=21$）。中断后模型从全局 sink 收敛到局部 sink，iso-mirror 在 $t^*$ 处出现明显突变并趋于稳定；无中断系统则持续不稳定振荡。
- **案例研究2（对抗性视角扩散）**：6个模型中1个为"对抗模型"（Science & Math主题），其余5个为"目标模型"（Society & Culture主题）。当攻击频率为每2步一次且有5个目标时，整个系统的平均视角在 $t=20$ 前已被完全同化；攻击间隔更长时系统可恢复。
- **案例研究3（极化缓解/促进）**：10个模型分属两类主题。在全连通网络下两极化平均趋于0；在 intra-class only（类内通信）网络下，极化度平均提升 **15倍**，两类分别收敛到距离很远的两个 sink。
- **最强结果**：极化度在回声室结构下提升15倍；对抗模型仅影响少数目标即可通过级联效应完全控制全网。

## 相关工作脉络
- **Generative Agents（Park et al., 2023）**：LLM智能体模拟人类社会行为，本文与其互补——前者关注行为层面模拟，本文聚焦定量视角追踪框架。
- **Opinion Dynamics via Classification（Chuang et al., 2023）**：将LLM响应分类为少数观点类别；本文的方法无需预定义类别，直接在连续嵌入空间追踪演化。
- **Cultural Evolution in LLM Populations（Perez et al., 2024）**：研究LLM群体文化演化；本文的贡献在于提供形式化图框架和定量分析工具（视角空间、iso-mirror等）。
- **Model Collapse（Shumailov et al., 2024）**：单个模型递归训练于自身输出导致性能退化；本文将其推广到多模型图网络场景，发现多sink现象。
- **Data Kernel for Foundation Models（Duderstadt et al., 2023）**：本文的核心技术基础，但原方法要求模型自带嵌入函数；本文通过surrogate kernel解决GPT/Claude等黑盒模型不可用的问题。

## 局限性与未来方向
- **交互机制过于简化**：每次仅与一个邻居交互、fine-tuning一步更新，不反映真实社区中复杂的互动模式。
- **案例研究配置有限**：仅测试三种拓扑结构，缺少对模型数量、初始分布、"局部"定义等变量的系统消融。
- **对抗模型假设单一**：第二案例仅考虑静态对抗者，现实中可能存在多个动态对抗者。
- **未全面验证视角空间的鲁棒性**：对评估集敏感，且未探索其他模型相似度度量方式。
- **社会代理有效性待验证**：需与社会学/认知科学家合作，评估LLM系统作为人类社区代理的合理性。

## 研究启发与可借鉴点
- **Surrogate Data Kernel设计**：通过外部嵌入函数绕过黑盒模型的嵌入缺失问题，可迁移到任何需要比较LLM输出的场景。
- **评估集选择敏感性原则**：Figure 2证明评估prompt需与目标差异相关，这一设计原则可指导后续模型基准测试的构建。
- **系统级动力学指标**：iso-mirror、polarization等抽象指标可有效压缩高维视角演化轨迹，适合用于监控AI系统的长期行为。
- **图通信拓扑的实验操控**：为研究LLM生态中的信息茧房、谣言扩散、对抗攻击提供了可控的实验平台。
- **可与本团队方向结合**：视角空间方法可直接应用于评估RAG系统、多Agent协作、模型蒸馏链路中的信息保真度衰减。

## 关键术语表
- **Perspective Space（视角空间）**：通过CMDS将多模型在固定prompt集上的响应差异映射到低维欧氏空间，用于可视化与定量分析模型观点分布。
- **Surrogate Data Kernel（替代数据核）**：使用独立于研究模型的外部嵌入函数 $\tilde{g}$ 对LLM输出进行编码，解决了黑盒模型无可用嵌入的问题。
- **CMDS（Classical Multidimensional Scaling）**：经典多维缩放，从距离矩阵恢复低维欧氏坐标的降维技术，是构建视角空间的核心数学工具。
- **Model Sink（模型汇聚点）**：模型视角演化过程中趋于稳定的吸引子区域，对应于模型输出的局部最优稳态。
- **Iso-mirror**：系统级动力学摘要指标，通过比较视角空间的时间序列分布检测整体演化行为的变化点。
- **Polarization（极化度）**：衡量不同群体（如两类模型）平均视角之间距离的归一化指标，用于量化回声室效应。
- **Communication Network（通信网络）**：将模型和数据库视为节点、信息流视为边的有向图结构，用于形式化LLM交互系统。
- **Adversarial Model（对抗模型）**：初始视角与其他模型显著不同的模型，用于模拟外部信息注入或 propaganda 攻击。

## 可复现要素
- **数据集**：Yahoo! Answers（Zhang et al., 2015）、Databricks Dolly 15k，均已公开。
- **代码/权重**：论文未明确声明开源；使用了 Pythia-410M、all-MiniLM-L6-v2、Graspologic 等公开组件。
- **关键超参**：指令微调 lr=$5\times10^{-5}$，batch=8，10 epochs；细调 lr=$5\times10^{-5}$，3 epochs；交互更新 lr=$10^{-5}$，1 epoch。
