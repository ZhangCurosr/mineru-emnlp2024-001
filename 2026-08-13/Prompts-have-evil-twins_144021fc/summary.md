---
title: "Prompts-have-evil-twins"
source: https://aclanthology.org/2024.emnlp-main.4.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:15:43"
field: "大语言模型提示工程与可解释性"
keywords: ["prompt optimization", "evil twin prompts", "KL divergence", "hard prompt", "GCG", "transferability", "prompt robustness"]
innovations: ["提出基于KL散度的提示功能相似性度量并将其转化为最大似然优化", "证明不可读提示可逼近人类可读提示的功能并在多模型间迁移", "系统对比GCG及可解释性约束在功能还原中的效果与trade-off"]
benchmarks: ["Alpaca", "HellaSwag", "OpenHermes-2.5", "Pythia suite", "Vicuna-7b-v1.5"]
---

# 论文速读：Prompts-have-evil-twins

## 一句话总结
该论文证明，许多人类自然语言提示词都可以被替换为对人类不可读但经证明能诱导语言模型产生相似行为的“邪恶双胞胎（evil twins）”提示词；这些不可读的提示词通过求解一个最大似然优化问题找到，并能在多种开源和闭源模型之间迁移。

## 研究问题与动机
- 核心问题：语言模型提示词是否必须对 humans 可理解，才能诱导期望行为？
- 现有方法的不足：主流 LLM 交互依赖人类可读指令提示词，但对模型如何解析提示词仍缺乏系统性的量化理解；对抗攻击类方法仅关注安全破坏目标，未将“功能等价性”作为一般优化目标。
- 工程/安全动机：若不可读提示可达到同样甚至更好的功能性，则提示词工程存在被忽视的优化空间；同时不可解释提示可能被用于绕过安全过滤。
- 理论动机：提出一种基于 KL 散度的“提示词功能相似性”量化度量，并推导为可优化的最大似然问题。

## 核心贡献（创新点）
- 提出用 Kullback-Leibler 散度定量衡量两个提示词 induced distribution 的功能相似性，从而把“提示词是否等价”从定性判断转为可计算指标。与以往多凭人工评估不同，本文给出信息论框架下的严格距离。
- 将提示词等价问题形式化为一个离散最大似然估计问题：给定若干由原始提示词生成的文档，寻找使这些文档似然最大的新提示词。这与 PEZ 等方法依赖共享 embedding 空间不同，本文仅需模型能对给定 prompt 计算条件 log-likelihood。
- 展示“evil twin”现象普遍存在：利用 GCG（Greedy Coordinate Gradient）并结合 warm start、pruning 与 fluency penalty 等变体，能在 Alpaca 等指令数据上找到比 GPT-4 重述更贴近原始提示功能的提示词。与既有方法相比，本文强调通用功能还原而非特定攻击目标。
- 系统分析 evil twin 的可迁移性与鲁棒性：发现优化得到的提示词在多种开源/闭源模型间可迁移，并在不同模型族内对 token 重排/替换呈现差异化的敏感度。这与仅在小范围或单模型上评估的已有工作形成对比。
- 探索可解释性约束的边界：加入 fluency penalty 和限制 token 词汇并不能显著改善 KL 接近度，说明“功能等价”与“人类可读”之间存在张力。

## 方法详解
- 功能相似性度量：给定提示词 $p$ 与 $p^*$，其在模型下的输出分布分别为 $\mathbb{P}_{LLM}(\cdot|p)$ 与 $\mathbb{P}_{LLM}(\cdot|p^*)$，定义
  $d_{KL}(p^* \| p) = \mathrm{KL}(\mathbb{P}_{LLM}(\cdot|p^*) \| \mathbb{P}_{LLM}(\cdot|p))$。
- 经验估计：从 $p^*$ 采样文档 $d_1, \ldots, d_n$，用
  $\hat{d}_{KL}^{(n)}(p^* \| p) = \frac{1}{n}\sum_i \left[\log \mathbb{P}_{LLM}(d_i|p^*) - \log \mathbb{P}_{LLM}(d_i|p)\right]$
  近似 KL。
- 优化目标：去掉与 $p$ 无关的常数项后，等价于最小化
  $L(p; d_1, \ldots, d_n) = -\sum_i \log \mathbb{P}_{LLM}(d_i | p)$，
  即求 hard prompt 的最大似然估计。
- 离散优化实现：由于 prompt 是离散 token 序列，采用 Greedy Coordinate Gradient (GCG) 进行逐位置 top-k token 替换搜索；同时实验 warm start（用 GPT-4 生成的候选 prompt 初始化）、fluency penalty（在损失中加入 $\gamma \log \mathbb{P}_{LLM}(p)$ 以偏好更自然 prompt）以及 vocabulary pruning（掩码掉非英文常见子词）。
- Soft prompt 对照：附录中也给出软 prompt 情形，使用梯度下降直接优化 embedding 矩阵 $Z \in \mathbb{R}^{k_p \times d}$，损失同为负对数似然；随样本量增加 KL 收敛。

## 实验与结果
- 数据集与模型：以 Alpaca 指令微调数据集（以及 HellaSwag、OpenHermes-2.5 等）中的提示词为主；主要评测模型包括 Vicuna-7B/13B、Llama2-7b/13b-chat、OpenHermes、Mistral 系列、Pythia 系列、Phi-2、Gemma 等，并迁移至 GPT-3.5-turbo、GPT-4、Claude 3 Haiku/Sonnet、Gemini Pro 等商业模型。
- 评估方式：计算优化后提示词相对 ground truth 的近似 KL 散度；跨模型迁移时采用 GPT-4 judge 按 1–3 分评估生成响应是否忠实于原提示。
- 关键结果：在图 1 示例中，如“Offer an opinion on the problems that could arise from using AI”，ground truth 为 0.0；GPT-4 reconstruction 为 14.0±0.5；优化后的 evil twin 为 4.3±0.4。另一例“Create a data model for a driver on a car-sharing platform”，GPT-4 为 15.9±0.4，优化后为 1.6±0.2。总体看，warm-start GCG 表现最佳，通常显著低于 GPT-4 直接重述和 cold-start GCG。
- 跨模型迁移：除 Claude 3 Haiku 外，多数开源与闭源模型上超过 50% 的优化提示词获得最高评分（score=3），例如 Gemini Pro、GPT-4、Llama2 等均表现良好。
- 模型大小间的迁移：较小模型上优化得到的提示词向较大模型迁移时性能较差，但较大模型上的优化提示词能良好迁移到较小模型。
- 鲁棒性：token 顺序敏感度因模型族而异（Pythia、Phi-2、Gemma 上优化 prompt 较地面真值更不敏感；Mistral 上更敏感；Vicuna 上差异不大）。而 token 替换敏感度方面，优化 prompt 通常比地面真值对单 token 替换更敏感，说明不可读 token 也承担功能作用。
- 可解释性约束效果：加入 fluency penalty 或词汇剪枝后，prompt 可读性有所提升，但 KL 散度并未改善，说明功能逼近与人类可读之间存在 trade-off。

## 相关工作脉络
- Prompt parsing 研究：Webson & Pavlick (2022)、Min et al. (2022)、Jang et al. (2023) 等表明 LLM 对 natural language instruction、few-shot label、否定形式等的解析与人类直觉不符；本文进一步证明即使是完全不可读文本也可保持功能性。
- 对抗攻击与 jailbreak：Zou et al. (2023) GCG、Liu et al. (2023) AutoDAN、Ebrahimi et al. (2018) HotFlip 等专注于生成恶意或越狱提示；本文借用 GCG 思想，但目标函数改为“功能等价”而非“诱导特定有害输出”。
- Soft/hard prompt tuning：Prefix tuning (Li & Liang, 2021)、P-Tuning (Lester et al., 2021) 等优化连续向量；AutoPrompt (Shin et al., 2020) 在离散 token 空间搜索。本文聚焦硬提示且目标为一般指令等效。
- PEZ (Wen et al., 2023)：利用多模态共享 embedding 空间反推图像对应的 prompt；本文不依赖多模态对齐，只需条件似然可计算，适用于纯对话式 LLM。
- Robustness / order-sensitivity：Ishibashi et al. (2023) 评估 discrete prompt 的重排鲁棒性；本文扩展至 KL 框架下的顺序与替换敏感性的跨族对比。
- 防御与安全性相关：Jain et al. (2023)、Cherepanova & Zou (2024) 研究不可解释输入的处理；本文揭示“不可解释但功能等价”的现象，并讨论安全含义与缓解手段。

## 局限性与未来方向
- 优化稳定性：GCG 并非在所有情况下都能稳定收敛到低 KL，部分实例未能找到良好解；需要探索更稳定的离散优化算法（支持多 token 插入/删除、长度可变等）。
- 对白盒梯度的依赖：寻找 evil twin 需要模型梯度访问，无法直接用于 GPT-4 等闭源模型；虽可通过开源模型优化后迁移应用，但迁移并非总能完美。
- 可解释性与功能性权衡：加入 fluency penalty 或限制词汇虽提高可读性，但未改善功能相似度，尚未找到兼顾两者的高效方案。
- 运行开销：GCG 收敛可能需要较多迭代，面向 prompt compression、conditional generation 等实际应用的延迟成本仍需进一步工程化。
- 安全风险评估有限：虽提到可被滥用构造隐蔽有害 prompt，但系统性地评估此类风险及防护策略仍是未来方向。

## 研究启发与可借鉴点
- 可复用的功能等价度量：用 KL 散度衡量 prompt 间行为分布相似度，并可推广到不同任务/模型对比或自动化提示评估流水线。
- 优化范式迁移：将 GCG 应用于非对抗目标（功能性还原）展示了离散 prompt 优化的通用性，可用于 prompt compression、prompt 蒸馏或风格迁移等下游任务。
- 实验设计参考：使用 GPT-4 judge 进行跨模型一致性和忠实度评估、结合 toy 示例表与大规模 win-rate 曲线，兼顾可解释性与统计显著性。
- 约束探索启示：fluency 和 vocabulary pruning 的失败说明单纯正则不足以同时维持可读性与功能等价，可启发后续研究在 latent space、语义约束或人类反馈信号上做更精细设计。
- 团队结合机会：若团队关注 prompt engineering、模型 interpretability 或 red-teaming，可在此基础上探索“可控可读的 evil twin”、跨模型提示迁移的边界条件及安全检测器设计。

## 关键术语表
- **Evil twin prompt**：对 human 不可读但在功能上能诱导 LLM 产生与原始自然语言提示相似输出的优化提示词。
- **KL divergence between prompts**：通过比较两提示词诱导的输出分布的距离，作为功能相似性的信息论度量。
- **Greedy Coordinate Gradient (GCG)**：一种在离散 token 空间逐位置做 top-k 梯度搜索并替换以最小化目标损失的优化算法。
- **Warm start / cold start**：warm start 指用外部（如 GPT-4）候选提示初始化优化过程；cold start 指从随机 token 序列开始。
- **Fluency penalty**：在优化损失中加入 $\gamma \log \mathbb{P}_{LLM}(p)$ 项，鼓励优化出的 prompt 本身更具语言流畅性。
- **Vocabulary pruning**：通过过滤 tokenizer 中非常用子词（如仅保留英文常见 token）来限制优化搜索空间。
- **Token-order sensitivity**：衡量 prompt 在随机打乱 token 顺序后功能分布变化程度的指标。
- **Soft prompt**：位于 embedding 空间的连续向量序列，可直接用梯度下降优化；与硬 prompt 的离散 token 表示相对。

## 可复现要素
- 数据集：Alpaca 指令微调数据集（Taori et al., 2023）、HellaSwag、OpenHermes-2.5 等；多数为公开数据集。
- 模型：Vicuna、Llama2、Mistral、Pythia、Phi-2、Gemma 等开源模型；GPT-3.5-turbo、GPT-4、Claude 3、Gemini Pro 等商业模型用于评估/迁移。
- 代码/权重：论文未明确声明开源仓库链接（仅引用原始 GCG、Alpaca 等来源）；具体实现细节见附录 Algorithm 2 与附录 E 的结果表。
- 关键超参：fluency 参数 $\gamma \in \{0.01, 0.05, 0.1, 1.0\}$；优化轮数 50/100 epochs（依数据集与模型而定）；GCG top-k 替换策略；词汇剪枝大致移除 15,000 个非英文常见 token。
