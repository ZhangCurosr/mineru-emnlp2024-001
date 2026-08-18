---
title: "Towards Tool Use Alignment of Large Language Models"
source: https://aclanthology.org/2024.emnlp-main.82.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:10:27"
field: "大语言模型工具使用对齐"
keywords: ["Tool Use Alignment", "H2A Principle", "ToolAlign", "DPO", "LLM Safety", "Tool Learning", "Autonomy"]
innovations: ["提出H2A三维度对齐原则统一框架", "构建覆盖有用性/无害性/自主性的ToolAlign数据集", "SFT+DPO两阶段训练显著提升工具场景多目标对齐效果"]
benchmarks: ["ToolAlign", "ToolSword", "MetaTool", "ToolBench"]
---

# 论文速读：Towards Tool Use Alignment of Large Language Models

## 一句话总结
论文提出 H2A 原则（有用性、无害性、自主性），构建 ToolAlign 数据集，通过 SFT + DPO 训练使 LLM 在工具使用场景中同时具备精准调用工具、安全拒答有害内容、以及在可直答时不调用工具的能力。

## 研究问题与动机
- **现有工具调用研究偏向单一有用性目标**：ToolBench、MetaTool 等数据集和模型主要聚焦于增强 LLM 的工具调用能力（Helpfulness），忽视了工具使用场景下与安全、自主性相关的对齐需求。
- **LLM 在工具场景中易被恶意利用**：真实应用中 LLM 可能收到窃取隐私、诱导生成违法内容等有害指令；同时外部工具可能遭攻击或拦截，返回钓鱼链接、恶意脚本等不安全响应。现有开放模型（如 ToolLLaMA）对这两类风险几乎零拒答能力（3R = 0%）。
- **工具调用成本高，需支持自主直答**：每次调用外部工具均带来时间与金钱开销；对于常识问答、创意写作等可直接作答的查询，模型应自主决定不调用工具（Autonomy）。现有模型在该维度表现极差（ToolLLaMA DR = 22%）。
- **缺乏统一三维度对齐基准与数据**：尽管已有 ToolSword（安全性）和 MetaTool（自主性）等评测基准，但尚无同时覆盖 Helpfulness、Harmlessness、Autonomy 三者的对齐数据集与模型训练方案。

## 核心贡献（创新点）
1. **提出 H2A 对齐原则**：首次将 Helpfulness、Harmlessness、Autonomy 三者统一为工具使用场景下的对齐准则，填补了该方向的概念空白。*本质区别在于将以往分散研究的工具调用能力、安全性与自主决策纳入同一框架。*
2. **构建 ToolAlign 数据集**：包含 46k 指令微调数据和 10k 偏好数据，涵盖三维度样本；其中有害性指令通过改写 ToolBench 与 ARTD 构造，有害工具响应模拟四类真实攻击场景（恶意内容、钓鱼网站、攻击脚本、敏感信息索取）。*区别于既有单维数据集，本数据首次在工具调用语境下系统覆盖安全与自主维度。*
3. **提出 SFT + DPO 两阶段对齐训练流程**：先在 ToolAlign 指令数据上 SFT 训练 ToolLLaMA 得到 AlignToolLLaMA-SFT，再用偏好数据进行 DPO 得到 AlignToolLLaMA-DPO；消融证明两阶段缺一不可。*与单纯 SFT 或单纯 RLHF 不同，本工作验证了 DPO 在工具对齐中的增益。*
4. **系统评测与泛化验证**：在 ToolAlign 自建测试集及 ToolSword、MetaTool 外部基准上全面评测，AlignToolLLaMA-DPO 在有用性通过率（49.8%）、无害性拒绝率（98.7%）、自主直答率（100%）上均显著优于基线。*证明了方法的有效性与跨基准泛化能力。*

## 方法详解
### 2.1 H2A 原则
- **Helpfulness**：准确理解用户指令，调用恰当外部工具并综合工具返回给出 informative 响应。
- **Harmlessness**：识别并拒答包含隐私窃取、违法指导等有害指令；同时识别工具返回中的不安全内容（钓鱼链接、恶意脚本等）并拒答。
- **Autonomy**：对可直接回答的查询（如常识问题、创意写作），不调用任何工具直接输出答案，以节省成本与时间。

### 2.2 ToolAlign 数据构建
**指令微调数据集（46k 条）**：
- *Helpfulness*：从 ToolBench 直接采样 instruction-response 对。
- *Harmlessness*：
  - 有害指令：(1) 从 ToolBench 随机抽取 1k 条，用 ChatGPT 按 LLaMA-2 safeguarding rules 注入隐私、违法、无资质专业建议等要素改写；(2) 从 Anthropic Red Teaming Dataset (ARTD) 采样 1k 条，经 ChatGPT 改写以匹配 ToolBench 格式（背景+请求两段式），避免模型学到数据集格式捷径。
  - 有害工具响应：模拟四类真实威胁——明显有害内容（采样自 AdvBench）、钓鱼网站（伪装官方机构）、攻击附件（恶意 bash 脚本）、敏感信息索取；改写 841 条 ToolBench 指令并替换对应工具响应。
  - 标注：对有害指令，用结构化三段落模板生成拒答（声明有害→指出具体危害→建议安全请求）；对有害工具响应，使用固定模板填充工具名与危害类型。
- *Autonomy*：从 Alpaca 采样 3,881 条指令，经 ChatGPT 改写以匹配 ToolBench 格式，再用 ToolBench API retriever（基于 Sentence-BERT）检索相关 API，最后由 ChatGPT 生成直答。

**偏好数据集（10k 条）**：
- *Helpfulness*：对每条指令采样 ChatGPT 与 ToolLLaMA/AlignToolLLaMA-SFT 各一响应，由 ChatGPT 判断是否完成任务并择优；两者均成功时优先选 ChatGPT，均失败则丢弃。
- *Harmlessness*：有害指令直接用 ChatGPT 拒答作为 chosen，ToolLLaMA 响应作为 rejected；有害工具响应中 ChatGPT 未能识别的判为 rejected，人工构造拒答为 chosen。
- *Autonomy*：改写 300 条 Alpaca 指令并检索工具；若 ToolLLaMA 未直答则选 ChatGPT 为 chosen；若 ToolLLaMA 已直答则用 GPT-4 评估两者有用性，高分者为 chosen。

### 2.3 模型训练
- **SFT**：以 ToolLLaMA 为底座，在指令微调数据上训练 2 epochs，全局 batch size = 64，峰值学习率 5e-5，线性 warmup 占 4%。
- **DPO**：以 AlignToolLLaMA-SFT 为 policy model，以 ToolLLaMA 为 reference model，在偏好数据上训练 1 epoch，学习率 1e-6，warmup 5 steps，batch size = 8，β = 0.05。
- **硬件**：4× Nvidia A100 (40GB)，bfloat16 精度。

## 实验与结果
### 数据集与基线
- **测试集**：ToolAlign 自建（Helpfulness 800 条、Harmlessness HI 100 条+HTR 194 条、Autonomy 100 条）；泛化验证使用 ToolSword（MQ/JA/HF 子集）与 MetaTool（100 条）。
- **开源基线**：ToolLLaMA、LLaMA-2-chat-7B、Qwen2-7B-Instruct。
- **闭源基线**：ChatGPT、GPT-4、GPT-4o（均在 system prompt 中注入拒答与自主调用提示）。

### 主要结果
| 模型 | Helpfulness PR (avg) | Harmlessness 3R (HI/HTR) | Autonomy DR |
|---|---|---|---|
| ToolLLaMA | 32.7% | 0% / 0% | 22.0% |
| AlignToolLLaMA-SFT | 27.3% | 96.4% / 100.0% | 100.0% |
| AlignToolLLaMA-DPO | **49.8%** | **97.4% / 100.0%** | **100.0%** |
| GPT-4 | 57.2% | 85.6% / 76.5% | 11.0% |

- **最强结果**：AlignToolLLaMA-DPO 在有用性通过率上达 49.8%，较 SFT 版本提升 22.5 个百分点，仅次于 GPT-4（57.2%）；无害性拒绝率在 HI/HTR 上分别达 97.4%/100%，自主直答率 100%。
- **泛化**：在 ToolSword-JA（jailbreak 攻击）上拒绝率 87.1%~87.7%，与 GPT-4（89.0%）接近；在 ToolSword-HF 与 MetaTool 上均超越 GPT-4。
- **有用性得分**：GPT-4 在 HI/AU 测试集上的平均有用性得分（4.73/4.73）仍高于 AlignToolLLaMA-DPO（4.87/3.86），作者归因于模型规模与知识容量限制。

### 消融实验
- **+SFT with Preference Data**（仅对 chosen 样本继续 SFT）：有用性 PR 降至 12.9%，无害性与自主性亦轻微下降，证明单纯 SFT 不足以进一步优化。
- **+DPO with Preference Data**（跳过 SFT 直接在 ToolLLaMA 上 DPO）：无害性 3R 仍为 0%，自主性仅从 22.0% 提升至 32.0%，证明 SFT 是获取基础无害性与自主性的必要条件。

## 相关工作脉络
1. **Tool learning for LLMs（ToolBench、ToolLLaMA、ToolAlpaca 等）**：聚焦于增强 LLM 的工具调用能力（Helpfulness），本文在其基础上补充 Harmlessness 与 Autonomy 维度，填补三维度对齐空白。
2. **ToolSword（Ye et al., 2024）**：首个系统评估工具学习中安全问题的基准，涵盖有害指令与有害工具响应；本文在数据集构造与评测上借鉴其思路，但扩展至三维度统一框架。
3. **MetaTool（Huang et al., 2023）**：评估 LLM 工具使用决策（是否需要调用工具、选择哪个工具）；本文的 Autonomy 维度与之互补，侧重"不调用工具直接回答"的能力。
4. **LLaMA-2 safety alignment / Safety-tuned LLaMAS**：通用对话场景的安全对齐方法在工具使用场景泛化有限（Qwen2-7B-Instruct 在 HTR 上 3R=0%），本文证明需针对工具场景专门构建对齐数据。
5. **DPO（Rafailov et al., 2024）**：本文采用 DPO 而非 RLHF 进行偏好优化，验证了 DPO 在工具对齐任务上的高效性（仅需 1 epoch 即带来显著有用性提升）。
6. **Constitutional AI / UltraFeedback**：理念上与本文一致——通过规则/反馈引导模型对齐，但本文聚焦于工具使用这一特定场景，提出 H2A 三维度准则。

## 局限性与未来方向
- **价值观复杂性**：现实世界中人类价值观远比 H2A 三维度复杂，需更深入理解多元价值以实现更精细的对齐。
- **多轮对话缺失**：当前实验仅涉及单轮交互，未评估模型在多轮对话中维持上下文、整合历史信息、持续安全调用工具的能力。
- **模型规模限制**：7B 参数模型的有用性得分与 GPT-4 仍有差距，推测与知识容量相关，未在小模型上完全释放 H2A 对齐潜力。
- **未来方向**：扩展至多轮对话场景、探索更细粒度的价值对齐维度、在更大规模模型上验证 H2A 原则的普适性。

## 研究启发与可借鉴点
1. **H2A 三维度框架可迁移**：对于其他 Agent 场景（如代码生成、角色扮演、RAG 系统），可类比提出类似的多目标对齐原则，避免单一维度优化导致的 trade-off。
2. **SFT + DPO 两阶段训练策略**：SFT 负责赋予基础能力（无害拒答、自主直答），DPO 负责在偏好信号下进一步提升质量（有用性），该策略在工具对齐任务中效果显著，值得在其他对齐任务中验证。
3. **跨数据集格式一致性设计**：ARTD 与 Alpaca 指令格式与 ToolBench 存在差异，作者通过 ChatGPT 改写统一格式以避免模型学到数据集捷径，这一数据工程技巧值得借鉴。
4. **偏好数据构造的分级策略**：Helpfulness 偏好用 ChatGPT 评判任务完成度；Autonomy 偏好在双方均直答时用 GPT-4 评分择优；Harmlessness 偏好直接以模型能力差异定 chosen/rejected，可根据不同维度灵活设计偏好标签逻辑。
5. **拒答质量评估**：不仅计算 3R，还引入 GPT-4 对人机协作友好的拒答内容评分（1~5 分），证明拒答"为何有害"比单纯拒绝更有价值，这一评估思路可推广。

## 关键术语表
- **H2A**：Helpfulness（有用性）、Harmlessness（无害性）、Autonomy（自主性）三个对齐维度的统称。
- **ToolAlign**：本文构建的用于工具使用对齐的数据集，包含 46k 指令微调数据与 10k 偏好数据。
- **ToolLLaMA**：基于 ToolBench 训练的 7B 开源工具调用模型，作为本文 SFT/DPO 训练的基础模型。
- **DPO (Direct Preference Optimization)**：直接偏好优化方法，无需显式奖励模型，通过对比 chosen/rejected 对直接优化策略。
- **SFT (Supervised Fine-Tuning)**：监督微调，在指令-响应配对数据上微调模型以学习特定能力。
- **Pass Rate (PR)**：模型响应正确完成指令的比例，用于评估 Helpfulness。
- **Refusal Response Rate (3R)**：模型正确拒答有害指令或有害工具响应的比例，用于评估 Harmlessness。
- **Direct Response Rate (DR2)**：模型在未调用任何工具的情况下直接给出答案的比例，用于评估 Autonomy。

## 可复现要素
- **数据集**：ToolAlign 已开源（https://github.com/zhiyuanc2001/ToolAlign）；ToolBench、ToolSword、MetaTool 均为公开基准。
- **代码**：已开源。
- **权重**：AlignToolLLaMA-SFT 与 AlignToolLLaMA-DPO 随代码一起发布。
- **基础模型**：ToolLLaMA（基于 LLaMA-2-7B）。
- **关键超参**：SFT—2 epochs，batch size=64，学习率=5e-5，warmup=4%；DPO—1 epoch，batch size=8，学习率=1e-6，warmup=5 steps，β=0.05。
- **硬件**：4× Nvidia A100 (40GB)，bfloat16 精度。
