---
title: "Towards-Tool-Use-Alignment-of-Large-Language-Models"
source: https://aclanthology.org/2024.emnlp-main.82.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:34:38"
field: "大语言模型工具使用对齐"
keywords: ["工具使用对齐", "H2A原则", "有害性对齐", "自主性", "偏好优化", "DPO", "大语言模型", "ToolAlign"]
innovations: ["提出H2A三维度对齐原则（有用性-无害性-自主性）", "构建ToolAlign多维度对齐数据集（46k指令+10k偏好）", "验证SFT+DPO两阶段训练在工具对齐中的有效性"]
benchmarks: ["ToolBench", "ToolSword", "MetaTool"]
---

# 论文速读：Towards Tool Use Alignment of Large Language Models

## 一句话总结
本文针对大语言模型在工具使用场景中需要兼顾**有用性（helpfulness）**、**无害性（harmlessness）**和**自主性（autonomy）**的三重对齐需求，提出了 **H2A 原则**并构建了 **ToolAlign 数据集**，通过在 ToolLLaMA 上依次进行 SFT 和 DPO 训练，使模型在保持工具调用能力的同时显著提升拒答有害指令/工具响应以及直接作答的能力。

## 研究问题与动机
- **现有工作只关注有用性**：已有工具学习研究（如 ToolBench、ToolLLaMA）主要聚焦增强 LLM 的工具调用能力（helpfulness），忽视了对齐人在工具使用场景下的价值观需求。
- **有害内容与不安全工具响应**：实际场景中，LLM 可能被诱导执行隐私窃取、危害性输出等指令；外部工具也可能被攻击返回钓鱼链接、恶意脚本等不安全响应，模型需具备识别与拒答能力。
- **冗余工具调用的成本问题**：对于常识问答等可直接回答的问题（如“三原色是什么？”），模型应直接作答以节省时间与金钱成本，而非一律调用工具。
- **缺乏统一的三维度对齐数据集**：虽有 ToolSword（聚焦有害性）、MetaTool（聚焦自主性）等单一维度评测，但没有同时覆盖有用性、无害性、自主性的综合对齐数据集。

## 核心贡献（创新点）
1. **提出 H2A 对齐原则**：首次系统性定义工具使用场景下 LLM 应遵循的“有用性-无害性-自主性”三重目标，明确了模型在工具调用中需兼顾的价值准则。
2. **构建 ToolAlign 数据集**：基于 ToolBench，扩充有害性（用户有害指令 + 不安全工具响应）与自主性（可直接回答的指令）数据，形成包含 46k 指令调优数据和 10k 偏好数据的多维度数据集。
3. **SFT + DPO 的两阶段对齐训练框架**：先用指令调优数据对 ToolLLaMA 做 SFT（AlignToolLLaMA-SFT）获得无害性与自主性基础能力，再用偏好数据进行 DPO（AlignToolLLaMA-DPO）进一步提升有用性，验证了该训练链路的有效性。
4. **实证结果表明显著增益**：AlignToolLLaMA-DPO 在有用性测试集平均通过率从 27.3% 提升至 49.8%，有害性拒答率达 98.7%，自主性直接回答率达 100%，并在 ToolSword、MetaTool 上展现出良好泛化。

## 方法详解
- **H2A 原则定义**：
  - **Helpfulness**：正确理解用户指令并通过调用外部工具生成信息丰富的回答。
  - **Harmlessness**：识别并拒答包含隐私窃取、违法建议、误导内容等有害指令；识别并拒答含钓鱼链接、恶意脚本、敏感信息索取等不安全工具响应。
  - **Autonomy**：对无需工具即可回答的查询直接作答，避免不必要的工具调用。

- **Instruction-tuning 数据集构建（46k 条）**：
  - **有用性**：从 ToolBench 采样指令-响应对。
  - **有害性**：
    - 用户有害指令（HI）：从 ToolBench 随机抽取 1k 条由 ChatGPT 改写为含隐私、非法、越界专业建议等不安全指令；另从 Anthropic Red Teaming Dataset 抽取 1k 条改写为 ToolBench 格式，并用 Sentence-BERT API 检索器为每条指令配 3-6 个相关工具 API。
    - 不安全工具响应（HTR）：构建四类（明显有害内容、钓鱼网站、攻击附件、敏感信息索取），替换 ToolBench 中的原始工具响应。
    - 拒答提示要求包含：明确说明不可答、指出有害部分及潜在危害、建议用户提出安全请求。
  - **自主性**：从 Alpaca 采样 3881 条指令，由 ChatGPT 改写为 ToolBench 风格，并使用 ToolBench 检索器检索相关 API，再让 ChatGPT 直接作答（不调工具）。

- **Preference 数据集构建（10k 条）**：
  - **有用性**：对每条指令取 ChatGPT 与 ToolLLaMA/AlignToolLLaMA-SFT 的响应，由 ChatGPT 评估是否完成任务，优先选 ChatGPT 响应为 chosen。
  - **有害性**：对有害指令，由 ChatGPT 生成拒答响应作为 chosen，ToolLLaMA 响应作为 rejected；对不安全工具响应，若 ChatGPT 未能识别则标记为 rejected，人工构造拒答模板为 chosen。
  - **自主性**：对改写后的自主指令，由 ChatGPT 直接作答；若 ToolLLaMA 未直接作答则选 ChatGPT，否则用 GPT-4 评估二者有用性得分后选高分者为 chosen。

- **模型训练配置**：
  - **SFT**：2 epochs，global batch size 64，peak lr=5e-5，4% warmup。
  - **DPO**：1 epoch，lr=1e-6，5 warmup steps，global batch size 8，β=0.05。
  - 硬件：4×Nvidia A100 40GB，bfloat16 精度。

## 实验与结果
- **评测指标**：
  - 有用性：Pass Rate（PR）、Win Rate（WR）[ToolEval 于 ToolBench]。
  - 有害性：Refusal Response Rate（3R）[HI 与 HTR 子集]。
  - 自主性：Direct Response Rate（DR2）。
  - 辅助：GPT-4 对拒答/直接回答质量的帮助性评分（1-5 分）。

- **核心结果（Table 2）**：
  - **ToolLLaMA**：HI 3R=0%，HTR 3R=0%，AU DR2=22.0%；I1-I PR=30.5%，WR=46.0%。
  - **AlignToolLLaMA-SFT**：HI 3R=96.4%，HTR 3R=100.0%，AU DR2=100%；I1-I PR=52.5%，WR=58.5%。相比 ToolLLaMA，有害性与自主性大幅提升，有用性略有下降（PR 27.3%）。
  - **AlignToolLLaMA-DPO**：I1-I PR=42.0%，WR=53.5%；I2-C PR=59.0%，WR=58.5%；I3-I PR=52.0%，WR=57.0%；平均 PR=49.8%，WR 显著高于 SFT 版。HI 3R=97.4%，HTR 3R=100%，AU DR2=100%。
  - 相较 GPT-4（HI 3R=85.6%，HTR 3R=76.5%，AU DR2=11.0%），本模型在有害性拒答与自主性上已超越 GPT-4。

- **消融（Table 3）**：
  - 仅对 preference chosen 再做 SFT（+SFT with Preference Data）：有用性 PR 降至 19.5%，有害性 HTR 3R 降至 81.5%，说明单纯 SFT 无法进一步对齐。
  - 直接在 ToolLLaMA 上做 DPO（+DPO with Preference Data）：HI/HTR 3R 均为 0%，AU DR2 仅 32%，说明无 SFT 基础时 DPO 难以学会有害性与自主性。

- **泛化（Table 4）**：
  - ToolSword-MQ：两模型拒答率 100%。
  - ToolSword-JA（越狱攻击）：AlignToolLLaMA-SFT 87.7%，AlignToolLLaMA-DPO 87.1%，接近 GPT-4 的 89.0%。
  - ToolSword-HF：SFT 95.3%，DPO 100%，均超过 GPT-4（40.7%）。
  - MetaTool 自主性：SFT 86.0%，DPO 98.0%，超过 GPT-4（28.0%）。

- **人类评估一致性（Section 5）**：GPT-4 评分与人工 Pearson 相关系数在 HI 集为 0.921，AU 集为 0.822，说明 GPT-4 自动化评测与人类判断高度一致。

## 相关工作脉络
1. **ToolBench / ToolLLaMA (Qin et al., 2023b)**：大规模工具调用数据集与基线模型，仅聚焦有用性。本文在其基础上补充有害性与自主性对齐。
2. **ToolSword (Ye et al., 2024)**：专门评测工具学习中有害性的基准（ malicious queries、jailbreak attacks、harmful feedback），但仅提供评测不涉训练。本文同时覆盖有害性训练与评测。
3. **MetaTool (Huang et al., 2023)**：聚焦自主性（是否应使用工具），提供 benchmark 但未提出系统对齐方案。本文引入自主性数据并验证训练效果。
4. **Safety-LLaMA (Bianchi et al., 2023)**：在通用指令上调优安全对齐，其对工具场景的有害性对齐效果有限（Qwen2-7B-Instruct HTR 3R=0%），本文证明需要专门针对工具场景构建对齐数据。
5. **DPO (Rafailov et al., 2024)**：直接偏好优化方法，本文首次将其引入工具使用场景的多维对齐训练，验证 SFT→DPO 两阶段设计的必要性。
6. **LLaMA-2-Chat / Qwen2-7B-Instruct**：通用安全对齐闭源/开源模型，在工具有害性（尤其 HTR）与自主性上表现较差（LA-2-chat HTR=0%、AU=0%；Qwen2 HTR=0%、AU=10%），凸显工具场景对齐的独立价值。

## 局限性与未来方向
- **单轮对话局限**：实验主要基于单轮指令-响应，未充分评估多轮对话场景中长期上下文、历史交互记忆下的工具调用安全与自主决策。
- **人类价值观复杂性**：H2A 仅覆盖三个维度，实际场景中人类价值更复杂（如文化差异、伦理边界），需更深入的价值理解。
- **模型规模限制**：当前基于 7B 参数模型，与 GPT-4 等大模型在知识容量与帮助性评分上仍有差距（Figure 2 中 AU 评分低于 GPT-4）。
- **未来方向**：扩展至多轮对话对齐、探索更细粒度的价值观对齐维度、在更大规模模型上验证泛化、开发工具使用场景的动态风险评估机制。

## 研究启发与可借鉴点
1. **原则驱动的对齐范式**：H2A 三维度原则框架可作为其他垂直场景（如代码生成、机器人控制、医疗决策）对齐研究的参考模板。
2. **SFT + DPO 两阶段链路设计**：证明先通过指令数据建立基础能力（无害性/自主性），再通过偏好数据强化有用性的训练顺序是关键，避免了单纯 DPO 导致的能力退化。
3. **跨数据集格式统一与改写**：将 ARTD、Alpaca 指令改写为 ToolBench 风格并配工具 API 检索，避免模型依赖数据集格式捷径，这一数据工程技巧可迁移至其他跨域对齐任务。
4. **偏好数据构造的对比策略**：利用 ChatGPT 作为高质量参考生成 chosen，同时保留弱模型响应作为 rejected，并以 GPT-4 辅助打分，这种构造方式兼顾了数据质量与多样性。
5. **自动化评测与人类评估一致性验证**：通过 Pearson 相关系数证明 GPT-4 评分与人类判断高度一致，为大规模评测提供了可信的自动化替代方案。

## 关键术语表
- **H2A（Helpfulness, Harmlessness, Autonomy）**：本文提出的工具使用对齐三原则，分别指有用性、无害性和自主性。
- **ToolAlign**：本文构建的多维度对齐数据集，包含 46k 指令调优数据和 10k 偏好数据。
- **ToolLLaMA**：在 ToolBench 上训练的工具调用基线模型，具有较强有用性但缺乏有害性与自主性对齐。
- **AlignToolLLaMA-SFT**：在 ToolAlign 指令数据上对 ToolLLaMA 做 SFT 得到的模型，具备基础有害性与自主性。
- **AlignToolLLaMA-DPO**：在 AlignToolLLaMA-SFT 基础上进一步 DPO 训练得到的模型，综合三维能力最优。
- **Pass Rate（PR）**：模型回答完成用户指令的比例，用于衡量有用性。
- **Refusal Response Rate（3R）**：模型对有害指令或不安全工具响应成功拒答的比例。
- **Direct Response Rate（DR2）**：模型在不调用工具的情况下直接作答的比例，衡量自主性。

## 可复现要素
- **数据集**：ToolAlign 指令调优数据与偏好数据，论文声明已开源（https://github.com/zhiyuanc2001/ToolAlign）。
- **代码/权重**：论文声明代码与数据集已开源；具体模型权重未在摘要中明确给出下载链接，需在 GitHub 仓库中查看。
- **SFT 超参**：epochs=2，global batch size=64，peak learning rate=5e-5，4% warmup，linear scheduler。
- **DPO 超参**：epochs=1，learning rate=1e-6，5 warmup steps，global batch size=8，β=0.05。
- **硬件**：4×Nvidia A100 40GB，bfloat16 精度。
- **基线模型**：ToolLLaMA v2、LLaMA-2-chat-7B、Qwen2-7B-Instruct（均公开可用）；闭源模型 ChatGPT、GPT-4、GPT-4o 需通过 API 调用。
- **评测工具**：ToolEval（ToolBench 自带）、GPT-4 turbo 用于有害性判定与帮助性评分。
