---
title: "Towards Tool Use Alignment of Large Language Models"
source: https://aclanthology.org/2024.emnlp-main.82.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:10:02"
field: "工具使用场景下的大语言模型对齐"
keywords: ["Tool Alignment", "Large Language Models", "Harmlessness", "Autonomy", "Preference Optimization", "DPO", "Tool Learning"]
innovations: ["提出H2A三维度对齐原则（帮助性、无害性、自主性）并构建ToolAlign数据集", "设计SFT+DPO两阶段训练流程实现多维能力协同优化", "在开源模型上实现超越GPT-4的无害性与自主性表现"]
benchmarks: ["ToolAlign", "ToolSword", "MetaTool", "ToolBench"]
---

# 论文速读：Towards Tool Use Alignment of Large Language Models

## 一句话总结
本文提出H2A对齐原则（Helpfulness、Harmlessness、Autonomy），构建ToolAlign数据集并通过SFT+DPO训练，使开源LLM在工具调用场景下同时具备帮助性、无害性与自主性，显著优于现有开源模型。

## 研究问题与动机
- **已有工作偏向帮助性**：现有工具学习研究（如ToolBench、ToolLLaMA）主要聚焦增强LLM的工具调用能力（帮助性），忽视了真实场景中LLM需与人类价值观对齐。
- **无害性缺失**：LLM可能被恶意利用输出有害内容（如窃取隐私、传播危险信息），且外部工具可能返回不安全响应（钓鱼链接、恶意代码），现有模型几乎无法拒绝此类请求（ToolLLaMA在有害指令上拒绝率为0%）。
- **自主性不足**：对于可直接回答的简单查询（如常识问题），LLM仍倾向于调用工具，增加时间与经济成本；ToolLLaMA在自主性测试集上仅22%直接回答。
- **开源模型与闭源模型差距大**：GPT-4虽具备一定无害性，但自主性仅11%；闭源ChatGPT在无害性与自主性上几乎为零；开源模型整体在这两个维度表现更差，缺乏系统对齐方法。

## 核心贡献（创新点）
1. **提出H2A对齐原则**：首次系统性定义工具使用场景下LLM应具备的三个维度——帮助性（正确调用工具）、无害性（拒绝有害指令与不安全工具响应）、自主性（可直答时不调用工具），填补了多维对齐的理论空白。
2. **构建ToolAlign数据集**：基于ToolBench构建包含46k指令微调数据与10k偏好数据的复合数据集，覆盖帮助性（40k）、无害性（2.8k有害指令+0.8k不安全工具响应）、自主性（3.9k）三类任务，是目前唯一同时涵盖三维度工具对齐数据的公开资源。
3. **设计SFT+DPO两阶段对齐训练流程**：先通过SFT让模型习得无害性与自主性基础能力，再通过DPO优化偏好选择，显著提升帮助性；消融实验证明两阶段缺一不可（仅SFT或仅DPO均无法同时兼顾三维指标）。

## 方法详解
- **H2A原则定义**：帮助性要求LLM理解指令并准确调用外部工具；无害性要求拒绝危险用户指令与不安全工具响应；自主性要求LLM在无需工具时直接作答以节省成本。
- **指令微调数据集构建**：
  - 帮助性：从ToolBench采样40k指令-响应对。
  - 无害性：通过两种途径生成——(1)将ToolBench中1k指令改写为含隐私/非法/无资质建议的有害指令，结合Llama-2安全规则；(2)从Anthropic红队数据集采样1k有害指令并用ChatGPT重写以匹配ToolBench格式；同时对841个工具的响应注入四类不安全内容（明显有害内容、钓鱼网站、攻击附件、敏感信息索取）。
  - 自主性：从Alpaca采样3.9k指令，经ChatGPT重写匹配ToolBench格式后直接生成响应。
  - 所有响应由ChatGPT(gpt-3.5-turbo)生成，有害响应采用三段落拒绝模板（明确拒绝→解释危害→建议安全请求）。
- **偏好数据集构建**：10k指令（帮助性10k、无害性600+300、自主性300），每指令取ChatGPT响应与ToolLLaMA/AlignToolLLaMA-SFT响应作为chosen/rejected对，由ChatGPT评估生成偏好标签。
- **训练流程**：
  - SFT：在ToolAlign指令数据集上训练2个epoch，全局batch size=64，峰值学习率5e-5，4% warmup。
  - DPO：在偏好数据集上训练1个epoch，学习率1e-6，5步warmup，batch size=8，β=0.05。
  - 基座模型选用已具备工具调用能力的ToolLLaMA，分别得到AlignToolLLaMA-SFT与AlignToolLLaMA-DPO。

## 实验与结果
- **评估基准**：自建ToolAlign测试集（帮助性7个类别共1400指令、有害指令100、不安全工具响应194、自主性100），并在ToolSword（恶意查询、越狱攻击、有害反馈）与MetaTool（工具使用意识）上验证泛化性。
- **核心结果**：
  - AlignToolLLaMA-SFT vs ToolLLaMA：有害指令拒绝率从0%升至96.4%，自主性从22.0%升至100.0%，但帮助性Pass Rate从32.7%降至27.3%（存在一定权衡）。
  - AlignToolLLaMA-DPO vs AlignToolLLaMA-SFT：帮助性Pass Rate从27.3%跃升至49.8%（超越GPT-4的57.2%差距缩小），无害拒绝率98.7%，自主性100%。
  - 在ToolSword越狱攻击测试上，AlignToolLLaMA-DPO拒绝率87.1%，与GPT-4的89.0%相当；在不安全工具响应与自主性上全面超越GPT-4。
- **消融实验结论**：仅对SFT模型继续SFT偏好数据会导致各项指标下降；仅对ToolLLaMA做DPO无法习得无害性（拒绝率仍为0%），证明SFT是基础、DPO是关键。

## 相关工作脉络
1. **ToolBench/ToolLLaMA**（Qin et al., 2023）：聚焦工具调用帮助性，使用3k+工具与126k指令，但完全忽略无害性与自主性对齐。
2. **ToolSword**（Ye et al., 2024）：评估工具学习中三个阶段的有害性（指令、调用、响应），仅提供benchmark，无对齐方法。
3. **MetaTool**（Huang et al., 2023）：聚焦工具使用意识（是否调用工具及调用哪个），仅覆盖自主性维度，无无害性。
4. **Safety-LLaMA**（Bianchi et al., 2023）：通用指令的安全微调，未针对工具场景设计，迁移到工具使用时效果有限。
5. **Constitutional AI / RLHF**（Bai et al., 2022）：通用对话对齐方法，本文定位为将其扩展至工具使用场景，并细化为H2A三原则。
6. **DPO**（Rafailov et al., 2024）：偏好优化方法，本文创造性地将其用于工具对齐的多维权衡优化。

## 局限性与未来方向
- **单轮对话限制**：当前实验仅评估单轮交互，未涉及多轮对话中工具调用的连贯性与安全性维护，真实应用场景需处理长上下文与历史交互。
- **人类价值观复杂性**：H2A三原则是对人类价值观的简化抽象，实际场景中价值判断更具层次性与文化差异性，需更深理解。
- **模型规模限制**：当前基座为7B参数，自主性回答的有帮助性评分（3.86）仍低于GPT-4（4.73），可能受限于模型内在知识容量。
- **工具类型覆盖**：基于ToolBench的3k+工具，主要为API类工具，对代码执行、物理控制等工具类型的泛化有待验证。

## 研究启发与可借鉴点
1. **多维对齐原则的构建方法**：H2A将抽象价值操作化为可量化评估的三个正交维度，该思路可迁移至Agent安全对齐、机器人控制等其他交互场景。
2. **格式一致性数据工程技巧**：将ARTD/Alpaca的指令通过ChatGPT改写为与ToolBench一致的结构（背景+任务），避免模型学习数据集格式捷径而非真正能力，值得后续跨数据集融合工作借鉴。
3. **SFT打基础+DPO提上限的两阶段范式**：消融实验证明仅靠DPO无法从零习得安全能力，SFT先建立基础行为模式再经偏好优化提升，该训练策略对安全敏感应用具有普适价值。
4. **拒绝响应的三段落模板设计**：明确拒绝声明+危害分析+安全建议的拒绝结构，既保证无害性又维持帮助性，可作为有害内容处理的标准范式。
5. **交叉维度评估的测试集构建**：不仅评估单一维度指标，还测试在ToolSword越狱、MetaTool等外部基准上的泛化，证明多维对齐的实际可靠性，实验设计严谨。

## 关键术语表
- **H2A原则**：工具使用场景下LLM对齐的三个核心维度——帮助性(Helpfulness)、无害性(Harmlessness)、自主性(Autonomy)。
- **ToolAlign**：本文构建的工具对齐数据集，包含46k指令微调数据与10k偏好数据，覆盖H2A三维度。
- **AlignToolLLaMA-SFT**：基于ToolLLaMA在ToolAlign指令数据集上经SFT训练的模型，习得无害性与自主性基础。
- **AlignToolLLaMA-DPO**：在AlignToolLLaMA-SFT基础上经DPO训练的模型，进一步优化帮助性并提升整体对齐效果。
- **3R (Refusal Response Rate)**：模型对有害指令或不安全工具响应给出拒绝回答的比例。
- **DR2 (Direct Response Rate)**：模型在自主性测试中直接回答问题而不调用任何工具的比例。
- **DPO (Direct Preference Optimization)**：直接基于偏好数据优化语言模型的对齐方法，无需显式奖励模型。
- **ToolBench**：包含3k+真实工具与126k指令的工具学习基准，本文帮助性数据的主要来源。

## 可复现要素
- **数据集**：ToolAlign已开源，地址为https://github.com/zhiyuanc2001/ToolAlign
- **代码**：已开源，同上地址
- **模型权重**：AlignToolLLaMA-SFT与AlignToolLLaMA-DPO应可通过仓库获取（论文未明确声明权重开源地址，仅提到代码与数据集）
- **基座模型**：ToolLLaMA（需自行获取或复现）
- **关键超参**：SFT学习率5e-5、batch size 64、2 epochs；DPO学习率1e-6、batch size 8、1 epoch、β=0.05
- **硬件**：4×NVIDIA A100 40GB，bfloat16精度
- **API依赖**：ChatGPT (gpt-3.5-turbo)用于数据生成，GPT-4 (gpt-4-turbo)用于评估
