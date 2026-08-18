---
title: "Towards Low-Resource Harmful Meme Detection with LMM Agents"
source: https://aclanthology.org/2024.emnlp-main.136.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:07:00"
field: "多模态内容安全检测"
keywords: ["低资源有害meme检测", "多模态大模型agent", "检索增强生成", "思维链推理", "无梯度学习"]
innovations: ["首次提出无梯度agent驱动的低资源有害meme检测框架LOREHM", "设计RSA与MIA双向增强策略实现向外检索与向内知识修正的结合", "通过错误轨迹驱动的洞察提取机制提升few-shot场景下的有害性判断能力"]
benchmarks: ["HarM", "FHM", "MAMI"]
---

# 论文速读：Towards Low-Resource Harmful Meme Detection with LMM Agents

## 一句话总结
本文提出了LOREHM框架，通过无梯度的agent驱动方式，结合向外检索相似meme标签与向内从错误判断中提取洞察的两步策略，实现了仅用少量标注样本即可高效检测网络有害meme的目标。

## 研究问题与动机
- **核心问题**：在低资源（few-shot）场景下，如何利用有限的标注数据对新兴有害meme进行有效检测，解决传统数据驱动方法难以应对快速演变的网络meme的问题。
- **现有方法不足**：
  - 传统DNN方法依赖大量标注数据，难以适应突发事件的快速变化
  - 现有few-shot方法（如LoRA微调）仍需更新模型参数，面临过拟合风险
  - In-context Learning（ICL）虽无需训练但推理开销大，且单纯依赖内部知识难以捕捉meme中隐含的有害信息

## 核心贡献（创新点）
- **首次从无梯度角度解决低资源有害meme检测问题**：通过让LMM作为agent而非训练模型的方式，避免参数更新带来的过拟合风险，区别于Mod-HATE等需要微调的方法。
- **提出双向增强的agent驱动框架**：向外利用检索的相似meme标签作为先验信号，向内通过知识修正行为提取通用洞察，两种策略互补而非增量叠加，区别于仅使用检索增强或仅使用ICL的基线方法。
- **构建可读的解释性推理过程**：通过Chain-of-Thought生成可解释的思维链，帮助人工审核员理解模型判断依据，弥补现有方法缺乏可解释性的缺陷。

## 方法详解
**框架名称：LOREHM（Low-Resource Harmful Meme Detection）**

核心设计为"向外看"与"向内看"的双策略：

**1. Relative Sample Augmentation (RSA) - 向外分析**
- 使用CLIP编码器（ViT-L/14@336px）分别提取图像和文本特征，按固定比例融合：$Emb = \alpha \cdot VE(\mathcal{T}) + \beta \cdot TE(\mathcal{T})$，其中$\alpha=0.2, \beta=0.8$
- 计算目标meme与参考集中所有meme的余弦相似度，选取Top-K个最相似meme构成检索集$\mathcal{H}$
- 采用投票机制获取初步预测$\mathcal{P}$：若检索集中有害meme数量超过K/2，则预测为有害

**2. Meme Insight Augmentation (MIA) - 向内分析**
- **经验收集**：对参考集中的每个meme进行零样本CoT推理，记录推理轨迹
- 筛选错误轨迹构成自反思集$R_{set}$，聚焦于模型难以判断的挑战性样本
- **洞察提取**：迭代处理错误轨迹，LMM基于当前洞察集执行四种操作：ADD（新增洞察）、DOWNVOTE（降低重要性）、UPVOTE（提升重要性）、EDIT（修改洞察内容）
- 最终形成容量上限为10的洞察集$\mathcal{E}_n$

**3. 推理阶段**
- 将RSA的初步预测$\mathcal{P}$作为先验，结合洞察集$\mathcal{E}_n$，通过最终CoT提示完成有害性判断
- 参考集大小$N=50$，检索集合大小$K=5$

## 实验与结果
**数据集**：HarM（COVID-19相关meme）、FHM（Facebook仇恨言论挑战）、MAMI（针对女性的冒犯性meme）

**主要结果**（50-shot设置）：

| 模型 | HarM Acc/F1 | FHM Acc/F1 | MAMI Acc/F1 |
|------|-------------|------------|-------------|
| GPT-4o (ICL) | 71.75/70.23 | 66.60/65.74 | 80.80/80.52 |
| LLaVA-34B (ICL) | 67.80/62.60 | 63.80/63.74 | 74.60/74.52 |
| **LOREHM (GPT-4o)** | **74.57/72.98** | **70.20/70.14** | **83.00/82.98** |
| **LOREHM (LLaVA-34B)** | **73.73/70.86** | **65.60/65.59** | **75.40/75.28** |

**关键结论**：
- LOREHM基于GPT-4o相比GPT-4o的ICL版本，在HarM、FHM、MAMI上的Macro-F1分别提升2.75%、4.40%、2.46%
- 消融实验证明RSA与MIA策略互补，联合使用效果最优
- LOREHM范式与LMM backbone选择正交，可迁移至新发布的更强模型

## 相关工作脉络
- **PromptHate / MR.HARM / Pro-Cap**：基于提示学习的有害meme检测方法，需要充分的数据驱动训练，而LOREHM完全无梯度
- **Mod-HATE**：基于LoRA微调的few-shot方法，仍需参数更新；LOREHM避免任何权重更新
- **LLaVA / GPT-4o**：基础多模态大模型；本文将其作为agent而非直接分类器使用，通过检索增强与自我反思机制增强其推理能力
- **Flamingo / OpenFlamingo**：预训练多模态模型；本文不依赖特定预训练策略，强调推理阶段的agent行为设计
- **Retrieval-Augmented Generation (RAG)**：传统RAG检索文本段落；本文扩展到多模态检索并结合投票机制与知识修正

## 局限性与未来方向
- **数据集边界有限**：目前仅覆盖仇恨言论和厌女内容，可扩展至冒犯性、讽刺、混合语言等更广泛的有害内容类型
- **仅聚焦few-shot**：尚未探索zero-shot设置及低资源领域/语言场景
- **洞察质量评估困难**：提取的洞察缺乏定量评估手段，主要依赖定性分析
- **LMM内生问题**：仍存在幻觉、固有偏见和泛化能力有限等问题，需进一步缓解
- **代码未公开**：论文提交时未包含源代码，仅承诺接受后开源

## 研究启发与可借鉴点
- **无梯度agent范式**：将大模型作为推理agent而非训练对象，适用于标注数据稀缺且需要快速适应新场景的任务
- **双向信息整合策略**：外部检索的先验信号与内部知识修正相结合的设计思路，可迁移至其他多模态理解任务
- **错误轨迹驱动的知识积累**：通过筛选模型错误样本进行针对性反思，而非均匀处理所有样本，提升学习效率
- **可解释推理输出**：CoT生成的自然语言推理过程可直接辅助人工审核，对内容安全领域具有实用价值
- **CLIP多模态融合检索**：固定的视觉-文本特征融合比例策略简单有效，可作为多模态检索的基线方案

## 关键术语表
- **LMM (Large Multimodal Model)**：大型多模态模型，如LLaVA-34B、GPT-4o，能够同时处理文本和图像输入
- **LOREHM**：本文提出的无梯度agent驱动框架，用于低资源有害meme检测
- **RSA (Relative Sample Augmentation)**：向外增强策略，通过检索相似meme及其标签为推理提供先验信号
- **MIA (Meme Insight Augmentation)**：向内增强策略，通过分析模型错误判断提取对有害性模式的通用洞察
- **In-context Learning (ICL)**：通过在提示中提供示例让模型进行 few-shot 推理，无需更新模型参数
- **Chain-of-Thought (CoT)**：思维链提示技术，引导模型生成逐步推理的自然语言解释
- **HarM / FHM / MAMI**：三个用于评估的公开meme数据集，分别涉及COVID-19、仇恨言论和性别歧视主题
- **Gradient-free**：无梯度方法，指不通过反向传播更新模型权重的推理策略

## 可复现要素
- **数据集**：HarM、FHM、MAMI均为公开数据集
- **代码/权重**：论文声明接受后开源；使用LLaVA-34B开源权重，GPT-4o通过API访问
- **关键超参**：参考集大小$N=50$，检索Top-K=$5$，视觉-文本融合比例$\alpha=0.2, \beta=0.8$，CLIP版本为ViT-L/14@336px，温度参数设为0
- **实验环境**：OpenAI API + 4块NVIDIA A40 48GB GPU
