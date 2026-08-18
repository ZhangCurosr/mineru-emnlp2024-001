---
title: "Towards-Low-Resource-Harmful-Meme-Detection-with-LMM-Agents"
source: https://aclanthology.org/2024.emnlp-main.136.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:34:01"
field: "多模态内容安全与有害信息检测"
keywords: ["low-resource harmful meme detection", "LMM Agent", "retrieval-augmented generation", "in-context learning", "gradient-free learning", "multimodal reasoning"]
innovations: ["提出无梯度双向增强框架LOREHM，结合向外检索标签先验与向内错误反思提取洞察", "首次将LMM作为Agent用于低资源有害模因检测而非仅做few-shot分类", "设计结构化洞察集管理机制(ADD/UPVOTE/DOWNVOTE/EDIT)实现知识累积式自我改进"]
benchmarks: ["HarM", "FHM (Hateful Memes Challenge)", "MAMI"]
---

# 论文速读：Towards-Low-Resource-Harmful-Meme-Detection-with-LMM-Agents

## 一句话总结
本文提出 LOREHM，一种基于 LMM（大语言视觉模型）Agent 的无梯度、少样本有害模因检测方法，通过"向外检索相似模因标签"与"向内从失败推理中提取洞察"两种策略，在仅 50 条标注数据的情况下，于 HarM、FHM、MAMI 三个数据集上均超越现有最强基线。

## 研究问题与动机
1. **低资源场景下的有害模因检测难题**：互联网模因演化迅速，新兴事件频发，难以快速积累大量标注数据；传统数据驱动方法在此类低资源设定下性能骤降。
2. **现有方法的不足**：既有方法依赖大规模标注数据训练（如微调或 LoRA），无法适应快速变化的模因趋势；而直接 few-shot ICL 虽能缓解，但每次推理需拼接全部演示样本，计算开销巨大。
3. **隐性有害信息难以捕捉**：模因的有害性往往隐藏于图文交互的微妙语境中，并非表层文本或图像可直接表达，需要深层的多模态推理能力。
4. **无梯度学习的必要性**：作者认为应避免通过梯度更新参数来适应新模因，而是模仿人类"积累经验但不修改模型权重"的方式，降低过拟合风险。

## 核心贡献（创新点）
1. **首次从无梯度视角利用 LMM Agent 解决低资源有害模因检测问题**：区别于前作对 LLM 推理知识的蒸馏，本文聚焦于让 LMM 作为自主 Agent 在少样本下自主学习。
2. **提出双向增强策略：向外检索（RSA）与向内反思（MIA）**：前者通过检索相似模因的显式标签提供先验信号，后者从 LMM 的零样本错误推理中提取可泛化的洞察知识，二者互补而非叠加。
3. **在三个基准数据集上实现最强 few-shot 性能**：基于 GPT-4o 的 LOREHM 在 HarM/FHM/MAMI 上分别提升 Macro-F1 2.75%/4.40%/2.46%，且方法论对开源/闭源 LMM 均具有正交可移植性。

## 方法详解
**框架整体流程**：给定参考集 $S_{\text{ref}}$（N=50 条标注样本）和测试集 $S_{\text{test}}$，将有害模因检测转换为自然语言生成任务。

**① Relative Sample Augmentation（向外检索）**
- **特征融合**：对每个模因 $M=\{\mathcal{I}, \mathcal{T}\}$，使用冻结的 CLIP ViT-L/14 视觉编码器和文本编码器生成多模态嵌入：$Emb = 0.2 \cdot VE(\mathcal{I}) + 0.8 \cdot TE(\mathcal{T})$。
- **相似度检索**：计算测试模因与参考集中每个模因的余弦相似度，取 Top-K（K=5）个最相似模因作为检索集 $\mathcal{H}$。
- **投票机制**：对检索集的 K 个标注标签进行多数投票，生成初步预测 $\mathcal{P} \in \{\text{harmful, harmless}\}$，作为后续 LMM 推理的先验信号。

**② Meme Insight Augmentation（向内反思）**
- **经验收集**：对参考集所有样本进行零样本 CoT 推理，记录 LMM Agent 的推理轨迹 traj。
- **错误轨迹筛选**：比较 traj 输出与真实标签，仅保留错误样本构成自反思集 $R_{\text{set}}$，避免对已正确样本重复浪费算力。
- **洞察提取**：迭代处理每个错误轨迹，通过 reflection prompt 让 LMM 执行四种操作之一作用于洞察集 $\mathcal{E}$：ADD（新增通用洞察）、UPVOTE（赞同既有洞察）、DOWNVOTE（降低既有洞察重要性）、EDIT（修改既有洞察）。洞察集容量上限为 10 条，最终得到 $\mathcal{E}_n$。

**③ 推理阶段**
- 将初步预测 $\mathcal{P}$ 与洞察集 $\mathcal{E}_n$ 共同作为 prior，输入给 LMM Agent 进行最终判断：$\mathrm{LMM}(X_{\text{CoT}}, \mathcal{I}_{\text{test}}, \mathcal{T}_{\text{test}}, \mathcal{P}, \mathcal{E}_n)$。

**关键超参**：$N=50$（参考集大小），$K=5$（检索数量），$\alpha=0.2, \beta=0.8$（嵌入融合权重），温度=0（贪心解码）。

## 实验与结果
**数据集**：HarM（COVID-19相关，合并两级有害为一类）、FHM（Facebook仇恨言论挑战）、MAMI（厌女内容识别）；测试集规模均为各 250-500 条。

**评估指标**：Accuracy 与 Macro-F1（主要报告指标）。

**主要结果**（Table 1）：
| 数据集 | LOREHM(GPT-4o) Acc | LOREHM(GPT-4o) Macro-F1 | 相对 GPT-4o ICL 提升(F1) |
|--------|-------------------|------------------------|------------------------|
| HarM   | 74.57%            | 72.98%                 | +2.75%                 |
| FHM    | 70.20%            | 70.14%                 | +4.40%                 |
| MAMI   | 83.00%            | 82.98%                 | +2.46%                 |

**消融实验**（Table 2）：
- 0-shot Prompt 性能最差，证明任务需要额外知识辅助。
- 50-shot ICL 可有效提升，但 RSA 和 MIA 单独使用已可超越或持平 ICL 基线。
- 两者结合（LoREHM）显著优于任一单独策略，说明 RSA 与 MIA 存在互补性。
- LLaVA-34B 与 GPT-4o 两种 backbone 均验证了方法的有效性。

**超参分析**（Figure 4）：
- 随 K 增大，不同 backbone 间差距缩小；随 N 增大，性能趋于饱和甚至下降，说明简单堆叠标注样本效果有限。

## 相关工作脉络
1. **PromptHate (Cao et al., 2022)**：将图文拼接为 prompt 用于 masked LM 预测；本文采用更复杂的 Agent 推理框架而非简单 prompt 拼接。
2. **MR.HARM (Lin et al., 2023a)**：从 LLM 蒸馏多模态推理知识用于检测；本文不使用蒸馏，而是让 LMM 在推理时实时进行知识修订。
3. **Mod-HATE (Cao et al., 2024)**：基于 LoRA 微调的低资源方案；本文彻底避免参数更新，走纯无梯度路线。
4. **LLaVA / GPT-4o**：作为本文的 backbone Agent；本文的贡献在于 agent 范式而非模型本身，可与更强 LMM 无缝替换。
5. **Reflexion / ExpeL**：需要环境反馈的 self-improvement 方法；本文因检测任务缺乏实时反馈环境而设计了适配的二分类反思机制。
6. **Hateful Memes Challenge / Multimodal Meme Benchmarks**：催生了本领域的基础数据集；本文聚焦于低资源 few-shot 设定下的泛化问题，填补了前沿空白。

## 局限性与未来方向
1. **当前仅覆盖 few-shot 设置**：尚未探索 zero-shot 极端低资源场景，未来将扩展。
2. **洞察质量难以量化评估**：提取的 $\mathcal{E}_n$ 偏向定性，缺乏系统性的可解释性度量方法。
3. **LMM 固有偏差与幻觉**：文中承认 LMM 存在 inherent bias 和 hallucination 问题，可能影响检测公正性（如宗教主题模因出现系统性误判）。
4. **任务边界较窄**：仅关注 harmful meme detection，未来可扩展至 offensive/sarcastic/code-mixed 等更广泛的模因理解任务。
5. **多语言与低资源社交语境**：尚未在语言多样性或跨平台社交语境中验证泛化能力。

## 研究启发与可借鉴点
1. **"向外+向内"的双向增强范式**：检索辅助信号与知识反思机制的解耦设计，为其他低资源多模态分类任务提供了可迁移的方法论框架。
2. **错误轨迹优先的学习策略**：仅从 LMM 的零样本错误样本中提取洞察，避免了重复正确样本的资源浪费，这一策略可推广至其他 self-improvement agent 设计。
3. **Vote-based prior 整合方式**：将检索结果的标签投票结果作为显式先验而非直接输入原文，兼顾了外部知识与模型自主判断的平衡，值得在其他检索增强场景复用。
4. **无梯度低资源检测思路**：完全避免参数更新、仅通过 prompt 编排实现知识积累，为资源受限场景下的模型部署提供了新路径。
5. **洞察集管理（ADD/UPVOTE/DOWNVOTE/EDIT）**：这一结构化知识维护机制可被借鉴到其他需要长期记忆积累的 agent 系统中。

## 关键术语表
**LOREHM**：本文提出的面向低资源有害模因检测的 LMM Agent 框架，全称呼应 Low-Resource Harmful Meme detection。

**RSA (Relative Sample Augmentation)**：向外增强策略，通过检索参考集中最相似的 K 个模因并以标签投票生成初步预测 $\mathcal{P}$，为 LMM 提供显式先验。

**MIA (Meme Insight Augmentation)**：向内增强策略，从 LMM 零样本推理的错误轨迹中提取可泛化的洞察知识，构建动态维护的洞察集 $\mathcal{E}_n$。

**LMM Agent**：将大语言视觉模型（如 LLaVA、GPT-4o）视为具备推理、检索和自我反思能力的自主代理，而非传统分类器。

**CoT (Chain-of-Thought)**：思维链推理提示策略，引导 LMM 在给出最终判断前逐步展开分析过程。

**Gradient-free**：本文核心特征，全程不进行任何模型权重更新，完全依赖 prompt 编排与知识检索实现少样本泛化。

**Self-reflect Set ($R_{\text{set}}$)**：从参考集中筛选出的 LMM 零样本推理错误的轨迹集合，作为知识修订的经验池。

## 可复现要素
- **数据集**：HarM、FHM、MAMI 均为公开数据集；FHM 需遵守 Facebook 使用协议。
- **代码**：论文声明将在接收后公开（"commit to making the code publicly available upon acceptance"），当前提交未附源码。
- **模型权重**：使用 LLaVA-34B（开源）和 GPT-4o（API 调用）；具体版本为 `llava-v1.6-34b` 和 `gpt-4o-2024-05-13`。
- **关键超参**：N=50（参考集大小），K=5（检索数量），$\alpha=0.2$、$\beta=0.8$（多模态嵌入融合权重），温度=0，洞察集容量=10。
- **编码器**：冻结 CLIP ViT-L/14@336px。
- **硬件**：4× NVIDIA A40 48GB GPU + OpenAI API。
- **随机性控制**：每种 N=50 设置下生成 5 个不同 random seed 的参考集，报告平均结果以降低 few-shot 方差。
