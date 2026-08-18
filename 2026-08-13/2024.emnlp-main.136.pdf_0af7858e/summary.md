---
title: "Towards Low-Resource Harmful Meme Detection with LMM Agents"
source: https://aclanthology.org/2024.emnlp-main.136.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:06:35"
field: "多模态有害内容检测"
keywords: ["有害梗图检测", "低资源学习", "多模态大模型", "代理", "检索增强生成", "思维链", "few-shot"]
innovations: ["无梯度低资源代理框架LOREHM，首次从gradient-free视角解决有害梗图检测", "外向外向检索投票与内向外向知识修订双轨互补策略", "错误驱动的动态洞察集管理（ADD/DOWNVOTE/UPVOTE/EDIT四操作）"]
benchmarks: ["HarM", "FHM (Hateful Memes Challenge)", "MAMI"]
---

# 论文速读：Towards Low-Resource Harmful Meme Detection with LMM Agents

## 一句话总结
论文提出 LOREHM，一个基于 LMM 代理的**无梯度**框架，通过外向外向检索相似梗图提供标签先验、内向外向从错误推理中修订知识提取洞察，在仅有 50 个标注样本的低资源场景下实现了优于 SOTA 方法的有害梗图检测性能。

## 研究问题与动机
- **低资源瓶颈**：现有数据驱动方法依赖大量标注样本，而新兴有害梗图（如突发事件相关）演化迅速，难以及时获取足量标注。
- **梯度更新易过拟合**：few-shot 下微调参数（如 LoRA）容易过拟合稀疏标注，无法应对梗图风格的持续演变。
- **隐性有害性难以捕捉**：梗图的有害信号常隐式存在于图文多模态交互中，表面内容可能无害，需要推理与常识补充（如反疫苗梗用幽默包装虚假信息）。
- **人类学习范式的启发**：人类检查者并非通过修改权重学习，而是积累"经验教训"——本文借此动机设计类比人类知识修订的代理机制。

## 核心贡献（创新点）
1. **首次从无梯度视角解决低资源有害梗图检测**：与已有 LoRA 微调方法（Mod-HATE）的本质区别在于完全不更新模型参数，避免过拟合。
2. **提出外向检索 + 内向知识修订的双路径代理框架**：与纯 ICL 方法的本质区别在于将检索结果的显式标签作为投票先验，并从错误轨迹中动态提炼通用洞察，而非简单拼接 few-shot 示例。
3. **设计可复用的"知识修订"机制（Insight Set 管理）**：通过 ADD/DOWNVOTE/UPVOTE/EDIT 四操作迭代维护洞察集，与 Reflexion/ExpeL 等需环境反馈的方法不同，专为二分类任务中"错误即揭示答案"的场景定制。

## 方法详解
- **整体架构**：LMM 代理分为三阶段——①外向外向 Relative Sample Augmentation (RSA) ②内向外向 Meme Insight Augmentation (MIA) ③融合推理。参考集 $S_{ref}$ 含 N=50 个标注样本，检索 Top-K 取 K=5。
- **外向外向（RSA）**：用冻结 CLIP（ViT-L/14@336px）分别编码图像 $\mathcal{I}$ 和文本 $\mathcal{T}$，融合为多模态嵌入 $Emb = 0.2 \cdot VE(\mathcal{I}) + 0.8 \cdot TE(\mathcal{T})$；计算测试梗与参考集中的余弦相似度，取 Top-K 检索集 $\mathcal{H}$；通过多数投票得到初步预测 $\mathcal{P}$（harmful 票数 > K/2）。
- **内向外向（MIA）**：对每个 $M_{ref} \in S_{ref}$ 执行 zero-shot CoT 推理，将推理结果与 ground truth 对比，收集错误轨迹构成自反思集 $R_{set}$；迭代地将每条错误轨迹输入 LMM，配合当前洞察集 $\mathcal{E}_{i-1}$ 执行 ADD/DOWNVOTE/UPVOTE/EDIT 操作更新 $\mathcal{E}_i$，洞察集容量上限设为 10。
- **最终推理**：将 $\mathcal{P}$ 作为 prior、$\mathcal{E}_n$ 作为知识指导，输入最终 CoT prompt：*"A classifier that can identify common features among multiple memes has labeled this meme as {P}. Please review the classifier's judgment carefully..."*，由 LMM 输出最终判断。

## 实验与结果
- **数据集**：HarM（COVID 相关，124 有害/230 无害）、FHM（仇恨言论，250/250）、MAMI（厌女内容，500/500）；测试集规模见原文 Table 3。
- **主要结果（Macro-F1，%）**：

| 模型 | HarM | FHM | MAMI |
|------|------|-----|------|
| GPT-4o (50-shot ICL) | 70.23 | 65.74 | 80.52 |
| LLaVA-34B (50-shot ICL) | 62.60 | 63.74 | 74.52 |
| **LoREHM (GPT-4o)** | **72.98** | **70.14** | **82.98** |
| **LoREHM (LLaVA-34B)** | **70.86** | **65.59** | **75.28** |

- **最强提升**：LoREHM(GPT-4o) 较 GPT-4o ICL 基线在 HarM 上 Macro-F1 提升 +2.75%，FHM 上 +4.40%，MAMI 上 +2.46%。
- **消融结论**：RSA 与 MIA 各自独立均有效且互补；N 增大后性能趋于饱和甚至下降，说明单纯增加样本数不够，需要推理策略。

## 相关工作脉络
- **PromptHate / Pro-Cap**：基于 prompt 的文本+图像描述拼接方法，依赖少量但固定的 prompt 模板；本文不依赖文本描述拼接，而是通过检索提供显式标签先验。
- **MR.HARM**：从 LLM 蒸馏多模态推理知识用于检测；本文不蒸馏，而是让 LMM 在线进行知识修订并动态维护洞察。
- **Mod-HATE**：基于 LoRA 微调的 few-shot 方法，仍属梯度更新范式；本文完全无梯度，避免参数过拟合。
- **Reflexion / ExpeL**：需环境提供实时反馈的自改进 agent 方法；本文针对二分类任务中"错误轨迹即含正确答案"的特性设计了专用洞察集合管理机制，不依赖外部环境反馈。
- **Atlas / In-context RAG**：检索增强生成主要用于文本任务；本文将其扩展到视觉-语言检索，并将检索结果转化为投票信号而非直接作为输入示例。

## 局限性与未来方向
- 仅评测了三种英语梗图数据集（HarM/FHM/MAMI），未覆盖冒犯性、讽刺、code-mixed 等其他有害内容类型。
- 聚焦 few-shot（N=50），zero-shot 场景尚未探索。
- 低资源/低资源语言（如中文社交媒体的梗图）未涉及。
- 提取的洞察质量目前只能定性评估，缺乏系统性可解释性度量。
- LMM 自身存在幻觉、内在偏差和泛化局限，未来需进一步缓解。

## 研究启发与可借鉴点
1. **"无梯度低资源学习"范式**：在样本极少且数据分布易变的场景下，将 LMM 视为可推理的代理而非可训练的参数容器，值得迁移到其他 few-shot 视觉-语言分类任务（如虚假新闻检测、极端内容识别）。
2. **Insight Set 四操作机制**（ADD/DOWNVOTE/UPVOTE/EDIT）可抽象为通用"错误驱动知识归纳"模块，适用于任何需要从少量失败案例中提炼通用规则的代理系统。
3. **RSA 投票先验设计**避免了将检索样本直接作为 ICL 示例带来的冗长 prompt 开销，同时与 MIA 形成正交互补——这种"显式标签先验 + 隐性知识修订"的双轨设计值得在开放域多模态推理中复用。
4. **CLIP 多模态嵌入融合比**（α=0.2, β=0.8）提示在梗图场景中文本权重更高，可在同类任务中做相似消融验证。

## 关键术语表
- **LOREHM**：Low-Resource Harmful Meme detection by LMM agents，本文提出的无梯度代理框架。
- **Relative Sample Augmentation (RSA)**：外向外向增强策略，通过多模态检索找到最相似的 K 个标注梗图，以投票方式提供初步有害性先验。
- **Meme Insight Augmentation (MIA)**：内向外向增强策略，从 LMM 零样本推理的错误轨迹中迭代提取通用有害性洞察。
- **Insight Set**：动态维护的洞察集合（容量上限 10），通过 ADD/DOWNVOTE/UPVOTE/EDIT 四操作持续进化。
- **Chain-of-Thought (CoT)**：思维链提示，引导 LMM 逐步推理后再给出有害/无害判定。
- **Self-Reflect Set ($R_{set}$)**：由零样本 CoT 推理中预测与真实标签不一致的轨迹组成的经验池。
- **Few-shot ICL**：在 prompt 中直接附带 N 个标注示例，利用 LLM/LMM 的上下文学习能力进行推理。
- **Gradient-free**：不更新任何模型权重的学习方式，仅依赖推理与检索完成知识利用。

## 可复现要素
- **数据集**：HarM、FHM、MAMI 均为公开数据集（Hateful Memes Challenge / SemEval 等），可公开获取。
- **代码**：论文未随提交公开代码，作者承诺论文接收后开源（Appendix C 声明）。
- **模型版本**：LLaVA-34B 使用 `llava-v1.6-34b`；GPT-4o 使用 `gpt-4o-2024-05-13`。
- **关键超参**：α=0.2，β=0.8；K=5（检索数）；N=50（参考集大小）；洞察集容量=10；temperature=0（greedy decoding）；CLIP 版本 `ViT-L/14@336px`。
- **实验环境**：OpenAI API + 4× NVIDIA A40 48GiB GPU。
