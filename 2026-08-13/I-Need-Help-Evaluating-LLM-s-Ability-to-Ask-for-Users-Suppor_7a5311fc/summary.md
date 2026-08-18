---
title: "I-Need-Help-Evaluating-LLM-s-Ability-to-Ask-for-Users-Suppor"
source: https://aclanthology.org/2024.emnlp-main.131.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:21:16"
field: "大语言模型交互与评估"
keywords: ["LLM主动求助", "Text-to-SQL", "人机协作评估", "Delta-Burden Curve", "BIRD", "不确定性校准", "大模型交互"]
innovations: ["提出AUDBC与DBC曲线量化性能提升与用户负担的权衡", "揭示外部执行反馈对LLM准确求助决策的关键作用", "解耦并分别评估LLM的求助识别能力与支持利用能力"]
benchmarks: ["BIRD", "Execution Accuracy"]
---

# 论文速读：I-Need-Help-Evaluating-LLM-s-Ability-to-Ask-for-Users-Suppor

## 一句话总结
本文以 Text-to-SQL 任务为例，提出了一套量化评估 LLM 主动请求用户支持能力的框架（DBC 曲线与 AUDBC 指标），发现多数当前 LLM 仅凭指令或自身生成结果难以准确判断求助时机，必须依赖外部反馈（如 SQL 执行结果）才能有效权衡性能提升与用户负担，相关工作与代码已开源。

## 研究问题与动机
- **LLM 错误归因视角缺失**：现有研究多将 LLM 幻觉/错误归因于模型能力不足，忽视了“信息不足时可主动求助”这一潜在路径。
- **性能提升与用户负担的固有 Trade-off**：用户介入需要提供支持（z），必然产生额外成本，亟需可量化的指标来刻画二者权衡关系。
- **校准研究的延伸空白**：既往 Well-calibratedness 研究聚焦于识别自身不确定性，尚未系统探索如何将不确定性转化为有效的求助决策与支持利用行为。
- **任务选择的合理性**：Text-to-SQL 存在天然歧义，且 BIRD 数据集自带人工标注外部知识作为支持源，适合作为实证案例。

## 核心贡献（创新点）
1. **提出 DBC 与 AUDBC 评估体系**，首次将“用户负担比例”与“平均性能增益”纳入同一归一化曲线进行量化。（与以往仅报告绝对准确率不同，本指标直接反映人机协作中的效率权衡。）
2. **设计并对比 DA / WA / EA 三种信息完整度递进的求助策略**，实证揭示了外部执行反馈（$\hat{r}$）对 LLM 决策质量的决定性作用。（区别于传统 Self-correction，本文聚焦于模型“何时该问”的元认知能力。）
3. **解耦评估 LLM 的两项关键能力**：通过 PR Curve 衡量“识别求助需求的能力”，通过 Flip Rate / FRC 衡量“利用支持修正结果的能力”。（揭示了二者可分离的现象，为精细化改进提供可操作维度。）
4. **给出黑盒模型的通用适配方案**：通过 Verbalized confidence prompt 绕过 log-probability 接口限制，使 Gemini、Claude 等闭源 API 也能实现超越随机基线的主动求助。

## 方法详解
- **形式化设定**：给定 LLM $f_\theta$、提示模板 $p(\cdot)$、用户指令 $x$ 与支持 $z$。LLM 输出求助信号 $\hat{a} \in [0,1]$，经阈值 $\tau$ 判定是否请求 $z$；带支持后的输出记为 $\hat{y}_{i,z}$。
- **核心指标**：
  - 用户负担 $B = N_{\text{ask}} / N$
  - 性能提升 $\Delta = \frac{1}{N} \sum_{i=1}^{N_{\text{ask}}} (h(y_i, \hat{y}_{i,z}) - h(y_i, \hat{y}_i))$
  - 绘制 $\Delta$-$B$ 曲线（DBC），计算归一化面积 AUDBC 作为综合评分。
- **三种信息输入策略**（共享提示模板 $p_{\text{ask}}$，仅 $w$ 不同）：
  - **Direct Ask (DA)**：$w = (db, x)$，仅依据数据库 Schema 与用户问题判断。
  - **Write then Ask (WA)**：$w = (db, x, \hat{y})$，先生成 SQL 再结合自身输出判断。
  - **Execute then Ask (EA)**：$w = (db, x, \hat{y}, \hat{r})$，引入 SQL 实际执行结果 $\hat{r}$ 作为外部反馈。
- **能力分解度量**：
  - 求助 Precision / Recall（$P_{\text{ask}}$, $R_{\text{ask}}$）及其 PR Curve。
  - Flip Rate $FR = \frac{1}{N_{\text{ask}}} \sum (h(y_i, \hat{y}_{i,z}) - h(y_i, \hat{y}_i))$，对应 FRC，专用于衡量支持利用效率。
- **黑盒适配**：将 Yes/No 的 softmax 置信度替换为 verbalized prompt，要求模型直接输出五位小数置信度 $0.abcde$。

## 实验与结果
- **数据集与评估**：BIRD 数据集，采用 Execution Accuracy (EX) 作为任务度量 $h(\cdot)$。
- **实验模型**：开源类 (WizardCoder-34B, Llama-3-70b-chat, DeepSeek-Coder-33B, Mixtral-8x22B)；闭源类 (GPT-3.5-turbo-0125, GPT-4-turbo-2024-04-09, GPT-4o-2024-05-13)；黑盒类 (Gemini-1.0-pro-001, Claude-3-haiku-20240307)。
- **主要结果 (Table 1 AUDBC)**：
  - **EA 方法整体最优**：GPT-4t (0.6641)、GPT-3.5 (0.6313)、Mixtral (0.6242)、DeepSeek-Coder (0.5848)。
  - **Llama-3-70b-chat 在无 $\hat{r}$ 时全面低于随机基线 (0.5000)**，印证了纯内部信息不足以支撑可靠求助决策。
  - GPT 系列与 Mixtral 在 WA/DA 上已能超越随机基线，但加入 $\hat{r}$ 后 AUDBC 仍有明显跃升。
- **支持强度对照 (Table 2)**：无支持时 EX 仅 0.17~0.31，Full support (B=1) 时 EX 提升至 0.28~0.51，证明外部知识确有增益，但 EA 能以更低 $B$ 逼近 Full support 的收益。
- **黑盒实验 (Table 3)**：Verbalized 方案整体略逊于真实 log-probs，但成功使 Gemini (0.5624) 与 Claude (0.6174) 超越随机基线，验证了工程可用性。
- **核心结论**：外部执行反馈是 LLM 准确决策的关键；“知不知”与“用不用”是两项可分离的能力。

## 相关工作脉络
1. **LLM 不确定性校准**（Kadavath et al., 2022; Xiao et al., 2022; Kuhn et al., 2023）：侧重预测置信度拟合；本文将其延伸至主动求助的行为决策层。
2. **Text-to-SQL 与外部知识增强**（Pourreza & Rafiei, 2023; Li et al., 2024 / BIRD）：多聚焦检索或 RAG 注入；本文聚焦于模型“何时触发检索/求助”的元认知门槛。
3. **幻觉缓解与 Self-Correction**（Rawte et al., 2023 综述及相关工作）：多关注生成后修正；本文前置到生成前的求助决策阶段，强调预防性交互。
4. **大模型评估指标体系**：传统评测以静态准确率为主；本文引入带权重的交互代价（B）与动态增益（Δ），属于 LLM-as-Agent / Human-AI Collaboration 评测的新视角。

## 局限性与未来方向
- **任务覆盖单一**：仅验证于 Text-to-SQL，需拓展至数学推理、代码补全、多轮 Agent 等更广泛的交互场景以检验泛化性。
- **支持类型局限**：目前仅使用人工标注的静态外部知识 $z$，未探索 API 调用、动态检索、多模态反馈或用户逐条澄清等多元支持形态。
- **对外部反馈的强依赖**：EA 效果最佳，但实际部署中即时执行或沙箱隔离可能受限；需研究无实时反馈条件下的轻量化替代信号。
- **未来方向**：扩展任务谱系；探索动态/多模态支持形式；针对 PR 与 FR 解耦现象分别设计微调或提示策略；优化黑盒模型的置信度校准稳定性。

## 研究启发与可借鉴点
1. **评估范式可直接迁移**：DBC/AUDBC 框架适用于任何“模型可请求外部干预”的场景（如 Agent 工具调用、在线评测、人机协同编程），为工程落地提供成本-收益量化依据。
2. **中间执行反馈的价值被低估**：EA 显著优于 WA/DA 的结果提示，在 Agent 反思循环中引入轻量级“预执行/仿真”步骤，可有效提升模型对失败样本的诊断精度。
3. **能力解耦分析具有诊断价值**：PR Curve 与 FRC 的分离说明“知道不懂”和“懂了会改”是独立能力；后续研究可按此分解针对性设计校准头（Calibration Head）或 Critic 模块。
4. **Verbalized 策略的工程友好性**：为无法暴露 log-probabilities 的商业 API 提供了即插即用的降级方案，便于在受合规限制的工业系统中快速原型验证主动求助 Agent。

## 关键术语表
- **Delta-Burden Curve (DBC)**：以用户负担比例 $B$ 为横轴、平均性能增益 $\Delta$ 为纵轴的权衡曲线，直观呈现不同阈值下的交互效率。
- **AUDBC**：DBC 曲线下方的归一化面积，值越高表示模型能以更少用户打扰换取更大性能提升。
- **Execute then Ask (EA)**：模型生成 SQL 并执行获取结果 $\hat{r}$ 后，结合 $\hat{r}$ 综合判断是否进一步求助的策略。
- **Flip Rate (FR)**：在模型请求支持的所有样本中，原本错误答案被支持成功修正的比例，专用于度量支持利用效率。
- **Well-calibratedness**：模型对自身预测不确定性的准确估计能力，是触发主动求助行为的认知基础。
- **Verbalized Confidence Score**：绕过 token log-probability 接口，通过自然语言指令让黑盒模型直接输出 $[0,1]$ 区间置信度分数的工程技巧。
- **BIRD Dataset**：大规模 Text-to-SQL 基准，包含数据库 Schema、自然语言问题、SQL 及人工标注的外部领域知识（支持源 $z$）。

## 可复现要素
- **数据集**：BIRD (Li et al., 2024)，官方公开，含人工标注支持知识。
- **代码**：已开源于 https://github.com/appier-research/i-need-help
- **关键超参**：求助阈值 $\tau \in [0,1]$（连续扫描绘制曲线）；评分函数 $s(\cdot)$ 为 Yes/No token 的 softmax；任务评估指标为 Execution Accuracy (EX)。
- **模型配置**：开源模型使用官方发布权重；闭源模型调用指定版本 API；temperature 与采样策略论文正文未明确给出（通常 eval 阶段取 temperature=0 或 low temperature，详见仓库代码）。
