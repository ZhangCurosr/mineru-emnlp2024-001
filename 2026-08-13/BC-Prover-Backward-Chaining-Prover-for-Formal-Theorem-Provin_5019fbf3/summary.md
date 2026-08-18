---
title: "BC-Prover-Backward-Chaining-Prover-for-Formal-Theorem-Provin"
source: https://aclanthology.org/2024.emnlp-main.180.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:27:17"
field: "形式化定理证明"
keywords: ["interactive theorem proving", "backward chaining", "formal verification", "large language models", "miniF2F", "Lean"]
innovations: ["首次将后向链式策略系统引入Lean ITP，通过伪步骤引导LLM从目标逆向推导子目标", "提出伪步骤+步骤规划的双层引导机制，桥接非形式化与形式化证明的语义鸿沟", "设计可与已有微调prover无缝协作的插件式框架，显著提升搜索效率"]
benchmarks: ["miniF2F-valid", "miniF2F-test"]
---

# 论文速读：BC-Prover: Backward Chaining Prover for Formal Theorem Proving

## 一句话总结
本文提出 BC-Prover，一个基于**后向链式推理（backward chaining）**和**步骤规划（step planning）**的交互式定理证明框架；该框架在 Lean 中通过伪步骤引导 LLM 从目标出发递归分解子目标，在 miniF2F 基准上显著超越 SOTA 基线方法。

## 研究问题与动机
- 现有 ITP（Interactive Theorem Proving）方法主要依赖前向链式策略（从已知假设出发逐条应用 tactics），在庞大搜索空间中盲目探索，效率低下。
- 直接利用非形式化证明（informal proof）作为上下文示例存在"形式化-非形式化"语义鸿沟，且非形式化证明通常省略中间推理步骤，容易误导 LLM。
- 以往工作（无论是微调还是提示方法）均忽视了**后向链式策略**——即从目标出发逆向推导所需引理/假设，难以有效缩小搜索空间。
- 非形式化证明与形式化证明之间存在推理颗粒度差异，需要一种"中间层"（伪步骤）来桥接二者。

## 核心贡献（创新点）
1. **提出 BC-Prover 框架**，首次将后向链式策略系统性地引入 Lean ITP 任务，通过伪步骤引导 LLM 递归发现可证子目标，将全局搜索转化为目标驱动的细粒度探索。
2. **伪步骤生成机制**：先对输入定理生成非形式化证明，再将其细化为结构化的伪步骤（pseudo steps），作为后续前向/后向推理的共同参考，弥补非形式化证明的推理跳跃。
3. **步骤规划（Step Planning）模块**：对当前证明状态进行注释描述，并基于伪步骤生成下一步战术规划，进一步约束搜索空间，减少无效探索。
4. **与微调模型的兼容协作**：BC-Prover 可无缝接入 ReProver、LLMStep 等已有微调 prover，以插件式方式显著提升其 Pass@1（如 BC-LLMStep 在 miniF2F-valid 达 35.2%）。
5. **系统性实验验证**：在 miniF2F 基准上，BC-Prover（gpt-4-turbo，search-k=16）在 valid/test 分别达 29.5%/30.7%，优于多类 SOTA 基线，且累计解决率（BC-Prover*）达 38.9%/36.9%。

## 方法详解
BC-Prover 分为两个阶段：**前置准备**（伪步骤生成）和**迭代证明构造**。

### 前置阶段：伪步骤生成
给定定理的形式化陈述 $X_t$ 和非形式化陈述 $X_h$，通过 LLM（$\mathcal{M}$）先后生成：
- **非形式化证明** $P_h$：$(X_t, X_h) \rightarrow P_h$
- **伪步骤** $P_s$：$(X_t, X_h, P_h) \rightarrow P_s$

伪步骤是对非形式化证明的细粒度展开，补充了非形式化证明中省略的显式证明步骤。

### 迭代证明构造（每轮迭代）
1. **后向链式（Backward Chaining）**：基于伪步骤 $P_s$ 和当前状态 $S_i$，LLM 生成 $d=8$ 个子目标-战术对 $(g^{sub}, t^{sub})$；经 Lean 验证后，将有效假设 $h^{goal}$ 加入状态：$S_i^{\theta} = [S_i, \mathbf{H}_g]$。
2. **步骤规划（Step Planning）**：对 $S_i^{\theta}$ 生成自然语言描述 $D$，再结合 $P_s$ 生成下一步规划 $N$（指示应使用哪些 tactics）。
3. **引理检索与重排序**：使用 LeanDojo 的 premise retriever 从 mathlib 检索候选引理 $\mathbf{L}_r$，再由 LLM 重排序 agent 选出 $n=5$ 个最相关的引理及其使用计划 $\mathbf{L}$。
4. **前向战术生成**：将 $S_i^{*} = [S_i^{\theta}, N, \mathbf{L}]$ 输入 LLM，生成 $k=16$ 条候选 tactic。
5. **搜索策略**：由于 API 调用无法获取 log probability，采用基于**状态复杂度（token 数最小）**的搜索策略替代 best-first search，选择使下一状态最简单的 tactic 执行。

核心公式：
- 伪步骤生成：$\mathcal{M}: (X_t, X_h) \rightarrow P_h$；$\mathcal{M}: (X_t, X_h, P_h) \rightarrow P_s$
- 后向链式：$\mathcal{M}: (P_s, S_i) \rightarrow \mathbf{H}$；$\mathbf{H}_g = \text{Lean}(\mathbf{H}, S_i)$；$S_i^{\theta} = [S_i, \mathbf{H}_g]$
- 步骤规划：$\mathcal{M}: S_i^{\theta} \rightarrow D$；$\mathcal{M}: (D, S_i^{\theta}, P_s) \rightarrow N$；$\mathcal{M}: (\mathbf{L}_r, S_i^{\theta}, P_s) \rightarrow \mathbf{L}$
- 前向生成：$S_i^{*} = [S_i^{\theta}, N, \mathbf{L}]$；$\mathcal{M}: S_i^{*} \rightarrow \{t_i^0, ..., t_i^k\}$

## 实验与结果
- **数据集**：miniF2F benchmark（Zheng et al., 2022），共 488 道题（AIME、AMC、IMO 等竞赛题及本科数学），分为 miniF2F-valid（255题）和 miniF2F-test（244题）。
- **评估指标**：Pass@1（单次尝试，最多 100 步迭代内完成证明）。
- **核心结果（Table 1）**：
  - BC-Prover（search-k=16）：valid = **29.5%**，test = **30.7%**，优于 GPT-4（22.9%/23.4%）约 6.5–7.3%。
  - BC-Prover*（累计解决率）：valid = **38.9%**（95/255），test = **36.9%**（90/244）。
  - BC-LLMStep：valid = **35.2%**，test = **32.0%**，相比 LLMStep 提升约 9%。
  - BC-ReProver：valid = **32.0%**，test = **31.6%**，相比 ReProver 提升约 8.2%。
- **消融实验（Table 2）**：移除 BC 导致 valid 下降 4.1%、test 下降 5.7%；移除 SP 导致 valid 下降 2.1%、test 下降 2.4%，两个模块均有效。
- **搜索效率分析（Table 3）**：BC-Prover 平均证明步数（1.80）和平均迭代次数（2.30）均显著低于 GPT-4（2.07 / 3.45），表明后向链式有效缩小了搜索路径。
- **计算成本（Table 4）**：BC-Prover 累计消耗约 45.13 美元（2.1M input tokens + 0.7M output tokens）。

## 相关工作脉络
1. **Polu & Sutskever (2020) / Expert Iteration (Polu et al., 2023)**：前向链式语言模型定理证明的开创性工作，通过自博弈迭代训练；本文采用提示范式而非微调，且引入后向链式弥补纯前向搜索的局限。
2. **ReProver (Yang et al., 2023) / LLMStep (Welleck & Saha, 2023)**：检索增强型 ITP prover；本文与之兼容（BC-ReProver、BC-LLMStep），在其基础上叠加后向链式子目标发现，获得额外性能提升。
3. **Jiang et al. (2023) Draft-Sketch-Prove**：利用非形式化证明辅助形式化证明；本文继承"非形式化→形式化"思路，但进一步提出**伪步骤**这一中间层，并首次系统性地将后向链式纳入 ITP。
4. **Copra (Thakur et al., 2024)**：基于 GPT-4 的 ITP agent，通过助手反馈反复修正 tactic；本文不使用反复修正策略，而是通过后向链式一次性提供高质量子目标。
5. **DT-Solver (Wang et al., 2023b)**：基于状态复杂度的搜索策略；本文的搜索机制借鉴了 DT-Solver 的 token-count 复杂度度量，替代不可行的 best-first search。
6. **LLemma-7B (Azerbayev et al., 2023) / Deepseek-Math-7B (Shao et al., 2024)**：面向数学的大规模预训练 LLM；本文使用 GPT-4 API，强调框架层面的搜索策略改进而非模型规模竞争。

## 局限性与未来方向
- **无法解决 IMO 级问题**：BC-Prover 在 Olympiad 难题（尤其是 IMO 级别）上仍未取得突破，表明后向链式策略对极高难度问题仍有限制。
- **伪步骤静态不更新**：伪步骤在证明开始时生成后固定不变，不能随证明进展动态调整，依赖 LLM 自我修正能力，可能在复杂证明中产生偏差。
- **生成的子目标假说质量参差不齐**：定量分析显示约 84% 的 backward chaining 假说对最终证明无实质贡献，部分假说过于平凡或与已有引理高度重叠。
- **步骤规划复杂度限制小模型适用性**：SP 模块生成的指令较复杂，难以适配 7B 量级模型或微调模型（文中仅能用 BC 部分与微调模型协作）。
- **未来方向**：设计动态更新的伪步骤机制；提升后向链式生成假说的质量筛选能力；探索更适合小模型的轻量化 SP 模块；扩展到更多微调 prover。

## 研究启发与可借鉴点
1. **后向链式策略的通用价值**：将"从目标出发逆向推导子目标"的思想系统化引入 ITP，为其他需要在大搜索空间中导航的任务（如代码生成、形式化验证）提供了可迁移的推理范式。
2. **伪步骤（pseudo steps）的中间表示设计**：在非形式化与形式化之间引入结构化中间层，有效桥接语义鸿沟；此思路可推广至其他 autoformalization 场景。
3. **基于状态复杂度的搜索替代策略**：在无法获取 log probability 的 API 场景下，用 token 数量衡量状态复杂度作为搜索指标，是一种实用且可复用的工程方案。
4. **插件式框架设计**：BC-Prover 作为独立框架可无缝集成到 ReProver、LLMStep 等已有 prover 中，证明了"搜索策略改进"+"模型能力"正交可叠加的设计理念，降低了后续研究的部署成本。

## 关键术语表
- **Interactive Theorem Proving (ITP)**：交互式定理证明，通过逐步应用 tactics 与 Lean 等证明助手交互以构造形式化证明的过程。
- **Backward Chaining（后向链式推理）**：从待证明目标出发，逆向推导所需子目标和辅助假设的推理策略。
- **Forward Chaining（前向链式推理）**：从已知假设出发，正向应用 tactics 逐步推进的证明策略。
- **Pseudo Steps（伪步骤）**：由非形式化证明细化而来的结构化中间步骤，比非形式化证明更精确、比形式化证明更灵活，用于指导后续推理。
- **Step Planning（步骤规划）**：基于当前证明状态和伪步骤，为下一步 tactic 生成详细规划和引理使用策略的模块。
- **Pass@1**：单次尝试成功完成证明的比例，是 ITP 任务的主要评估指标。
- **miniF2F**：涵盖 488 道数学定理的 Lean 形式化证明基准，源自 AIME、AMC、IMO 等竞赛和本科数学课程。
- **Tactic**：Lean 中的证明步骤指令，用于更新证明状态，必须经证明助手验证。

## 可复现要素
- **数据集**：miniF2F benchmark（公开，https://github.com/homotopyfag/minif2f）
- **代码/权重**：论文未明确声明开源（Appendix A.2 提及部分基线因未公开代码/模型而无法复现，本文自身代码公开情况未明确说明）
- **关键超参**：LLM = gpt-4-turbo-2024-04-09；forward k = 16；backward d = 8；lemma re-ranking n = 5；temperature = 0；最大迭代次数 = 100
- **评估环境**：Lean 3，Pass@1 指标
