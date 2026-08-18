---
title: "AMR-Evol-Adaptive-Modular-Response-Evolution-Elicits-Better"
source: https://aclanthology.org/2024.emnlp-main.66.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:25:15"
---

# 论文速读：AMR-Evol: Adaptive Modular Response Evolution Elicits Better Knowledge Distillation for Large Language Models in Code Generation

## 一句话总结
本文提出AMR-Evol框架，通过“模块化分解”与“自适应响应演化”两阶段流程，将教师模型直接蒸馏的代码响应拆解为子函数模块，并借助已验证的函数模块数据库检索复用，从而显著提升开源代码大模型在复杂代码生成任务上的知识蒸馏质量与最终性能。

## 研究问题与动机
- 现有开源代码大模型的蒸馏工作（如Code Evol-Instruct、Magicoder、OSS-Instruct）普遍聚焦于**指令生成与复杂度扩展**，却忽视了**响应质量**的优化，导致蒸馏数据标签本身存在逻辑错误或偏离需求。
- 直接让教师模型为复杂指令生成完整代码响应时，容易因任务过难而产生次优甚至错误代码（如图1），这些低质响应会直接误导学生模型的监督学习。
- 雇佣编程专家人工编写单元测试验证成本高昂；依赖教师模型自生成单元测试并自我修复（Self-repair）同样存在“教师自己写错测试或修不好”的不确定性，缺乏一种低成本、自动化且可累积验证的响应优化机制。

## 核心贡献（创新点）
- 提出AMR-Evol（Adaptive Modular Response Evolution）两阶段响应蒸馏框架，首次将模块化编程思想系统引入代码知识蒸馏管线，填补了现有工作重指令、轻响应的空白。
- 设计模块化分解（MD）阶段，以直接蒸馏响应为种子，引导教师模型将复杂任务拆解为多个细粒度、功能明确的子模块函数，降低单次生成的认知与逻辑负担。
- 提出自适应响应演化（ARE）阶段，构建经单元测试验证的函数模块数据库，通过密集向量检索将语义最相近的已验证模块作为in-context示例辅助教师模型迭代生成优化响应，并动态收录新生成的高差异模块。
- 在HumanEval、MBPP及EvalPlus（HE-Plus/MBPP-Plus）三个主流代码基准上，对两种代表性学生模型与三种不同复杂度级别进行系统评估，全面超越Direct、CoT、AnsRepair等基线，且与同规模SFT数据的开放代码大模型相比展现显著性能优势。

## 方法详解
- **直接响应蒸馏（Direct Response Distillation）**：教师模型$\mathcal{M}_t$根据生成的代码指令$I$直接输出初始响应$R_d$，构建基础数据集$\mathcal{D}_{direct} = \{(I, R_d)\}$作为后续演化种子。
- **模块化分解（Modular Decomposition, MD）**：引导教师模型将$R_d$拆解为一组子模块函数$\{F_1^m, F_2^m, \ldots, F_n^m\}$（公式1），每个$F_i^m$对应原始任务的一个细分需求，使蒸馏焦点从“全量生成”转向“分步求解”。
- **函数模块数据库构建**：采用self-instruct策略让教师生成多样化种子函数，参考CodeT流程让教师为每个函数生成配套单元测试；仅将通过全部单元测试的函数入库，确保数据库内模块具备功能正确性。
- **自适应响应演化（Adaptive Response Evolution, ARE）**：
  - 使用句子嵌入模型$\mathcal{M}_r$（如`Alibaba-NLP/gte-large-en-v1.5`）将新分解模块与库中已验证模块映射为密集向量（公式2）。
  - 通过余弦相似度计算$\text{Sim}(F_i^m, F_j^v)$（公式3）检索与目标子模块最相似的已验证模块。
  - 教师模型结合指令$I$、分解模块$\{F_i^m\}$与检索到的相关模块$\{F_i^v\}$生成演化后的优化响应$R_{amr}$（公式4）。
  - 演化过程中若识别出与库内模块差异显著的新函数，经教师模型生成单元测试验证通过后自动并入数据库，实现知识的持续沉淀。
- **知识蒸馏训练**：学生模型$\mathcal{M}_s$在演化数据集$\mathcal{D}_{amr} = \{(I, R_{amr})\}$上进行自回归SFT，优化目标为$\mathcal{L}(\theta) = - \sum_{(I, R_{amr}) \in \mathcal{D}_{amr}} \log P(R_{amr} | I; \theta)$（公式5）。

## 实验与结果
- **实验设置**：教师模型为`gpt-3.5-turbo-1106`，学生模型为`deepseek-coder-6.7b-base`与`CodeLlama-7b-hf`；指令集基于MBPP训练子集经self-instruct与Code Evol-Instruct迭代生成，划分为Complexity Level 1/2/3共约30k样本；嵌入模型为`gte-large-en-v1.5`。
- **基线对比（Table 1 & 2）**：
  - DeepSeek-Coder-6.7B为Student时，Level 1下AMR-Evol较Direct取得最大提升：HE +3.6，HE-Plus +3.1，MBPP +2.8，MBPP-Plus +4.0。
  - 随复杂度升至Level 3，优势依然稳定（HE-Plus +1.8，MBPP-Plus +2.5）；CodeLlama-7B上Level 2/3在HE/MBPP/HE-Plus上获得+1.2~+2.8的提升。
- **人工评估（Figure 3）**：两位资深程序员对三个复杂度级别各120个样本进行正确性标注，AMR-Evol在各级别下的准确率均显著领先其他方法。
- **消融实验（Table 3）**：移除MD（w/o MD）或ARE（w/o ARE）均导致性能下降；MD缺失会使检索粒度与任务规模不匹配，ARE缺失则退化为完全依赖教师单次生成能力。
- **与开放代码LLM对比（Table 4 &
