---
title: "Embedding-and-Gradient-Say-Wrong-A-White-Box-Method-for-Hall"
source: https://aclanthology.org/2024.emnlp-main.116.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:17:01"
---

# 论文速读：Embedding-and-Gradient-Say-Wrong-A-White-Box-Method-for-Hall

## 一句话总结
本文提出 EGH 白盒幻觉检测方法，通过构造条件输入（问题+答案）与无条件输入（仅答案）的对比，提取 Transformer 内部 embedding 差值与输出概率分布差异的一阶梯度特征，融合后输入轻量 MLP 分类器，在 HaluEval、SelfCheckGPT 与 HADES 等基准上均取得 SOTA 性能。

## 研究问题与动机
- **核心问题**：LLM 生成文本时常出现与输入或事实脱节的幻觉（Hallucination），现有检测方法多依赖黑盒 API 查询或外部提示，缺乏对开源/白盒模型内部信号的充分利用。
- **现有方法不足**：token 级概率/不确定性指标会丢弃大量模型内部信息；直接采用 KL 散度、交叉熵等单一标量衡量条件与非条件分布差异过于粗糙，阈值设定困难且精度有限。
- **研究动机**：白盒设置下可直接访问完整 token 概率矩阵与 embedding 层；论文假设幻觉生成时模型对源文本的信息依赖降低，因此可通过建模 $P(A|Q)$ 与 $P(A|\mathbf{0})$ 的分布差异来量化幻觉程度，并利用一阶泰勒展开将其转化为可高效提取的 embedding 与梯度特征。

## 核心贡献（创新点）
- **提出 EGH 白盒检测框架**：首次系统性地利用 LLM 内部 embedding 与一阶梯度信息构建幻觉检测器，与依赖黑盒查询或外部提示（如 ChatGPT/CONLI/KnowHalu）的方法形成本质区别。
- **基于泰勒展开的特征解耦设计**：将条件-无条件输出分布差异 $D(P(A|Q), P(A|\mathbf{0}))$ 进行一阶泰勒近似，忽略高计算成本的高阶项，与直接使用标量概率距离（如简单 KL/CE 阈值法）的方法存在本质区别。
- **条件/无条件双输入对比机制**：构造联合输入 $[Q,A]$ 与零填充输入 $[\mathbf{0},A]$ 两次前向传播，直接量化模型对源文本的依赖度，区别于 Filippova (2020) 需分别训练独立条件 LM 与无条件 LM 的重训练方案。
- **轻量级可插拔分类器**：设计仅含三层 MLP 的幻觉检测端，通过可学习权重 $\lambda$ 融合 $E$ 与 $G$，在冻结 LLM 参数的情况下实现即插即用，与 SAPLMA 等仅依赖单类隐层状态的方法相比特征维度更丰富。
- **多模型与多数据集全面验证**：在 LLaMa-2-7B、OPT-6.7B 以及传统模型 BERT/GPT-2/RoBERTa 上验证有效性，并在 HaluEval、SelfCheckGPT、HADES 三个基准上均达到 SOTA 或显著提升，证实方法的泛化性与跨架构适配能力。

## 方法详解
- **双输入构造**：给定问题 $Q$（$m$ 个 token）和生成答案 $A$（$n$ 个 token），构造两种输入序列：(1) 条件输入 $[Q, A]$，模拟有源文本时的生成；(2) 无条件输入 $[\mathbf{0}, A]$，其中 $\mathbf{0}$ 为长度为 $m$ 的零向量 padding，模拟无源文本时的生成，两者通过 zero-padding 对齐维度。
- **分布差异定义**：定义 $D([Q,A]) = \text{Difference}(P(A|Q), P(A|\mathbf{0}))$，具体实现为各 token 条件概率分布的 KL 散度之和：$D([Q,A]) = \sum_{i=1}^{n} D_{KL}[P(A_i|Q) || P(A_i|\mathbf{0})]$。
- **Taylor 展开与特征提取**：对 $D([Q,A])$ 在 $[\mathbf{0}, A]$ 处进行一阶泰勒展开。忽略高阶余项 $R_1$，保留两项关键因子：
  - **Embedding 特征 $E$**：用 $[Q,A]$ 与 $[\mathbf{0},A]$ 在最后一层的隐藏状态之差表示，即 $E = E(A|Q) - E(A|\mathbf{0})$，维度为 $\mathbb{R}^{n \times h}$。
  - **Gradient 特征 $G$**：对 $D([\mathbf{0}, A])$ 关于 embedding 层参数求梯度，即 $G = \nabla D([\mathbf{0}, A])$，维度同为 $\mathbb{R}^{n \times h}$，表征概率分布差异对模型内部表示的敏感度。
- **检测器融合与训练**：将 $E$ 与 $G$ 线性融合后输入三层 MLP 分类器 $f(\cdot)$（激活函数 ReLU），融合公式为 $y_{hal} = f(\lambda E + (1-\lambda)G)$，其中 $\lambda$ 为超参（实验取 0.8）。LLM 参数全程冻结，仅训练 MLP 及融合权重。
- **算法流程**：先后执行两次前向传播获取概率分布与 embedding，计算 KL 散度总和，反向传播获取梯度，待分类器输出幻觉标签 $y_{hal} \in \{0,1\}$。

## 实验与结果
- **数据集**：HaluEval（涵盖 QA、Dialogue、Summary 及 General 任务，5000 通用
