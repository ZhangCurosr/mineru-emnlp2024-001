---
title: "On-the-Role-of-Context-in-Reading-Time-Prediction"
source: https://aclanthology.org/2024.emnlp-main.179.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:14:15"
---

# 论文速读：On-the-Role-of-Context-in-Reading-Time-Prediction

## 一句话总结
本文通过 Hilbert 空间投影将 surprisal 投影至词频正交补空间，构造出与词频严格不相关的纯语境预测器；实证发现正交化后语境对阅读时间的独立解释方差大幅缩减，表明既往基于 surprisal 的研究可能因共线性而高估了语境在实时语言加工中的作用。

## 研究问题与动机
1. **核心问题**：Surprisal theory 假设加工努力与词级条件 surprisal 呈仿射关系，但 surprisal 与词频高度共线，难以区分“纯语境贡献”与“词频贡献”。
2. **现有方法不足**：既往线性回归通常同时引入 surprisal 与频率，但由于 surprisal、PMI 与频率三者线性相关，模型预测力完全等价，导致语境效应被错误归因。
3. **动机**：从可解释性出发，需构建一个与词频正交的预测器，量化语境在剔除词频影响后的真实解释力占比。
4. **理论缺口**：缺乏对语言模型派生预测器的几何解耦框架，未能形式化分离语境信息与非语境先验信息。

## 核心贡献（创新点）
1. **提出正交化 surprisal 预测器**：将 surprisal 投影至词频正交补空间，得到与频率严格不相关的纯语境信号。与已有工作本质区别：从函数空间几何视角形式化解耦，而非仅在样本层面做残差化。
2. **证明 surprisal 与 PMI 的线性等价性**：推导得出 $\mu_H = v_H - \iota_H$，表明含 surprisal+频率 与 PMI+频率 的线性模型预测力完全相同。区别在于：打破了 surprisal 的唯一性地位，揭示既往实证结果同样支持基于关联的 PMI 理论。
3. **量化语境独立贡献并纠偏高估倾向**：在多语言 MECO 数据上使用 LMG 方差分解，发现正交化后语境解释力骤降，词频成为绝对主导。区别在于：首次用严谨的相对重要性度量分离共享方差，纠正了心理语言学文献中的归因偏差。
4. **构建跨建模假设的稳健验证 pipeline**：将正交化推广至词长预测器，并用 OLS 线性模型与 GAMs 非线性模型交叉验证结论。区别在于：超越单一线性框架，证明解耦结论对建模假设不敏感。

## 方法详解
- **概率与 Hilbert 空间形式化**：定义规范化的前缀概率 $\pi_H(c)$ 与条件概率 $p_H(\bar{u}|c)$；将 surprisal $I_H$、PMI $M_H$、频率 $Y_H$ 视为 $L^2$ Hilbert 空间 $\mathcal{H}$ 中的随机变量，内积为 $\langle X, Y \rangle = \mathbb{E}[XY]$。
- **线性依赖推导**：由定义得 $\mu_H(c, \bar{u}) = \log \frac{p_H(\bar{u}|c)}{q_H(\bar{u})} = v_H(\bar{u}) - \iota_H(c, \bar{u})$，三者共线，任意两者组合的线性回归预测值等价。
- **正交化构造**：对中心化后的变量，将 surprisal 投影至频率正交补：$\text{proj}_{Y_H^\perp}(I_H) = I_H - \frac{\text{Cov}(I_H, Y_H)}{\text{Var}(Y_H)} Y_H$，使新预测器与频率协方差为 0。
- **实用近似**：精确内积计算不可行，改用训练语料上的无偏样本协方差/方差估计；形式上等价于对频率回归的残差化，但解释为泛函空间的几何投影。
- **实验建模**：OLS 线性回归三组预测因子：(i) surprisal + 频率 + 词长；(ii) PMI + 频率 + 词长；(iii) 正交化 surprisal + 频率 + 正交化词长；均加入前一词溢出效应（spillover）。
- **重要性评估**：采用 LMG 方法（按预测因子加入顺序平均化解释方差比例）替代 Delta Log-Likelihood，获取更准确的绝对贡献度量；辅以 GAMs（基函数维度 $k=6$）验证非线性稳健性。

## 实验与结果
- **数据集**：Multilingual Eye-movement Corpus (MECO)，包含 11 种语言（Dutch, English, Finnish, German, Greek, Hebrew, Italian, Korean, Russian, Spanish, Turkish）的词级 gaze duration。
- **预测器来源**：Surprisal/PMI 由 mGPT（多语言 GPT-2 变体）计算；频率来自 Speer (2022)；词长按相同正交化流程处理。
- **主要结果**：三模型整体 $R^2$ 相同（均值 0.6–0.8），证实线性等价性；但 LMG 分解显示原始 surprisal 解释方差显著，正交化 surprisal 解释方差大幅缩减，PMI 介于两者之间。
- **最强结果**：词频（frequency）在所有语言中均为最强预测因子；正交化后语境贡献退居次席，多数语言中甚至低于前词频率溢出效应。
- **结论**：语境对阅读时间的独立贡献远小于既往认知，surprisal theory 的实证支持很大程度上源于其与词频的共线性被误归因于语境效应。

## 相关工作脉络
1. **Surprisal Theory (Hale 2001; Levy 2008)**：本文理论起点，但本文指出未解耦频率的 surprisal 会高估语境贡献。
2. **PMI 在心理语言学的应用 (Tsipidi et al. 2024; Wilcox et al. 2024)**：本文证明 PMI 与 surprisal 在线性框架下预测力等价，为关联理论提供统一解释。
3. **词频主导效应 (Shain 2019, 2024; Bybee 2006)**：本文与 Shain 结论一致，但提供了 Hilbert 空间投影的严格解耦机制。
4. **Lossy-context Surprisal (Futrell et al. 2020; Kuribayashi et al. 2022)**：本文解释为何限制上下文会提升拟合度：受限 surprisal 更接近频率，而频率本就更强预测阅读时间。
5. **大模型尺寸与 surprisal 拟合度倒U曲线 (Oh & Schuler 2023; Oh et al. 2024)**：本文为本团队观察到的现象提供新解释：大模型成功解耦 surprisal 与频率后，纯语境信号本身微弱，导致拟合度下降。

## 局限性与未来方向
- 使用固定效应线性回归而非混合效应模型，$R^2$ 估计可能系统性偏低。
- 语言样本偏向印欧语系（7/11），韩语因 Hangul 文字系统呈现独特分布，结论泛化性受限。
- 正交化依赖样本协方差估计，实践中仅为“近似正交”，无法保证严格解耦。
- 未充分控制眼动轨迹细节（如单字多跳 saccade），词长解释力可能混杂运动规划成本。
- 未来方向：结合混合效应模型与偏效应量；扩展至更多语言家族；验证正交化框架在其他认知指标（ERP、瞳孔直径）中的适用性。

## 研究启发与可借鉴点
1. **共线预测因子的几何解耦范式**：Hilbert 空间投影/残差化方法可迁移至句法复杂度、语义一致性等 NLP-认知交叉任务，避免共享方差被错误归因。
2. **LMG 作为标准重要性度量**：在多维共线预测器评估中，LMG 比 Delta LL 提供更稳定、可加的方差解释比例，建议纳入团队实验管线。
3. **大模型 surprisal 的心理语言学解读**：本文提示“更好预测罕见词的 LM 反而拟合人类阅读时间更差”并非模型退化，而是成功解耦了本就微弱的纯语境信号；对模型选择与认知 benchmark 设计有直接启发。
4. **正交化词长策略**：将正交化推广至词长等基线协变量可分离“纯长度效应”，值得在眼动/ERP 研究中作为控制变量标准化使用。
5. **开源数据清洗实践**：作者公开了 MECO 数据集的 ID bug 修复代码，体现了高可复现性，可作为多语言眼动研究的数据预处理参考。

## 关键术语表
- **Surprisal**：语言模型条件概率负对数 $-\log p(\text{word}|\text{context})$，表征词级预测误差与信息量。
- **Orthogonalized Surprisal**：将 surprisal 投影至词频正交补空间后得到的预测器，严格剔除共线性，代表纯语境效应。
- **Pointwise Mutual Information (PMI)**：词与上下文联合概率与独立概率之比的对数；本文证明其与 surprisal 在线性模型中等价。
- **Unigram Surprisal / Frequency**：词的一元语言模型概率负对数，表征词的固有熟悉度，与语境无关。
- **LMG**：通过枚举预测因子所有加入顺序并平均解释方差比例，量化各因子相对重要性的统计方法。
- **Gaze Duration**：眼动阅读任务中读者对目标词首次连续注视的总时长，广泛用于衡量初始加工难度。
- **Normalized Prefix Probability**：语言模型前缀分布的形式化定义，用于严格推导条件概率与无条件概率的关系。
- **Spillover Effect**：当前词阅读时间受前一词预测因子影响的溢出效应，常作为控制变量纳入回归模型。

## 可复现要素
- 数据集：MECO（Multilingual Eye-movement Corpus），公开可用；论文已提供原始数据 bug 修复代码。
- 代码：https://github.com/rycolab/context-reading-time（已开源）
- 权重/模型：mGPT、Speer (2022) 词频数据，均可公开
