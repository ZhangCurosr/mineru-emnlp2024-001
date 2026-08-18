---
title: "UNIGEN: Universal Domain Generalization for Sentiment Classification via Zero-shot Dataset Generation"
source: https://aclanthology.org/2024.emnlp-main.1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:04:05"
field: "低资源自然语言处理"
keywords: ["零样本数据集生成", "领域泛化", "监督对比学习", "伪标签重标记", "去噪记忆银行", "轻量级任务模型"]
innovations: ["通用提示词驱动零样本跨域数据集生成", "基于伪标签软重标记的生成阶段去噪方法", "结合样本权重的去噪记忆银行机制"]
benchmarks: ["SST-2", "IMDB", "Rotten Tomatoes", "Amazon", "Yelp", "CR", "Tweet", "Amazon Review 29-domain"]
---

# 论文速读：UNIGEN: Universal Domain Generalization for Sentiment Classification via Zero-shot Dataset Generation

## 一句话总结
UNIGEN 提出一种基于零样本数据集生成的通用领域泛化方法：用通用提示词从大型预训练语言模型（PLM）生成领域不变的训练数据，再训练一个参数量仅为 PLM 不到 10% 的小型任务模型（TAM），使 TAM 能在任意共享标签空间的领域上直接推理，无需人工标注与领域适配。

## 研究问题与动机
1. **零样本数据集生成的领域局限**：现有方法（如 ZEROGEN）依赖特定领域提示词（如"The movie review in positive sentiment is:"），生成的训练数据只反映单一领域，导致训练出的 TAM 跨领域泛化能力差。
2. **多领域适配成本过高**：若采用 ZEROGEN/SUNGEN 策略覆盖 5 个领域，需分别生成 5,000k 数据并训练 5 个独立 TAM，生成与训练成本随领域数线性增长。
3. **现有领域泛化方法需要多源标注数据**：传统文本领域泛化（如 Tan et al., 2022）依赖多个源领域的人工标注数据，难以在真实低资源场景中获取。
4. **生成数据噪声问题未解决**：PLM 生成的文本可能带有错误标签或无关内容（Ye et al., 2022b; Gao et al., 2023），已有去噪工作仅关注训练阶段，未在生成阶段引入去噪机制。

## 核心贡献（创新点）
1. **通用提示词驱动零样本数据集生成**：设计领域不变提示词"The text in <y> sentiment is:"替代领域特定提示词，使 PLM 在生成阶段即脱离领域约束；与 ZEROGEN 的本质区别在于不依赖目标领域的先验信息。
2. **基于伪标签的软重标记去噪方法**：利用 PLM 对生成文本重新计算软标签，过滤置信度过低的样本（阈值 $T_{RE}=0.2$），比硬重标记和完全无重标记在 SST-2 上分别提升 1.09 和 3.16 个百分点。
3. **去噪记忆银行机制**：将 SUNGEN 学习的样本权重用于记忆银行，仅保留权重超过阈值（$T_{MB}=0.8$）的高质量样本参与对比学习，避免噪声样本污染负样本分布。
4. **轻量级统一模型替代多模型适配**：仅需 1 次生成 + 1 次训练即可获得跨域 TAM，相比 ZEROGEN/SUNGEN 的 5 倍数据与 5 倍模型成本，大幅降低实际应用门槛。

## 方法详解
### 3.1 基础框架
- **零样本数据集生成（扩展自 ZEROGEN）**：从均匀分布采样伪标签 $y_{syn}$，输入通用提示词 $\mathcal{T}_{uni}(y_{syn})$，由 PLM $\mathcal{P}$ 生成文本 $\mathbf{x}_{syn} \sim \mathcal{P}(\cdot|\mathcal{T}_{uni}(y_{syn}))$，构成合成数据集 $\mathcal{S}_{syn} = (\mathcal{X}_{syn}, \mathcal{Y}_{syn})$。
- **监督对比学习（SCL）**：
$$
\mathcal{L}_{SCL} = -\sum_{\mathbf{z}_i \in B} \frac{1}{|P(i)|} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau_{SCL})}{\sum_{\mathbf{z}_a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau_{SCL})}
$$
其中 $P(i)$ 为同标签正样本集，$A(i)$ 为除锚点外的所有样本集，$\tau_{SCL}$ 为温度参数。
- **记忆银行与动量编码器**：记忆银行 $\mathcal{M}$ 存储历史样本以扩大负样本数量；动量编码器 $\theta_k \gets m\theta_k + (1-m)\theta_q$ 平滑更新，$m=0.999$。

### 3.2 UNIGEN 训练损失
$$
\mathcal{L} = \mathcal{L}_{CE} + \alpha \mathcal{L}_{SCL}
$$
其中 $\alpha=0.5$ 平衡交叉熵与监督对比损失。

### 3.3 伪标签重标记去噪
对生成文本 $\mathbf{x}_{syn}$，通过 verbalizer $\mathcal{M}$ 将标签转为文本形式，利用 PLM 计算 logits：
$$
\ell(y_i|\mathbf{x}_{syn}) = \mathcal{P}(\mathcal{M}(y_i)|\mathcal{T}_{uni}(\mathbf{x}_{syn}))
$$
经温度 $\tau_{RE}=0.1$ 的 softmax 得软标签 $\hat{y}_i$，丢弃最大概率低于 $1 + T_{RE} = 0.7$（二分类场景）的样本。

### 3.4 去噪记忆银行
采用 SUNGEN 的双层优化 learned 样本权重 $w$，仅将 $w > T_{MB}=0.8$ 的高质量样本存入记忆银行 $\mathcal{M}$，使对比学习的负样本主要来自干净数据。

## 实验与结果
### 数据集
7 个情感分类数据集：IMDB、SST-2、Rotten Tomatoes（电影评论）；Amazon（产品评论）；Yelp（餐厅评论）；CR（电子产品评论）；Tweet（推特消息）。

### 评估基线
- **PROMPTING**：GPT2-XL（1.5B 参数）直接推理
- **ZEROGEN**：领域特定提示词生成 1,000k 数据
- **SUNGEN**：零样本噪声鲁棒加权生成
- **SUPERVISED**（Tan et al., 2022）：需 3 个源领域人工标注数据的有监督领域泛化

### 主要结果
| 模型 | 参数量 | 平均准确率（7 域） | 对比 PROMPTING |
|------|--------|-------------------|----------------|
| GPT2-XL (PROMPTING) | 1.5B | 78.25% | — |
| LSTM + UNIGEN | 7M | 64.52% | 低于基线 |
| DistilBERT + UNIGEN | 66M | **75.68%** | 接近 PROMPTING |
| **RoBERTa + UNIGEN** | **125M** | **81.45%** | **超越 PROMPTING 3.20 个百分点** |

- **RoBERTa + UNIGEN** 在 7 个测试域中全部优于 GPT2-XL PROMPTING，且参数量仅为 GPT2-XL 的 **8.3%**。
- 在 Amazon 29 域大规模实验中，RoBERTa + UNIGEN 平均 **89.51%**，与 GPT2-XL PROMPTING 的 **89.30%** 相当。
- **消融结果**：去掉软重标记（-3.16%）、去掉去噪记忆银行（-0.84%）、去掉 SCL（-2.99%）均显著降分。
- **PLM 选择**：GPT2-XL 作为生成器效果最佳， newer 小参数模型（Gemma-2b、Qwen2-1.5B）表现反而下降。

## 相关工作脉络
1. **ZEROGEN**（Ye et al., 2022a）：首次提出用 PLM 生成零样本训练数据，但依赖领域特定提示词，泛化能力受限；UNIGEN 通过通用提示词与去噪机制扩展其适用范围。
2. **PROGEN**（Ye et al., 2022b）：引入 in-context feedback 缓解生成噪声；UNIGEN 在生成阶段直接引入软重标记，无需额外反馈循环。
3. **SUNGEN**（Gao et al., 2023）：训练阶段噪声鲁棒加权；UNIGEN 将此思想延伸至记忆银行过滤，并加入生成阶段重标记。
4. **SUPERVISED**（Tan et al., 2022）：基于监督对比学习与记忆银行的有监督领域泛化；UNIGEN 在零样本设定下复现其对比学习框架，并去除对多源标注数据的依赖。
5. **Test-time learning**（Jeong et al., 2023）：论文提及可作为未来方向，利用测试时上下文示例生成少量领域特定数据微调 TAM。

## 局限性与未来方向
1. **单领域性能低于领域适配基线**：UNIGEN 的 TAM 在单一领域的性能弱于在该领域专门训练的 ZEROGEN/SUNGEN 模型，存在"通用性 vs. 专用性能"的 tradeoff。
2. **提示词设计仍较简单**：当前通用提示词"The text in <y> sentiment is:"缺乏对多样表达的引导，更有效的提示词可能进一步提升生成质量。
3. **小参数 TAM（LSTM/TextCNN）表现不佳**：附录 E 指出 TextCNN 因固定窗口限制难以理解多样化生成文本，仅预训练知识蒸馏模型（TinyBERT）在小参数场景下表现良好。
4. **未来方向**：（1）结合 test-time learning 生成少量目标域数据微调；（2）开发统一框架自动优化各 PLM 的超参与提示词；（3）将重标记策略扩展至 ZEROGEN/SUNGEN 等现有方法。

## 研究启发与可借鉴点
1. **生成阶段去噪可显著提升下游性能**：UNIGEN 在 PLM 生成后立即进行软重标记过滤，比仅在训练阶段去噪（SUNGEN）效果更优，提示未来数据集生成工作应将去噪视为生成流程的组成部分而非后处理。
2. **记忆银行与噪声鲁棒性的结合**：将样本权重用于记忆银行筛选，使对比学习的负样本质量同步提升，这一思路可迁移至其他对比学习场景的噪声治理。
3. **通用提示词替代领域适配的可行性**：用一条通用提示词覆盖多领域生成，避免了为每个新领域重新设计提示词的成本，对多领域低资源场景具有实用价值。
4. **PLM 生成器选型反直觉结论**：更大更新的 PLM 未必带来更好的 TAM 性能（GPT2-XL 优于 Gemma-2b/Qwen2-1.5B），提示未来工作应系统评估不同规模 PLM 在数据集生成中的边际收益。

## 关键术语表
- **Zero-shot Dataset Generation**：利用 PLM 根据提示词生成合成训练数据，无需人工标注即可训练下游模型。
- **Task-Specific Model (TAM)**：参数量远小于 PLM 的轻量级任务模型，部署时推理成本低。
- **Universal Prompt**：不绑定特定领域的提示词模板（如"The text in <y> sentiment is:"），用于生成领域不变的训练数据。
- **Supervised Contrastive Learning (SCL)**：利用标签信息构造正负样本对，拉近同标签样本表示、推远不同标签样本表示的对比学习变体。
- **Momentum Encoder**：通过指数移动平均平滑更新编码器的参数，使记忆银行中的表示更稳定。
- **Denoising Memory Bank**：仅保留高质量（高权重）样本的记忆银行，过滤噪声样本对对比学习的干扰。
- **Pseudo-Relabeling**：用 PLM 对生成文本重新计算软标签，用于过滤或校正原始伪标签。
- **Domain Generalization**：在多个源领域训练，使模型能泛化到未见目标领域，无需目标域标注数据。

## 可复现要素
- **数据集**：IMDB、SST-2、Rotten Tomatoes、Amazon、Yelp、CR、Tweet 均为公开数据集；Amazon Review 29 域（5-core）亦公开。
- **代码**：论文注明"Please refer to attached source code"，但未提供链接；需联系作者或查看 ACL Anthology 页面。
- **关键超参**：top-k=40、top-p=0.9、$\tau_{RE}=0.1$、$T_{RE}=0.2$、$\alpha=0.5$、$\tau_{SCL}=0.2$、记忆银行大小=64、动量系数 $m=0.999$、$T_{MB}=0.8$；DistilBERT/RoBERTa 学习率 $2e-5$，训练 3 epoch；LSTM 学习率 $1e-3$，训练 5 epoch。
- **硬件**：单卡 NVIDIA A100 40GB GPU。
