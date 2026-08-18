---
title: "UniFashion: A Unified Vision-Language Model for Multimodal Fashion Retrieval and Generation"
source: https://aclanthology.org/2024.emnlp-main.89.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:24:35"
field: "多模态学习与 Fashion AI"
keywords: ["跨模态检索", "组合图像检索", "时尚图像生成", "多模态大语言模型", "扩散模型", "统一多模态模型"]
innovations: ["首次统一时尚域检索与生成任务，通过生成辅助检索实现任务互强化", "提出两阶段训练策略：先预训练生成模块建立跨模态表示，再微调编码器处理组合输入", "利用Q-Former+LLM+扩散模型的统一架构，在一个框架内同时支持CMR、CIR、FIC、FIG四类任务"]
benchmarks: ["FashionGen", "Fashion-IQ", "VITON-HD", "MGD"]
---

# 论文速读：UniFashion: A Unified Vision-Language Model for Multimodal Fashion Retrieval and Generation

## 一句话总结
论文提出UniFashion，一个统一框架首次将多模态检索（跨模态检索、组合图像检索）与生成（图像描述、图像生成）任务整合到单一模型中，通过结合LLM与扩散模型实现可控高保真生成，并在多个时尚任务上显著超越现有SOTA方法。

## 研究问题与动机
1. **单任务模型的局限性**：现有方法通常针对单一任务设计（如仅做检索或仅做生成），难以应对时尚域中异构的多种任务需求，限制了其在实际电商场景中的应用。
2. **检索任务的忽视**：当前MLLMs主要关注生成任务，对依赖嵌入能力的检索任务（如图像到文本、文本到图像检索）关注不足，且无法处理组合图像检索（CIR）任务。
3. **生成任务的缺失**：现有统一时尚模型大多缺乏图像生成能力，而试穿（try-on）和时尚设计等任务需要基于多模态输入生成目标图像。
4. **文本编码器能力受限**：现有时尚图像生成工作通常使用CLIP文本编码器，可能无法有效捕捉文本上下文信息，限制了生成质量。

## 核心贡献（创新点）
1. **首次统一检索与生成任务**：论文首次在时尚领域深入研究多模态检索与生成任务的协同建模，提出通用统一模型UniFashion，能够处理所有时尚相关任务，区别于现有仅专注于单一任务类型的工作。

2. **任务间互强化机制**：通过生成辅助检索（如利用Caption生成能力辅助CIR任务）和联合训练提升多模态编码器质量，这种双向增益机制是先前工作未探索的。

3. **两阶段训练策略**：提出独特的两阶段训练方案——第一阶段冻结Q-Former微调LLM和扩散模块，建立跨模态表示能力；第二阶段冻结LLM和扩散模块仅微调Q-Former，使模型能够处理组合输入，这一设计区别于现有单阶段训练方法。

4. **全面的性能提升**：在FashionGen（CMR）、Fashion-IQ（CIR）、FashionGen（FIC）和VITON-HD/MGD（FIG）等多个基准上显著超越现有单任务SOTA模型。

## 方法详解
**整体架构**：UniFashion由三部分组成——Q-Former（多模态编码器）、LLM模块（文本生成）和扩散模块（图像生成）。

**第一阶段：跨模态预训练**
- **跨模态检索**：使用Q-Former分别编码图像和文本，通过ITC损失（对比学习）和ITM损失（匹配学习）进行训练，损失函数为 $\mathcal{L}_{\mathrm{cross}} = \mathcal{L}_{\mathrm{ITC}}(t_c, Z_I) + \mathcal{L}_{\mathrm{ITM}}(Z_C, Z_I)$，其中 $\mathcal{L}_{\mathrm{ITC}}$ 使用可学习温度参数λ。
- **跨模态生成**：通过Task Specific Adapter（TSA）将Q-Former输出映射到LLM和扩散模型空间。文本生成采用图像 grounding 文本生成（ITG）目标：$\mathcal{L}_{\mathrm{ITG}} = -\frac{1}{L}\sum_{l=1}^{L}\log p_\phi(w_l^g|w_{<l}^g, f_\theta(q))$。图像生成采用扩散模型噪声预测损失：$\mathcal{L}_{\mathrm{q2I}} = \mathbb{E}_{\epsilon^y,\mathbf{x}_0}[\|\epsilon^x - \epsilon_\eta^x(\mathbf{x}_{t^x}, f_\zeta(q), t^x)\|^2]$。总损失：$\mathcal{L}_{\mathrm{ph1}} = \mathcal{L}_{\mathrm{cross}} + \mathcal{L}_{\mathrm{ITG}} + \mathcal{L}_{\mathrm{q2T}}$。

**第二阶段：组合多模态微调**
- **组合图像检索（CIR）**：将参考图像$I_R$和修改文本$t$通过Q-Former联合编码为$Z_R$，与目标图像编码$Z_T$对齐，损失为：$\mathcal{L}_{\mathrm{cir}} = \mathcal{L}_{\mathrm{ITC}}(e_{cls}, Z_T) + \mathcal{L}_{\mathrm{ITC}}(e_{cls}, Z_C) + \mathcal{L}_{\mathrm{ITM}}(t, Z_T)$。
- **组合多模态生成**：类似第一阶段，但使用$Z_R$作为条件，总损失：$\mathcal{L}_{\mathrm{stage2}} = \mathcal{L}_{\mathrm{cir}} + \mathcal{L}_{\mathrm{ITG}} + \mathcal{L}_{\mathrm{q2I}}$。

**指令微调**：针对不同数据集（Fashion200K、FashionGen、Fashion-IQ-cap）设计不同指令模板，引导LLM生成不同风格的描述。

## 实验与结果
**数据集**：
- FashionGen：260.5K图像-文本对，用于CMR和FIC
- Fashion-IQ：18K训练三元组，用于CIR
- VITON-HD和MGD：用于图像生成任务

**主要结果**：
- **跨模态检索（FashionGen）**：UniFashion在Image-to-Text（R@1: 71.44, R@5: 93.79, R@10: 97.51）和Text-to-Image（R@1: 71.41, R@5: 93.69, R@10: 97.47）上均显著超越FAME-ViL（平均R@10从93.52提升至97.47），均值达到87.55。
- **图像描述（FashionGen）**：BLEU-4达35.53，CIDEr达169.5，超越FAME-ViL的30.73和150.4。
- **组合图像检索（Fashion-IQ）**：平均R@10=58.93，R@50=76.93，超越SPRC的54.92和74.97。消融实验显示利用Caption生成辅助检索（UniFashion w/o cap vs UniFashion）带来+3.89的提升。
- **图像生成（VITON-HD）**：FID=8.42，KID=0.67，略低于StableVITON的8.23但接近，考虑到仅微调Q-Former而冻结扩散模块的情况下表现优异。MGD数据集上FID=12.43，CLIP-S=31.29，优于MGD的FID=12.81。

**消融验证**：Table 5显示，引入LLM和扩散模块后CIR性能从64.76提升至67.93，验证了生成任务对检索的辅助作用。

## 相关工作脉络
1. **FAME-ViL**：针对时尚的多任务视觉语言模型，支持CMR和CIR，但缺乏图像生成能力，UniFashion在此基础上增加了生成模块。

2. **GRIT**：将生成和嵌入任务统一到文本中心应用的模型，启发了本文在时尚域探索检索与生成的协同，但时尚域具有更复杂的异构任务特性。

3. **SPRC / TG-CIR**：专门的CIR方法，未考虑生成辅助，UniFashion通过联合训练证明生成能力可提升检索性能。

4. **StableVITON / MGD**：单独的虚拟试穿/时尚设计生成模型，仅关注生成任务，UniFashion将其纳入统一框架并实现多任务协同。

5. **BLIP-2**：采用Q-Former桥接图像编码器和LLM的方法被本文继承，但BLIP-2仅支持生成，本文扩展支持检索任务。

6. **FashionBERT / KaleidoBERT**：专门的时尚域跨模态检索模型，但无法处理组合检索和生成任务。

## 局限性与未来方向
1. **计算复杂度**：训练阶段涉及多个复杂模块（Q-Former、LLM、扩散模型），计算开销较大；推理阶段检索任务只需Q-Former，但图像生成任务因扩散过程仍较慢（A100上每张图约3.15秒）。

2. **生成与检索的潜在冲突**：消融实验显示引入扩散模型可能对图像描述生成能力产生轻微负面影响，可能源于LLM与扩散模型的对齐差异。

3. **未来方向**：论文建议探索更高效的采样方法（如DPM-Solver++）以提升推理效率；可扩展至更多时尚相关应用和电商场景。

## 研究启发与可借鉴点
1. **检索-生成协同训练策略**：证明生成任务可作为检索任务的辅助监督信号（如利用Caption生成辅助CIR检索），这一思路可迁移到其他多模态领域（如医学影像、文档理解）。

2. **两阶段参数冻结训练**：先训生成模块建立跨模态表示，再训编码器适应组合输入的策略，为多任务统一模型提供了可借鉴的训练范式。

3. **任务特定适配器（TSA）**：用轻量适配器桥接不同模态空间（Q-Former到LLM/扩散模型），避免了直接微调大规模预训练模型的资源消耗，值得在多模态统一模型中复用。

4. **指令微调适配不同数据分布**：针对不同数据集的Caption风格差异设计不同指令模板，解决了统一模型在多源数据上的适应性问题。

5. **半自动数据增强**：利用LLaVA-1.5（13B）为Fashion-IQ训练集生成详细Caption，展示了大模型辅助数据构建的实用价值。

## 关键术语表
- **Cross-Modal Retrieval (CMR)**：跨模态检索，在图像和文本之间进行双向检索匹配的任务。
- **Composed Image Retrieval (CIR)**：组合图像检索，基于参考图像和修改文本共同确定目标图像的检索任务。
- **Q-Former**：Querying Transformer，BLIP-2中的轻量多模态编码器，通过可学习查询桥接图像和文本模态。
- **Task Specific Adapter (TSA)**：任务特定适配器，线性投影层，将Q-Former输出映射到LLM或扩散模型的特征空间。
- **Image-Text Contrastive Learning (ITC)**：图像-文本对比学习，通过对比损失拉近匹配对的嵌入距离。
- **Image-Text Matching (ITM)**：图像-文本匹配，二分类任务判断图像-文本对是否匹配。
- **Latent Diffusion Model**：潜在扩散模型，在压缩的潜在空间中进行去噪生成的扩散模型，如Stable Diffusion。
- **Frechet Inception Distance (FID)**：评估生成图像质量的指标，衡量生成分布与真实分布的距离，值越低越好。

## 可复现要素
- **数据集**：FashionGen（公开）、Fashion-IQ（公开）、VITON-HD（公开）、MGD（公开）
- **代码**：已开源，地址 https://github.com/xiangyu-mm/UniFashion
- **权重**：Q-Former初始化自BLIP-2，LLM采用Vicuna-1.5（13B），扩散模块使用Stable Diffusion v1.4和Paint-by-Example权重，VAE微调自VITONHD
- **关键超参**：Q-Former使用32个query，维度768；LoRA rank=128，alpha=256；LLM学习率2e-5（cosine），batch=32；扩散模型学习率1e-4，迭代360k次，batch=32；采样步数50
