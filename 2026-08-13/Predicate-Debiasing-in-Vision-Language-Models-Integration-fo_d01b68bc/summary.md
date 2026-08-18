---
title: "Predicate-Debiasing-in-Vision-Language-Models-Integration-fo"
source: https://aclanthology.org/2024.emnlp-main.97.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:15:04"
field: "视觉-语言多模态学习"
keywords: ["Scene Graph Generation", "Predicate Debiasing", "Vision-Language Models", "Label Shift", "Logits Adjustment", "Long-tail Learning", "Model Ensemble"]
innovations: ["提出LM Estimation通过约束优化估计预训练模型不可达的谓词分布", "设计certainty-aware动态集成策略融合零样本VLM与SGG模型", "首次系统解决预训练VLM在SGG中的predicate bias问题"]
benchmarks: ["Visual Genome"]
---

# 论文速读：Predicate-Debiasing-in-Vision-Language-Models-Integration-fo

## 一句话总结
论文针对场景图生成（SGG）中的**子表示不足（underrepresentation）**问题，提出通过集成预训练视觉语言模型（VLM）来增强表示，并设计了**拉格朗日乘数估计（LM Estimation）**方法校准VLM中因预训练语料谓词分布不平衡导致的predicate bias，最终通过certainty-aware动态集成显著提升SGG性能。

## 研究问题与动机
- **子表示不足是SGG固有难题**：SGG三元组（主体-谓词-客体）具有指数级组合多样性，训练集无法覆盖全部组合，导致测试集中存在大量训练未见过或稀有三元组，预测质量差（如图1所示"carrying"类样本置信度极低）。
- **直接引入零样本VLM存在严重predicate bias**：预训练语言数据中谓词分布极度不平衡（如常见关系频次远高于长尾关系），导致VLM直接推理时偏向高频谓词，无法有效弥补长尾样本。
- **现有去偏方法不适用于预训练VLM**：传统SGG去偏方法依赖训练过程干预（如reweighting loss），但预训练数据保密且预训练目标与SGG无直接标签对应，无法获取预训练分布π_pt。
- **需要一种无需训练的去偏与集成方案**：设计训练free的方法，在post-hoc阶段调整logits并动态融合两个模型的优势，最大化泛化能力。

## 核心贡献（创新点）
- **首创将预训练VLM集成用于缓解SGG子表示问题**：区别于以往仅优化SGG架构的工作，本文利用VLM的全面知识补偿训练集中稀有/未见三元组，定位从"架构改进"转向"知识迁移"。
- **提出LM Estimation（拉格朗日乘数估计）**：在无法获取预训练分布的约束下，通过约束优化最小化交叉熵损失间接估计π_pt，与已有依赖显式训练分布的去偏方法形成本质区别——前者无需访问预训练数据，后者需完整分布信息。
- **设计certainty-aware动态集成策略**：基于每个样本在两个模型中的置信度分数自适应调整融合权重，而非固定比例融合，使强表示样本来自最优模型，弥补单一模型局限。
- **提出无训练（training-free）端到端部署方案**：整个流程无需额外微调，仅需后处理logits调整和加权融合，显著降低计算与内存开销。

## 方法详解
- **双分支架构**：模型由固定零样本VLM分支$f_{zs}$（如ViLT、Oscar）和可训练任务特定分支$f_{sg}$（微调VLM或PENET等SGG模型）组成；$f_{zs}$使用MLM提示"{z_i} is [MASK] {z_j}"提取K个谓词logits，背景类仅由$f_{sg}$预测。
- **谓词偏差的形式化建模**：基于label shift假设（条件概率$P(x|r)$在训练/测试域相同），推导出训练分布$P_{tr}(r)$与测试目标分布$P_{ta}(r)$不一致时产生的偏差，明确两种场景：①$f_{sg}$在均匀目标下需去偏；②$f_{zs}$在两种目标下均需去偏（因π_pt≠π_sg且π_pt≠1/K）。
- **Post-hoc Logits Adjustment**：对初始logits $\mathbf{o}^k$调整后得到$\hat{\mathbf{o}}^k(r) = \mathbf{o}^k(r) - \log P_{tr}(r) + \log P_{ta}(r)$，其中$f_{sg}$用$\pi_{sg}$替换$P_{tr}$，$f_{zs}$用待估计的$\pi_{pt}$替换。
- **LM Estimation核心推导**：构建约束优化问题$\pi_{pt} = \arg\min_{\pi_{pt}} R_{ce}(\mathbf{o}^k - \log \pi_{pt} + \log \pi_{sg}, r)$，约束为$\pi_{pt}(r)\geq 0$且$\sum_r \pi_{pt}(r)=1$，使用Lagrange乘数法求解；关键在于将$P_{ta}(r)$设为$\pi_{sg}$以消除SGG数据集分布干扰，确保解出的$\pi_{pt}$纯粹反映预训练分布特征。
- **τ-calibration温度缩放**：对debias后的logits除以温度τ再做softmax，避免过置信问题，得到$\hat{P}_{zs}$和$\hat{P}_{sg}$。
- **Certainty-aware权重计算**：定义$\text{conf} = \max_{r} \hat{P}(r|z_i,z_j,\mathbf{x}_{i,j})$， ensemble权重$W_{cer} \propto \text{sigmoid}(\text{conf}_{sg} - \text{conf}_{zs})$，使高置信模型获得更大权重。
- **背景类处理与最终输出**：背景类概率仅由$f_{sg}$提供，非背景类概率按$W_{cer}$加权融合后输出最终预测。

## 实验与结果
- **数据集**：Visual Genome（VG），108,077张图像，150个物体类、50个谓词类，70%训练/30%测试，5,000张验证集。
- **评估任务**：Predicate Classification (PredCls) 和 Scene Graph Classification (SGCls)，主要指标为Recall@K和mRecall@K。
- **基线模型**：fine-tuned ViLT、fine-tuned Oscar、PENET-Rwt（重新实现），zero-shot ViLT、Oscar作为$f_{zs}$。
- **最强结果（Table 2）**：ViLT ft-la + Ours达到**mR@100 = 46.5（+2.0提升）**、**R@100 = 69.8（+1.4提升）**；PENET + Ours达到**R@100 = 71.1（+1.0提升）**，均为当时最优。
- **mean Recall显著提升**：相比Recall，mRecall增益更突出（+1.4/+2.0/+1.6 vs +1.7/+1.4/+1.0），表明对长尾谓词的改善尤为明显。
- **未见三元组增益（Table 3）**：Debiased Oscar在unseen triplets上mAcc从6.68提升至16.01（+9.33），ensemble进一步达19.56（+5.71相对于fine-tuned），远超all triplets增益。
- **稀有类别增益（Appendix Table 5）**：rare类别mRecall@100增益达+3.15~+4.13，confirm underrepresentation问题的缓解效果。

## 相关工作脉络
- **传统SGG去偏方法（Tang et al., 2020; Dong et al., 2022; Li et al., 2021; Yan et al., 2020）**：通过训练阶段干预（reweighting、hybrid attention、bipartite graph）缓解关系类别不平衡，但依赖训练时分布信息，无法迁移至预训练VLM的post-hoc去偏。
- **预训练VLM在关系识别中的早期应用（He et al., 2022; Zhang et al., 2023; Yu et al., 2023）**：探索prompt-tuning实现open-vocabulary SGG，但未系统解决预训练数据predicate bias问题，本文填补了这一空白。
- **长尾分类中的Logits Adjustment（Menon et al., 2020）**：首次将post-hoc logit adjustment引入SGG去偏，本文将其扩展至不可达预训练分布的VLM场景，通过LM Estimation解决分布估计难题。
- **VLM集成策略（Kumar et al., 2022）**：calibrated ensembles用于缓解分布偏移下的精度权衡，本文借鉴其思想设计certainty-aware indicator，但应用于VLM与SGG模型的跨域融合场景。
- **视觉关系检测新方法（Zheng et al., 2023; Lin et al., 2022b）**：PENET和RU-Net代表当前SGG SOTA，本文证明其可与VLM集成进一步突破性能瓶颈，展示方法的通用兼容性。

## 局限性与未来方向
- **推理计算开销增加**：ensemble框架引入$f_{zs}$的额外推理成本，尤其在Scene Graph Detection（SGDet）场景中，每幅图像需对80×80个object pair逐一forward VLM，计算代价极高。
- **性能依赖预训练质量**：最终集成效果高度依赖$f_{zs}$的预训练数据规模与质量，若预训练语料覆盖不足，去偏后增益有限。
- **Pair-wise推理效率瓶颈**：采用逐对forward的VLM调用方式，在object数量多时耗时显著，Appendix B.1建议未来结合更高效detector以降低计算负担。
- **未充分考虑三元组条件先验**：附录C讨论指出，若直接对三元组分布去偏会抹除条件先验（如"man-horse"更可能是"riding"而非"carrying"），当前仅对谓词去偏是合理折衷，但未来可探索更精细的去偏粒度。
- **仅评估PredCls和SGCls**：因计算成本过高跳过SGDet子任务，未验证方法在完整pipeline中的适用性，未来需在检测+关系联合场景下验证。

## 研究启发与可借鉴点
- **LM Estimation的可迁移性**：该约束优化框架可推广至其他预训练模型（如LLM、多模态大模型）的post-hoc去偏场景，尤其适用于预训练分布不可获知的情况，为"黑盒模型校准"提供新思路。
- **Certainty-aware动态集成范式**：基于样本级置信度差值的sigmoid加权策略简洁有效，可复用于其他多模型集成场景（如检测-分割联合、跨域适配），作为plug-and-play模块降低工程复杂度。
- **从"去偏训练"到"去偏推理"的范式转换**：传统方法依赖训练阶段干预，本文证明仅通过后处理logits调整即可实现等效去偏，为资源受限场景（如部署阶段calibration）提供低成本解决方案。
- **未见三元组作为评估基准**：Table 3中unseen triplet上的mAcc/Acc指标设计精妙，能直接反映模型对子表示问题的改善程度，可作为未来SGG工作的标准评测维度。
- **跨任务知识迁移的debiasing必要性**：本文揭示"直接集成预训练模型≠性能提升"的关键洞察——未经去偏的VLM甚至会损害下游性能，提醒研究者在新旧模型融合前必须校验分布对齐问题。

## 关键术语表
- **Scene Graph Generation (SGG)**：场景图生成，从图像中识别物体及其间关系并构建结构化图表示的视觉-语言任务。
- **Predicate Bias**：谓词偏差，指模型因训练数据中谓词分布不平衡而系统性地偏向高频谓词的预测倾向。
- **Underrepresentation**：子表示不足，训练集中某些三元组组合出现极少或未出现，导致模型对这些样本预测能力弱。
- **LM Estimation**：拉格朗日乘数估计，通过约束优化最小化交叉熵损失间接估计预训练数据中不可达的谓词分布。
- **Post-hoc Logits Adjustment**：后处理logits调整，在推理阶段对模型输出logits进行$\log P_{tr} \to \log P_{ta}$偏移校正的去偏技术。
- **Certainty-aware Ensemble**：置信度感知集成，根据每个样本在多个模型中的预测置信度动态分配融合权重的策略。
- **Label Shift**：标签偏移，指训练集与测试集的目标变量（标签）分布不一致的现象。
- **Mean Recall (mRecall)**：平均召回率，对所有关系类别分别计算召回率后取均值，衡量模型对长尾类别的识别能力。

## 可复现要素
- **数据集**：Visual Genome (VG)，公开可用（https://visualgenome.org/）。
- **代码**：论文未明确声明代码开源状态。
- **权重**：使用ViLT和Oscar的pretrained checkpoint，公开可下载；PENET为重新实现版本。
- **关键超参**：τ-calibration温度τ（论文未明确给出具体数值，标注为"outlined in (Kumar et al., 2022)"）；Faster R-CNN独立训练；ViLT冻结backbone只训练MLP head 50 epochs；Oscar端到端fine-tune 70k steps。
- **评估设置**：150 object classes + 50 predicate classes，70/30 train/test split，5,000 validation images。
