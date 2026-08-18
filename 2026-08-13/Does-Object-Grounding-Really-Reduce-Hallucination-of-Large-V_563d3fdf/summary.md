---
title: "Does-Object-Grounding-Really-Reduce-Hallucination-of-Large-V"
source: https://aclanthology.org/2024.emnlp-main.159.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:33:34"
field: "多模态大模型幻觉与定位"
keywords: ["object hallucination", "grounding objective", "referring expressions", "grounded captioning", "CHAIR-MEN", "LVLM", "OOD evaluation", "faithfulness-informativeness tradeoff"]
innovations: ["首次系统验证 RE/GC 定位目标对开放描述幻觉无显著降低效果", "提出 CHAIR-MEN 语义匹配扩展使 CHAIR 可泛化至 365 类大词表数据集", "在 O365/COCO 与 O365/non-COCO 上构建 OOD 幻觉评测协议消除训练数据污染"]
benchmarks: ["POPE", "CHAIR / CHAIR-MEN", "FaithScore", "CIDEr", "CLIPScore", "Objects365"]
---

# 论文速读：Does Object Grounding Really Reduce Hallucination of Large Vision-Language Models?

## 一句话总结
本文首次系统评估了在大规模视觉-语言模型（LVLM）训练中引入细粒度物体定位目标（指代表达 RE 和定位式图像描述 GC）是否能有效减少开放图像描述中的**物体幻觉**。结果显示，这些定位目标对幻觉几乎没有降低效果，质疑了学术界广泛持有的"定位→去幻觉"因果直觉。

## 研究问题与动机
1. **核心问题**：LVLM 在开放图像描述中普遍存在物体幻觉（生成图像中不存在的对象/属性），近期有工作（Shikra、Ferret、Kosmos-2 等）声称加入 grounding 目标可降低幻觉，但缺乏严谨实证支撑。
2. **现有评估的缺陷一**：已有 grounding 去幻觉的实证几乎全部基于 POPE 等**问答式（QA）**评测，而非更贴近真实应用的开放描述任务。
3. **现有评估的缺陷二**：幻觉度量所用数据集 MSCOCO 已被大量 LVLM 训练语料覆盖，很可能严重**低估**了模型在分布外场景下的真实幻觉水平。
4. **动机**：需要一套更接近"野外"生成场景、并使用 OOD 数据的系统评测协议来检验"定位→去幻觉"这一直觉假设。

## 核心贡献（创新点）
1. **首个系统性分析**：首次在开放图像描述任务上，以控制变量方式（同架构、同训练协议，仅变训练 mix）检验 RE 与 GC 两个 grounding 目标对幻觉的因果效应。
2. **CHAIR-MEN 扩展**：提出将 CHAIR 从基于字符串匹配升级为基于名词短语嵌入的语义匹配，解决其对更大类别集（如 Objects365 的 365 类）难以适配的问题。
3. **OOD 评测数据集构建**：在 MSCOCO 之外引入 Objects365，并新增 O365/COCO（80 类）与 O365/non-COCO（285 类）两套 POPE 测试集，消除训练数据污染偏倚。
4. **双重幻觉度量互补**：同时使用 CHAIR（基于对象检测）与 FaithScore（LLM 抽事实 + VQA 校验）两种机制，避免单一度量的 artifact 风险。
5. **反直觉实证结论**：跨 Vicuna / Llama-3 / Phi-3 三 Backbone 一致发现，RE/GC 训练目标几乎不降低幻觉；仅推理时强制生成本地坐标（GC-at-inference）有微小缓解，但以牺牲描述信息量为代价。

## 方法详解
- **Grounding 目标类型**
  - **Referring Expressions (RE)**：双向任务——给定文本描述生成对应 bbox，或给定 bbox 生成自然语言描述；使用 RefCOCO + Visual Genome 各 320k 样本。
  - **Grounded Captioning (GC)**：生成图像描述时，在文本中穿插对象的相对 bbox 坐标（如 `A cat [0.10, 0.05, 0.64, 1.00] on a mat`）；使用 Flickr30k-Entities 150k 样本。
- **Region 编码**：采用 relative 坐标作为"纯文本" token 输入（如 `[0.10, 0.05, 0.64, 1.00]`），避免引入额外可训练参数；多 bbox 用分号拼接。
- **模型架构与控制**：三 LLM 主骨架（Vicuna 1.5 7B、Llama-3 8B Instruct、Phi-3-mini）冻结权重，4-bit QLoRA 微调（r=64, α=128）；图像编码器为 OpenAI CLIP ViT-L/14-224；默认对齐模块为 2 层 MLP pooling（Chu et al., 2024），另实验 3 层 Perceiver Resampler（32 query tokens）。
- **训练 Mix（四种组合）**：Base（仅 caption + long caption + VQA yes/no）、+RE、+GC、+RE+GC；预训练阶段仅训练对齐模块（560k 样本），指令微调阶段训练整个 LoRA adapter。
- **CHAIR-MEN 公式**：对生成文本提取名词短语 → 用 Sentence-BERT 编码器得到嵌入 → 与图像中对象的类别名嵌入做余弦相似度；设两个阈值 $t_1 = 0.73$（正样本）与 $t_2 = 0.78$（负样本），分别归属到图中有/无对应类别或拒绝匹配；阈值通过使 MSCOCO 上的得分复现 vanilla CHAIR 校准。
- **FaithScore 实现**：用小规模 LLM 代替 GPT-4 抽取原子事实（atomic facts），再用 OFA（ViT-L VQA）作为事实校验器；报告事实数量作为信息量指标。
- **推理时 GC 策略**：同一 prompt 强制模型生成带坐标的描述，对比标准 caption。

## 实验与结果
- **数据集**：MSCOCO（5000 张测试集）+ Objects365（5386 张验证集）；POPE 新增 O365/COCO（1500 例，80 类）与 O365/non-COCO（1500 例，285 类），分 random / popular / adversarial 三种负样本策略。
- **表达定位能力（定位目标有效性自检）**：精度@50（Table 2），+RE+GC 相对于 +RE 在所有模型上均有大幅提升（例如 Phi-3：R+ 68.23 → 73.33；Rg 65.50 → 73.33），说明 RE 与 GC 两个目标在表征层面是**相容且互补**的。
- **QA 式幻觉（POPE，Table 1 摘选关键值）**：
  - Llama-3 Base (MSCOCO adv.) 75.83 → +GC 78.90 → +RE 79.93 → +RE+GC 79.93；差异**不一致且幅度很小**。
  - O365/non-COCO (adv.) 58.20 → +GC 60.37 → +RE 64.57 → +RE+GC 61.27；基线低（OOD 更难），但 grounding 目标同样未给出稳定下降。
  - Phi-3 / Vicuna 趋势相同：定位目标对 POPE 准确率没有一致的实质性改善。
- **开放描述幻觉（Table 3 摘要，CHAIR-MEN 指标）**：
  - 所有模型在 MSCOCO 与 O365 上的 CHAIR<sub>i</sub> 都在 3.3–5.2 区间内，Base 与 +RE/+GC/+RE+GC 之间差值**通常 < 0.5**，无显著差异。
  - FaithScore 同理（~90.5–92.2）；Caption 质量（CIDEr、CLIPScore、字数、Coverage、Objects 数）也基本持平，说明定位目标既不提升也不损害生成质量。
- **推理时强制 GC（Table 4，相对 Base 的绝对变化，Objects365 示例）**：
  - 幻觉略有下降（CHAIR<sub>i</sub> 降幅约 0.01–1.91），但同时 Word 数下降（-0.03 至 -2.37）、Facts 数下降（-0.06 至 -0.41），存在**信息量↔忠实度**的权衡。
- **强结果与提升幅度**：最强的是推理时强制 GC 对 O365 的轻微改善（如 Llama-3 +GC CHAIR<sub>i</sub> 从 3.84 降至 2.77，降幅约 1.07），但代价是覆盖率由 56.43 降至 50.80、Objects 均值 1.61→1.39。总体而言，提升幅度均 **< 3 个绝对点**。

## 相关工作脉络
1. **Shikra (Chen et al., 2023b)、Ferret (You et al., 2023)、Kosmos-2 (Peng et al., 2023)**：均为以 grounding 为核心训练目标的 LVLM 架构；本文承认它们在定位任务上表现优异，但指出其"定位→去幻觉"的泛化论断缺乏系统验证。
2. **POPE (Li et al., 2023b)**：当前 QA 式幻觉评测基准；本文不否定其价值，但强调它与开放描述的评估错位，并提出了 OOD 扩展（O365/COCO 与 O365/non-COCO）。
3. **CHAIR (Rohrbach et al., 2018)**：原版基于字符串匹配的开放描述幻觉指标；本文提出语义替代 CHAIR-MEN，使该指标可泛化到 365 类大词表场景。
4. **FaithScore (Jing et al., 2023)**：参考-free、基于 LLM+VQA 的事实一致性度量；本文将其与 CHAIR-MEN 联合使用，以缓解单度量的偏差风险。
5. **BLIP-2 (Li et al., 2023a)、LLaVA (Liu et al., 2023b)、InstructBLIP (Dai et al., 2023a)**：主流 LVLM 训练管线；本文对齐模块与预处理策略直接引用 Chu et al. (2024) 的 MLP pooling 与 LLaVA 式的 560k 预训练语料。
6. **RE/GC 训练目标本身**：RefCOCO/Visual Genome/Flickr30k-Entities 是经典的表达定位与定位描述语料；本文将其作为"训练 mix 中的一项任务"而非"全新架构"进行消融控制。

## 局限性与未来方向
1. **算力约束导致规模有限**：单卡 RTX 3090 上仅训练 2–4 GPU 天，数据集规模远小于工业级 LVLM（如 LLaVA、Qwen-VL），无法排除"更大规模下定位目标才显现去幻觉效果"的可能性。
2. **架构选择固定**：仅实验了 MLP pooling 与 Perceiver Resampler 两种对齐模块，未见其他变体（如 cross-attention、多尺度特征融合）。
3. **自动指标局限**：CHAIR-MEN 与 FaithScore 均为 imperfect 自动度量，尽管两者互为补充，仍可能有系统性偏差；人工评估仅少量定性样本。
4. **未探索更多定位目标**：仅关注 RE 与 GC 两类最流行目标，其他如 region-VQA、mask generation 等未纳入。
5. **未来方向**：需设计新的去幻觉机制（而非依赖定位目标）、更大规模验证定位-幻觉因果链、结合人类评估与细粒度属性/关系幻觉度量。

## 研究启发与可借鉴点
1. **控制变量的消融范式**：以"同架构、同预训练、仅变训练 mix"的方式对比 Base / +RE / +GC / +RE+GC，清晰剥离出单一目标的因果效应——这种"训练 mix 消融"设计值得在后续工作中复用。
2. **CHAIR-MEN 语义匹配扩展**：将 CHAIR 从字符串匹配升级为 embedding-based 语义匹配，并标定阈值使其复现原版在 MSCOCO 上的得分——为 CHAIR 迁移到更大词表（LVIS、OpenImages、O365）提供了可直接复用的方法。
3. **OOD 评测协议**：在 POPE 上额外构造 O365/COCO（80 类）与 O365/non-COCO（285 类），揭示基于训练集的评测会乐观低估幻觉；建议在幻觉评测中常态化引入 OOD 集。
4. **信息量↔忠实度的权衡量化**：通过同时报告 CHAIR / FaithScore（忠实度）与 #Words / Coverage / Objects / Facts（信息量），提醒社区避免只关注一面；后续研究应把双指标一并汇报。
5. **定位目标在表征层面有效 ≠ 在生成层面有效**：Table 2 显示 RE+GC 显著提升定位精度，但 Table 1/3 却未发现幻觉下降——这提示**表征能力提升到生成行为之间存在 gap**，可在本团队方向中作为研究假设验证。

## 关键术语表
- **LVLM (Large Vision-Language Model)**：融合视觉编码器与大语言模型的 multimodal 模型，如 LLaVA、BLIP-2、Qwen-VL 等。
- **Object Hallucination**：LVLM 在图像描述/QA 中生成图像内不存在的对象、属性或关系。
- **Referring Expressions (RE)**：给定图像区域描述生成对应 bbox（或反向），经典定位任务。
- **Grounded Captioning (GC)**：在图像描述中嵌入对象的 bbox 坐标，实现"文-图"局部对齐。
- **CHAIR / CHAIR<sub>i</sub> / CHAIR<sub>s</sub>**：基于 MSCOCO 80 类的图像描述幻觉指标；CHAIR<sub>i</sub> 是对象级幻觉率，CHAIR<sub>s</sub> 是图像级是否存在幻觉的二元比例。
- **CHAIR-MEN**：作者提出的语义化 CHAIR 变体，用名词短语嵌入余弦相似度替代字符串匹配。
- **FaithScore**：LLM 抽取原子事实后用 VQA 模型校验其真实性的参考-free 幻觉度量。
- **POPE**：基于 MSCOCO 标注的"Is there X in the image?" Yes/No 问答幻觉评测基准。
- **OOD (Out-of-Distribution)**：指测试数据（Objects365）与 LVLM 训练数据（以 MSCOCO 为主）不重叠，用于消除训练泄露造成的评估偏倚。
- **LoRA (Low-Rank Adaptation)**：在大模型微调中仅更新低秩投影矩阵以节省显存，本文 r=64、α=128。

## 可复现要素
- **数据集**：MSCOCO（公开）、Objects365（公开）、RefCOCO（公开）、Visual Genome（公开）、Flickr30k-Entities（公开）、VQAv2（公开）、LLaVA-Detailed（公开）。
- **代码开源**：**论文未提及**代码仓库链接。
- **权重开源**：**论文未提及**。
- **关键超参**：LoRA r=64, α=128；AdamW + cosine schedule；预训练 lr=1e-3, batch=32；指令微调 lr=2e-4, batch=16/32/64（按 Vicuna/Phi-3/Llama-3 不同）；重复惩罚 1.15；bbox 坐标保留 2 位小数。
- **硬件**：单卡 NVIDIA RTX 3090；训练耗时 2–4 GPU 天/模型。
- **LLM backbone**：Vicuna 1.5 7B、Llama-3 8B Instruct、Phi-3-mini；图像编码器 OpenAI CLIP ViT-L/14-224。
