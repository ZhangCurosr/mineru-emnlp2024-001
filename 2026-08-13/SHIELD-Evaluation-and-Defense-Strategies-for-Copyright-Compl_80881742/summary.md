---
title: "SHIELD-Evaluation-and-Defense-Strategies-for-Copyright-Compl"
source: https://aclanthology.org/2024.emnlp-main.98.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:29:25"
field: "大语言模型安全与合规"
keywords: ["LLM版权合规", "jailbreaking攻击", "防御机制", "Agent-based防御", "版权评估基准", "拒绝率"]
innovations: ["首次系统性评估jailbreaking攻击对LLM版权防护的绕过效果", "提出轻量级Agent-based实时防御机制无需修改模型参数", "构建覆盖多版权状态的五维数据集及拒绝率综合评测协议"]
benchmarks: ["BS-C", "BS-PC", "BS-NC", "SSRL", "BEP"]
---

# 论文速读：SHIELD-Evaluation-and-Defense-Strategies-for-Copyright-Compl

## 一句话总结
本文提出 SHIELD 框架，首次系统性地评估大语言模型在版权合规方面的表现：构建了包含不同版权状态文本的评测数据集，首次引入 jailbreaking 攻击测试版权防护的鲁棒性，并提出一种轻量级 Agent-based 防御机制，可在不修改模型参数的情况下显著降低受版权保护文本的生成量。

## 研究问题与动机
1. **当前 LLM 面临"侵权"与"过度保护"双重困境**：由于各国版权法差异复杂，模型可能在某些场景下生成受版权保护的文字（侵权），也可能拒绝生成本应允许的公共领域内容（过度保护）。
2. **缺乏全面、多维度的版权合规评测基准**：既有工作仅关注文本相似度指标，缺少涵盖多类型版权状态（受保护/公共领域/部分国家保护）的数据集及拒绝率（Refusal Rate）等综合评估体系。
3. **版权防护机制缺乏攻击鲁棒性评估**：尚无工作系统研究 jailbreaking 攻击对版权保护机制的绕过能力，现有模型的安全防护存在明显漏洞。
4. **已有防御手段存在局限**：机器遗忘（Unlearning）导致信息损失，对齐方法引发过度保护，MemFree 解码可能产生幻觉，且这些方法通常需要模型参数访问权限，不适用于 API 模型。

## 核心贡献（创新点）
1. **构建了首个覆盖多版权状态的系统化评测数据集（BS-C/BS-PC/BS-NC/SSRL/BEP）并人工核验**：与仅用 LCS 评估记忆程度的既往工作不同，本文同时评估模型"能否生成公共领域文本"和"是否拒绝受版权内容"两个维度。
2. **首次系统评估 jailbreaking 攻击对版权防护机制的绕过效果**：引入了 76 种现有 jailbreak 攻击模板，证明简单提示工程即可显著提升受版权文本生成量，揭示了当前模型防护的脆弱性。
3. **提出轻量级 Agent-based 实时防御机制（无需训练/微调）**：与 MemFree 等修改解码过程的方案相比，本文方案通过检测+网络验证+引导拒绝三步流程实现防御，不干扰非版权相关请求，且避免幻觉问题。

## 方法详解
**SHIELD 防御机制由三个核心组件构成：**

- **Copyright Material Detector（版权材料检测器）**：基于 N-Gram 语言模型，对每个受版权保护的素材 c 训练一个 N-Gram 模型 $P_c$。给定输入或生成文本 T，计算任意子串 $T_s$（长度 > $N_T$）被判定为版权文本的概率 $P(T_s|c)$，若超过阈值 θ（实验设为 0.5）且连续命中 5 次，则判定为包含版权材料。

- **Copyright Status Verifier（版权状态验证器）**：调用网络服务验证检测到的版权材料的实际状态。首先查询 Project Gutenberg 数据库判断是否为公共领域；若未找到，则调用 Perplexity AI（LLaMA-3-Sonar 模型）在线搜索版权状态并以 JSON 格式返回结果。支持异步调用和 TTL 缓存以保证实时响应。

- **Copyright Status Guide（版权状态引导器）**：根据验证器输出，利用 In-context Few-shot 示例引导 LLM 生成合适响应——若检测到版权材料，指导模型明确拒绝并给出警告；若为公共领域材料或无版权相关内容，则让模型正常生成。

**评测协议设计：**
- 五个数据集：BS-NC（100本非版权书籍）、BS-C（50本受版权书籍）、BS-PC（20本部分版权书籍）、SSRL（100首Spotify热门歌词）、BEP（100首经典英文诗歌）
- 三种提示类型：Prefix Probing、Direct Probing、Jailbreaking
- 关键指标：LCS（最长公共子串长度）、ROUGE-L 分数、Refusal Rate（拒绝率）

## 实验与结果
**评测模型**：API 模型包括 Claude-3 Haiku、Gemini Pro、Gemini 1.5 Pro、GPT-3.5 Turbo、GPT-4o；开源模型包括 LLaMA-2 7B、LLaMA-3 8B、Mistral 7B。

**核心发现**：
1. **当前模型普遍存在版权问题**：GPT-3.5 Turbo 和 Gemini Pro 在 Direct Probing 下 LCS 均值分别达 17.80 和 11.98，LLaMA-3 为 12.00，表明大量受版权文本被直接生成。
2. **Claude-3 过度保护**：BEP 和 BS-NC 数据集上拒绝率高达 75-81%，会拒绝生成公共领域文本。
3. **Jailbreaking 攻击有效**：对 GPT-4o 和 Claude-3 而言，jailbreak 使拒绝率从 80-100% 下降至 20-22%，最大 LCS 显著提升（Claude-3 从 8 升至 128）。
4. **SHIELD 防御效果显著（表3）**：对所有 API 模型将拒绝率提升至接近 100%，LCS 从均值 5-10+ 降至 1.9-2.4；对开源模型同样有效（LLaMA-3 拒绝率从 5% 提升至 95%）。
5. **SHIELD vs MemFree**：MemFree 仅降低复制文本量但不改变拒绝率；SHIELD 在拒绝率和复制量控制上全面优于 MemFree，且不产生幻觉。

## 相关工作脉络
1. **Chang et al. (2023)** 使用 cloze probing 评估 LLM 对版权文本的记忆程度；本文扩展至完整生成场景，增加了拒绝率和攻击鲁棒性评估，从"是否记忆"深入到"是否合规生成"。
2. **Karamolegkou et al. (2023)** 首次用 LCS 量化 LLM 与原文本的相似性；本文在此基础上构建了多维度评测体系，新增了五种不同类型数据集和拒绝率指标。
3. **Ippolito et al. (2023, MemFree)** 使用 N-Gram 修改解码过程以防止逐字复制；本文保留了 N-Gram 检测但改为 Agent-based 方法，避免了解码修改导致的幻觉问题，且无需模型参数访问。
4. **Wei et al. (2024)** 同步工作评估了版权下架方法；本文更系统地涵盖了从评测基准、攻击鲁棒性到防御机制的完整链条。
5. **Liu et al. (2024b)** 的 jailbreak 模板库被首次引入版权保护场景进行系统性鲁棒性评估，打开了"版权防护对抗攻击"的新研究方向。

## 局限性与未来方向
1. **N-Gram 检测器可能被相似文本误导**：无法识别不在数据库中的版权材料，需要持续更新版权素材库。
2. **未实现真正的实时检测**：当前在生成前或生成后检测，而非与生成过程并行滑动窗口检测，实际部署中响应时间可能较长。
3. **仅限英文材料**：未测试代码、技术书籍或其他语言（中文、西班牙文等）的版权合规问题。
4. **部分版权（BS-PC）处理方案未定**：不同国家版权状态差异复杂的场景如何处理仍有争议，需结合地理定位等技术。
5. **拒绝率计算基于简单模式匹配**：可能不够精确，且未提供缓解过度保护公共领域文本的方案。
6. **数据集规模相对有限**：相比生产环境使用的数据量较小，且版权状态声明仅在写作时有效，需定期更新。

## 研究启发与可借鉴点
1. **Refusal Rate 指标的引入值得借鉴**：将安全领域的拒绝率评估范式引入版权保护场景，实现了"既能拒绝有害内容，又能正常生成合规内容"的双目标评估，方法论可迁移至其他合规性评测。
2. **"检测→验证→引导"的 Agent-based 防御架构可复用**：不修改模型权重、不依赖模型内部访问，仅通过工具调用和提示工程实现防御，适用于所有 API 模型，部署成本低。
3. **结合网络搜索实现动态版权状态验证**：利用 Project Gutenberg + Perplexity AI 的组合验证版权状态，解决了版权法随时间和地域变化的动态性问题，思路可扩展到其他法律合规领域。
4. **Jailbreaking 攻击评估版权防护的框架可推广**：首次将对抗攻击评估引入版权保护领域，揭示了表面安全防护的脆弱性，为后续研究提供了可复用的攻击-防御评测范式。
5. **使用 Pair 自动进化攻击缓解过度保护的思路**：论文在附录中展示了利用自动化 jailbreak 方法（Pair）来减少对公共领域内容的过度保护，为"防御与可用性平衡"问题提供了新思路。

## 关键术语表
**SHIELD**：System for Handling Intellectual Property and Evaluation of LLM-Generated Text for Legal Defense，本文提出的综合性版权合规评测与防御框架。
**Refusal Rate（拒绝率）**：模型拒绝生成受版权保护文本的响应比例，衡量模型合规能力的关键指标。
**Prefix Probing**：输入受版权文本的前 50 个词，让模型续写，用于评估模型隐式复制版权内容的倾向。
**Direct Probing**：直接要求模型生成指定版权文本（如"请提供某书前100词"），测试模型的主动合规能力。
**MemFree**：基于 N-Gram 的解码时干预方法，通过修改 logits 防止生成逐字复制的版权文本。
**Jailbreaking Attack（越狱攻击）**：通过精心设计的提示绕过模型安全限制的攻击方法，本文首次将其引入版权保护评估。
**BS-PC（Partially Copyrighted）**：在部分国家受版权保护而在其他国家为公共领域的文本集合，反映跨国版权差异。
**N-Gram Language Model**：基于 n-gram 统计的语言模型，本文用于检测生成文本是否与已知版权素材高度相似。

## 可复现要素
- **数据集**：BS-C、BS-PC、BS-NC、SSRL、BEP 共五个数据集，**未公开**，仅应研究需求可申请获取（见 Appendix I）。
- **代码/权重**：论文未提供开源代码，SHIELD 防御机制为原型实现。
- **关键超参**：N-Gram 阈值 θ=0.5，连续命中 5 次判定为版权文本，训练 10-gram 模型；temperature=0。
- **测试模型**：5 个 API 模型（Claude-3 Haiku、Gemini Pro、Gemini 1.5 Pro、GPT-3.5 Turbo、GPT-4o）和 3 个开源模型（LLaMA-2 7B Chat、LLaMA-3 8B Instruct、Mistral 7B Instruct）。
- **Jailbreak 模板**：采用 Liu et al. (2024b) 的 76 个单轮攻击模板（见 Appendix H）。
