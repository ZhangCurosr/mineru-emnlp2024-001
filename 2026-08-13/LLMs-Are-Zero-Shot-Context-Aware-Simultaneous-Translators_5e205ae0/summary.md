---
title: "LLMs-Are-Zero-Shot-Context-Aware-Simultaneous-Translators"
source: https://aclanthology.org/2024.emnlp-main.69.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:22:58"
field: "零样本同时机器翻译"
keywords: ["simultaneous machine translation", "zero-shot translation", "large language models", "context-aware translation", "response priming", "background information injection"]
innovations: ["零样本使用指令微调 LLM 执行 SiMT，无需微调与复杂分段策略", "轻量背景信息注入显著提升技术与术语密集场景翻译质量", "响应预激进机制约束输出空间并稳定提升多语言 BLEU"]
benchmarks: ["TED-TST-2024", "FLEURS", "AMBIEVAL"]
---

# 论文速读：LLMs-Are-Zero-Shot-Context-Aware-Simultaneous-Translators

## 一句话总结
本文证明**开箱即用的指令微调大语言模型（LLM）可在零样本设置下执行同时机器翻译（SiMT）**，无需昂贵微调或专门数据集；通过注入轻量背景信息可大幅提升翻译质量，尤其在技术主题上表现突出，为构建大规模多语言、上下文感知且术语准确的下一代 SiMT 系统提供了可行路径。

## 研究问题与动机
- **现有 SiMT 系统局限于句级翻译**，忽略前文建立的上下文以及隐含的语外信息（如演示文稿、术语表），导致长文本逻辑不一致、术语不统一。
- **人类同传译者 routinely 依赖背景知识**（术语表、主题准备），但现有 SiMT 系统不具备此类能力。
- **之前尝试将 LLM 用于 SiMT 的工作依赖微调**（如 TRANSLLAMA、Simul-LLM），需构造专用数据集并搜索超参，资源消耗大。
- **开源 LLM 的推理、翻译与上下文学习能力**已接近或超越专用 MT 模型，值得探索其在零样本 SiMT 中的潜力。

## 核心贡献（创新点）
1. **零样本 SiMT 可行性证明**：开箱即用的 Llama-3-70B-Instruct 无需微调即可在多个语言对上达到与 SOTA 基线相当甚至更优的 BLEU 与延迟指标。
2. **轻量背景信息注入机制**：在 system prompt 中加入主题与命名实体描述（JSON 格式），可显著提升翻译质量，尤其在技术/专业内容上；小模型（8B）则难以从该机制中受益。
3. **响应预激进（Response Priming）**：将部分目标翻译预先写入 assistant 角色而非 user 角色，有效约束输出空间，避免模型产生不必要的注释或道歉，稳定提升各语言对 BLEU。

## 方法详解
- **级联架构**：音频 chunk（200 ms）→ 在线 ASR（whisper-small.en）→ 文本缓冲 → LLM（Llama-3-70B-Instruct）翻译。
- **READ/WRITE 策略**：生成完整词时执行 WRITE（将新源词与新译词追加至 prompt）；生成 `<|eot_id|>` 时执行 READ（仅追加新源词）；设定最小音频长度阈值以控制延迟–质量权衡。
- **响应预激进**：partial target 置于 assistant 段，限制生成序列空间，防止模型输出额外解释。
- **背景信息注入**：system message 中包含 JSON 格式的 topic 与 named_entities（含 entity 与 description），由 gpt-4-turbo 基于整篇 TED 演讲生成。
- **推断公式**：$p(y_t \mid y_{<t}, x_{\le t}, b)$，其中 $b$ 为跨句子不变的背景信息。
- **工程优化**：使用 vllm 库（张量并行）与 whisper-jax 实现快速推理；温度设为 0（贪婪解码）。

## 实验与结果
- **数据集**：TED-TST-2023、自建 TED-TST-2024（避开预训练泄露）、FLEURS、AMBIEVAL（歧义术语数据集）。
- **基线**：SEAMLESSSTREAMING、NAIST、FBK、TRANSLLAMA。
- **核心结果（en-de，LAAL≈2000 ms）**：
  - TED-TST-2024：Ours BLEU=32.30，AL=1720.00，LAAL=2022.05；优于 SEAMLESS（31.75）与 TRANSLLAMA（25.71）。
  - FLEURS：Ours BLEU=42.60，LAAL=2008.48；略低于 NAIST（39.80）但延迟更优。
  - AMBIEVAL：Ours BLEU=42.60，大幅领先所有基线。
- **多语言表现**：在 en-es/fr/it/ru 上亦取得竞争力结果（Table 5、6）。
- **消融**：
  - 移除 response priming 导致所有语言对 BLEU 下降（约 1–2 分）。
  - 移除背景信息导致 BLEU 下降约 5–6 分（70B 模型）；8B 模型无明显收益。
- **ASR 纠错**：LLM 能自动纠正 whisper 错误（如 "bala"→"Hezbollah"），优于 NLLB-200。
- **推理速度**：RTF=0.86（4×A100 80GB），满足实时翻译要求（RTF≤1）。

## 相关工作脉络
- **NAIST / FBK / SEAMLESSSTREAMING**：专用 SiMT 系统，需复杂策略（wait-k、local agreement）与微调；本文零样本无需此类设计。
- **TRANSLLAMA（Koshkin et al., 2024）**：同样使用 LLM 但需微调因果对齐数据；本文证明零样本即可达到可比性能。
- **Simul-LLM（Agostinelli et al., 2024）**：微调 LLM + sophisticated segmentation policy；本文避免微调与策略搜索。
- **NLLB-200（离线 MT）**：作为质量参考基线，但在 ASR 错误恢复与上下文适应上不及 LLM。
- **Prefix-alignment / Meaningful-unit fine-tuning**：针对特定语言对的微调范式；本文强调零样本与多语言通用性。

## 局限性与未来方向
- **API 闭源模型兼容性**：GPT-4/Claude 等通过 API 访问时受限于固定 prompt 结构，不支持 response priming。
- **8B 小模型性能不足**：无法有效利用背景信息，指令遵循能力较弱。
- **级联架构的 ASR 幻觉**：在低延迟条件下 whisper 可能生成不存在词汇；未来需端到端 speech-to-text-to-translation 系统。
- **背景信息生成依赖强模型**：当前使用 gpt-4-turbo，可探索更轻量方法。
- **潜在优化方向**：local agreement、weight quantization（如 awq）、更 sophisticated prompting。

## 研究启发与可借鉴点
- **响应预激进可迁移**：任何需约束 LLM 输出格式的生成任务（如代码补全、结构化输出）均可借鉴此技巧。
- **零样本 SiMT 验证流程**：无需微调即可快速评估 LLM 在多语言 Streaming 场景的潜力，适合前期可行性研究。
- **背景信息工程**：轻量 JSON 格式的上下文注入机制可复用至术语约束翻译、领域适配等任务。
- **延迟–质量权衡控制**：通过最小音频长度阈值间接调节 WER 与延迟，设计简洁且有效。
- **消融设计**：分别验证 priming 与 background info 的贡献，为后续改进提供清晰归因。

## 关键术语表
- **Simultaneous Machine Translation (SiMT)**：边接收源语边生成译文的实时翻译范式，需平衡翻译质量与延迟。
- **Response Priming**：将部分目标文本预置入 assistant prompt，约束模型生成空间以避免冗余输出。
- **READ/WRITE Policy**：SiMT 中决定何时读取新源词（READ）与输出生成词（WRITE）的调度策略。
- **Average Lagging (AL) / Length-Adaptive AL (LAAL)**：衡量 SiMT 延迟的指标，LAAL 对句子长度归一化。
- **Background Information Injection**：在 system prompt 中嵌入主题/术语描述，辅助 LLM 做出上下文一致的术语选择。
- **TER (Term Error Rate)**：评估术语准确性的常用指标；本文未直接报告但 AMBIEVAL 数据集隐含此需求。
- **Cascaded ASR-LLM Pipeline**：先经离线/在线 ASR 转文本，再交由 LLM 翻译的级联架构。
- **Zero-shot SiMT**：无需针对 SiMT 任务微调，直接利用 LLM 固有能力进行实时翻译。

## 可复现要素
- **数据集**：TED-TST-2023（公开）、TED-TST-2024（作者提供）、FLEURS（公开）、AMBIEVAL（作者提供）。
- **代码**：https://github.com/RomanKoshkin/toLLMatch（已开源）。
- **模型权重**：Llama-3-70B-Instruct（需申请访问）；whisper-small.en（Hugging Face 开源）。
- **关键超参**：音频 chunk 长度 200 ms；最小写入阈值 1.2–1.8 s；温度 0；vllm 张量并行。
- **硬件**：4×A100 80GB GPU。
