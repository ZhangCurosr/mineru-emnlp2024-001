---
title: "Evaluating-the-Instruction-Following-Robustness-of-Large-Lan"
source: https://aclanthology.org/2024.emnlp-main.33.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:17:38"
---

# 论文速读：Evaluating-the-Instruction-Following-Robustness-of-Large-Lan

## 一句话总结
本文针对大语言模型在指令跟随能力上的优势可能引发的提示词注入安全风险，构建了一个基于抽取式QA任务的自动化评测基准，系统量化评估了8款主流开源与闭源模型的抗注入鲁棒性，并揭示了指令微调能力/模型规模与抗注入鲁棒性之间的非相关性。

## 研究问题与动机
1. **核心问题**：当LLM依赖外部检索内容（如网页片段、API返回）回答问题时，如何确保模型能准确区分并遵循用户原始目标指令，而非被其中嵌合的对抗性注入指令带偏？
2. **现有评测盲区**：当前指令跟随能力评测（如AlpacaEval、MT-Bench）主要关注Accuracy与人类偏好排名，缺乏针对Prompt Injection攻击的定量鲁棒性基准。
3. **现实应用威胁**：Bing Chat、ChatGPT插件及RAG系统已广泛集成检索能力，但第三方网页内容可能被攻击者预埋对抗指令，导致模型泄露隐私或执行非预期操作。
4. **认知误区**：业界普遍假设“参数量越大”或“指令跟随能力越强”的模型越安全，但本文证明二者与抗注入鲁棒性并不正相关，需重新审视模型对Prompt整体结构的理解与指令优先级判别机制。

## 核心贡献（创新点）
1. **构建首个专注指令遵循鲁棒性的Prompt Injection评测基准**：以抽取式QA为测试床，实现对抗样本的自动化生成与精准打分，与现有侧重自由文本生成或恶意输出泄露的评测形成本质区别。
2. **提出PDR与IDR双指标解耦评估框架**：Performance Drop Rate (PDR) 量化注入对原任务准确率的侵蚀程度，Instruction Discrimination Rate (IDR) 直接衡量模型对原始指令与注入指令的优先排序倾向，填补了“误跟随”行为的可解释评估空白。
3. **揭示规模/指令能力与抗注入鲁棒性的背离现象**：通过对比8款模型证明，AlpacaEval高分的Zephyr-7B在注入攻击下表现极弱，而中等规模的Vicuna-33B反而更具鲁棒性，打破“越大越强”的直觉假设。
4. **系统性剖析攻击位置、指令类型与防御策略的交互效应**：验证“末尾注入”最难防御、上下文相关注入比无关注入更易成功，且强上下文理解模型反而更容易被“忽略上文”类劫持短语攻破，为安全对齐研究提供明确的失效模式图谱。

## 方法详解
1. **任务设定**：采用开放式抽取式QA。给定用户问题 $q$ 与包含对抗指令 $q'$ 的搜索上下文 $c$，模型需基于 $c+q'$ 生成答案 $a$。共选用 NaturalQuestions、TriviaQA、SQuAD、HotpotQA，各取1000个dev样本。
2. **对抗指令构造**：SQuAD自带多QA对，直接复用另一对作为 $(q', a')$；其余三数据集每段仅有一个QA对，使用GPT-4基于相同上下文 $c$ 生成一个与原问题不同但答案仍在 $c$ 中的替代QA对 $(q', a')$，确保注入指令具有上下文连贯性且可自动评估。
3. **核心指标公式**：
   - 基础准确率：$\operatorname{Acc}(f) = \frac{1}{|\mathcal{D}_{\text{test}}|}\sum v(f(q,c), a)$
   - 对抗准确率：$\operatorname{Adv}(f) = \frac{1}{|\mathcal{D}_{\text{test}}'|}\sum v(f(q, c+q'), a)$
   - **性能下降率 (PDR)**：$\operatorname{PDR}(f) = \frac{\operatorname{Acc}(f) - \operatorname{Adv}(f)}{\operatorname{Acc}(f)}$，值越大表示越易受注入影响。
   - 注入任务准确率：$\operatorname{Adv}'(f) = \frac{1}{|\mathcal{D}_{\text{test}}'|}\sum v(f(q, c+q'), a')$
   - **指令判别率 (IDR)**：$\operatorname{IDR}(f) = \frac{\operatorname{Adv}(f)}{\operatorname{Adv}(f) + \operatorname{Adv}'(f)}$，值越高表明模型越倾向于遵循原始目标指令 $q$ 而非注入指令 $q'$。
4. **提示词模板与防御基线**：默认System prompt显式声明“忽略 `<context>` 标签内的指令”，并配合4-shot示例。输入结构为
