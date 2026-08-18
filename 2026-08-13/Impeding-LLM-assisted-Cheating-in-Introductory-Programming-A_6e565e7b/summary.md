---
title: "Impeding-LLM-assisted-Cheating-in-Introductory-Programming-A"
source: https://aclanthology.org/2024.emnlp-main.27.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:21:24"
field: "AI赋能教育的安全与诚信"
keywords: ["LLM作弊防御", "对抗扰动", "编程教育", "黑盒攻击", "学术诚信", "代码生成"]
innovations: ["首次系统性将黑盒对抗扰动应用于教育场景阻止LLM辅助作弊", "提出Efficacy度量结合SHAP代理模型指导最小化扰动选择", "用户研究揭示未察觉细微扰动保持高防御效力的关键发现"]
benchmarks: ["CS1/CS2编程作业集(58题)", "5个LLM基线性能评估(GPT-3.5/Copilot/Mistral/CodeLlama/CodeRL)"]
---

# 论文速读：Impeding-LLM-assisted-Cheating-in-Introductory-Programming-A

## 一句话总结
本文针对计算机入门课程中学生利用LLM作弊的问题，提出并评估了多种对抗性扰动技术，通过修改编程作业题目描述来降低LLM生成正确代码的能力；实验表明，组合扰动可使LLM生成解决方案的平均正确率降低77%，且微小的未察觉扰动最具实际防御潜力。

## 研究问题与动机
- **核心问题**：教育者如何在不依赖检测技术的前提下，通过修改作业提示词来阻碍学生利用ChatGPT等LLM工具直接复制生成代码？
- **检测方法的局限性**：现有LLM生成内容检测工具可靠性不足，存在误报和漏报风险，无法作为有效防线。
- **工业级LLM不可控**：教师无法直接控制或限制学生可访问的ChatGPT、GitHub Copilot等商业工具的底层能力，因此只能从"输入端"（题目本身）入手。
- **学术诚信紧急威胁**：已有研究表明LLM能成功解决CS1/CS2级别的编程问题，传统闭卷作业形式面临严峻挑战。

## 核心贡献（创新点）
1. **首次系统性研究黑盒对抗扰动在学术诚信场景的应用**——与现有对抗攻击研究不同，本文聚焦教育领域的防作弊需求，而非模型鲁棒性评估。
2. **设计10种面向编程作业题目的对抗性扰动技术**——包括同义词替换、句子删除、Unicode伪装字符替换等，并定义"Efficacy"量化指标评估扰动效果。
3. **引入SHAP+代理模型指导扰动选择**——利用CodeRL作为代理模型计算Shapley值来识别对LLM输出影响最大的token/句子，实现"效果-改动量"的最优权衡。
4. **设计并执行用户研究验证扰动在实际作弊场景中的有效性**——发现未察觉的细微扰动（如token替换、字符删除）保持高Efficacy，而高变化扰动即使被发现也难以被学生有效反转。

## 方法详解
**整体框架**分为三阶段：测量LLM基准性能 → 设计并评估扰动技术 → 用户研究验证。

**扰动技术设计**：

1. **核心扰动（7种）**：
   - **Token (remove)**：基于SHAP值排序，移除最重要的5个token
   - **Character (remove)**：对顶部5个token各随机删除一个字符
   - **Token (synonym)**：用GPT-3.5生成的上下文中文化同义词替换top-5 token（替换所有出现位置）
   - **Sentence (remove)**：顺序删除占总句数约1/3的句子
   - **Sentence (rephrase)**：基于SHAP累积值排序句子，删除top-3句子后用GPT-3.5改写
   - **Token (unicode)**：将top-5 token中的字符替换为视觉相似的Unicode Lookalike字符
   - **Random (insert)**：在top-5 token中插入冗余字符（连字符、下划线）

2. **探索性扰动（3种）**：
   - **Tokens (synonym)**：仅替换top-5 token中SHAP值最高的出现位置（非全局替换）
   - **Prompt (unicode)**：对整个题目文本进行全量Unicode Lookalike替换
   - **Random (replace)**：手动识别题目中的文件名/函数名/类名并替换为随机字符串（代码中再做反向映射以维持评分兼容）

**Efficacy定义**：
$$
Efficacy = max\Big\{0, 100 \times \frac{S_{no\_prtbr} - S_{prtbr}}{S_{no\_prtbr}}\Big\}
$$
其中$S$为正确率分数。该定义 favors 低绝对分数下的下降（如从70%降至40%比从100%降至70%更有价值）。

**SHAP代理模型机制**：使用CodeRL作为surrogate model，对每个token计算Shapley additive explanations值，用于识别对模型预测贡献最大的语言单元，指导扰动目标的选择。

## 实验与结果
**数据集**：
- 亚利桑那大学CS1（30题）和CS2（28题）课程编程作业
- 过滤后：84个short problem（单函数/类实现）、22个long problem（多函数交互/未指定数量）
- 自建测试Oracle（复用教师测试用例）

**评估模型**：GPT-3.5 (gpt-3.5-turbo-0301)、GitHub Copilot (Codex)、Mistral (mistral-large-2402)、Code Llama (7b-instruct)、CodeRL

**LLM基线性能**：
- **CS1**：所有5个模型在所有题目上均未生成完全正确的解（部分因触发学术诚信保护拒绝回答）
- **CS2**：GitHub Copilot表现最优，short问题平均正确率51.47%，long问题26.99%；CodeRL最差（short 12.47%，long 0%）
- **关键发现**：多函数交互类long problem显著更难，所有模型在CodeRL上long problem正确率为0%

**扰动Efficacy（Table 2）**：
- **Combined扰动**覆盖：CodeRL 93.75%题目、Code Llama 100%、Mistral 100%、GPT-3.5 97.14%、GitHub Copilot 90.91%的题目实现>80% Efficacy
- **最有效单技术**："Sentences (remove)"和"Prompt (unicode)"但伴随较高edit distance（可检测风险）
- **Finding 3**：生成解的多样性（unique variations）与扰动成功率强相关（失败组平均13.9 vs 成功组26.0）

**用户研究（30名已修课本科生，Table 4-6）**：
- 基准正确率71.28%，**组合扰动后平均Efficacy达76.67%（约77%下降）**
- **未被察觉的细微扰动**（如Character remove、Token unicode）保持高Efficacy（16%-43.75%），而高变化扰动（Prompt unicode）即使被发现仍能保持35.71% Efficacy
- **Finding 5**：学生发现扰动后未必能成功反转（32/49案例仍依赖ChatGPT绕过）
- 最有效的防御策略组合：Update problem statement（31.11% Efficacy）> No unusualness found（15.43%）> Expected to be bypassed（9.17%）

## 相关工作脉络
1. **Finnie-Ansley et al. (2022, 2023)**：首次实证Codex可解决CS1/CS2编程问题，揭示LLM作弊威胁——本文在此基础上提出主动防御而非被动检测路径。
2. **Wang et al. (2023a) "On the robustness of ChatGPT"**：评估ChatGPT对抗样本鲁棒性，属通用NLP安全领域——本文首次将对抗扰动定位到教育防作弊这一具体垂直场景。
3. **Bielik & Vechev (2020) "Adversarial robustness for code"**：白盒代码鲁棒性研究——本文聚焦黑盒setting，不依赖模型内部梯度信息。
4. **Sadasivan et al. (2023)**：指出LLM生成文本可轻易规避当前检测器——强化了对"题目设计防御"而非"检测结果判定"路线的需求。
5. **Wermelinger (2023) / Jesse et al. (2023)**：观察Copilot/Codex输出质量不稳定——本文将此现象转化为可利用的防御机制（扰动放大敏感性）。
6. **Boucher & Anderson (2022) "Bad Characters"**：研究不可见NLP对抗攻击——本文借鉴Unicode lookalike技术，但应用于题目prompt而非恶意注入。

## 局限性与未来方向
- **用户研究对象偏差**：实验对象为已修课学生（比目标群体更能识别/反转扰动），结果可能保守估计了实际效果。
- **扰动可理解性风险**：高变化扰动可能影响学生理解，需教师权衡防御效果与题目清晰度。
- **LLM快速演进**：GPT-4.0在CS1问题上（15.71% short / 13.11% long）明显优于GPT-3.5（均为0%），且缺乏学术诚信guardrails，未来扰动技术可能被更强模型绕过。
- **代理模型近似误差**：CodeRL作为surrogate model与目标模型存在能力差距。
- **用户研究规模有限**：仅6道题目、30名参与者，推广性存疑。
- **未来方向**：设计"不影响诚实学生理解但阻碍LLM"的扰动原则、结合prompt engineering反向验证、追踪模型演进下的扰动持久性、探索无需人工干预的自动化扰动生成管线。

## 研究启发与可借鉴点
1. **"Efficacy vs. Detectability"权衡框架**可直接迁移至其他AI滥用防御场景（如AI辅助写作、AI辅助数学解题），通过量化"扰动效果-人类可检测性"双轴来评估防御策略。
2. **SHAP引导的扰动选择策略**：利用可解释性方法（LIME、SHAP）识别关键输入单元，再施加最小改动——此范式可复用于其他黑盒LLM防护场景。
3. **用户研究的"逆向保守性"设计**：选择有经验的学生而非新手，提供效果的保守下界——这一伦理与科学兼顾的实验设计思路值得推广。
4. **AST相似度衡量生成解多样性的方法**：用AST比较而非字符串比较来判断LLM输出变异性，可迁移到对抗攻击通用评估中。
5. **题目"多函数交互"作为天然防御**：发现long problem（多函数协作）显著难被LLM直接解答，提示课程设计者可从问题结构层面增强抗作弊能力。

## 关键术语表
**Adversarial Perturbation**：通过对输入施加微小、特定的修改，导致模型输出显著错误，此处应用于修改编程作业题目描述以降低LLM解题成功率。
**Efficacy（扰动效力）**：扰动前后LLM正确率下降的百分比，是本文衡量扰动防御效果的核心指标。
**SHAP (Shapley Additive exPlanations)**：基于博弈论的可解释性方法，计算每个输入token对模型预测的贡献值，用于指导扰动目标选择。
**Short Problem vs Long Problem**：前者要求实现单个明确指定的函数/类；后者涉及多函数/类交互或未指定数量的组件，显著更难被LLM直接解答。
**Unicode Lookalike Substitution**：将普通字符替换为视觉上几乎相同但编码不同的Unicode字符（如a→à），不改变人类阅读体验但破坏LLM分词/表示。
**Test Oracle**：包含预设测试用例的脚本，用于自动评估学生或LLM生成代码的正确性得分（通过率百分比）。
**Blackbox Setting**：仅能通过输入-输出交互评估/攻击模型，无法获取内部梯度或参数信息，符合实际防作弊场景（教师无法访问模型内部）。
**Surrogate Model**：用易用的代理模型（如CodeRL）近似目标模型行为，用于计算Shapley值指导扰动，替代对目标模型本身的梯度计算。

## 可复现要素
- **数据集**：亚利桑那大学CS1/CS2编程作业（58题，含自制测试Oracle），论文未公开原始数据集，但说明"replication package, which includes the data and source code, will be available to researchers on request"。
- **代码**：源代码和复现包应研究者请求提供（论文未公开开源仓库）。
- **模型版本**：GPT-3.5 (gpt-3.5-turbo-0301)、Code Llama (7b-instruct via Ollama)、Mistral (mistral-large-2402)、CodeRL (HuggingFace)、GitHub Copilot (Codex)。
- **关键超参**：temperature=0（确定性生成）、输出token上限设为最大值、CodeRL top-5 token扰动、句子删除比例约1/3。
- **用户研究**：30名参与者、每题至少3人覆盖、补偿$20，IRB已批准。
