---
title: "Large-Language-Model-as-an-Assignment-Evaluator-Insights-Fee"
source: https://aclanthology.org/2024.emnlp-main.146.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:10:04"
field: "LLM评估与教育应用"
keywords: ["LLM-based Evaluator", "Prompt Hacking", "Automated Grading", "Education AI", "Instruction Following", "Goal Hijacking"]
innovations: ["首次千人大规模LLM评分器真实课堂部署实证", "系统性揭示学生群体中的prompt hacking行为及检测方法", "四种LLM TA部署方案的系统对比与可接受性调查"]
benchmarks: ["LLM TA学生 survey (N=838)", "Six-course real-world assignment evaluation"]
---

# 论文速读：Large-Language-Model-as-an-Assignment-Evaluator-Insights-Fee

## 一句话总结
本文报告了在拥有1,028名学生的大型大学课程中使用GPT-4作为自动作业评分器（LLM TA）的首次大规模实证经验，发现当学生可自由访问时LLM评分器总体可接受（75%），但同时暴露了严重的指令遵循缺陷和提示词黑客攻击风险。

## 研究问题与动机
- **真实课堂应用研究空白**：LLM-based evaluators已在学术研究中广泛应用（如自动评分与人类评分一致性研究），但鲜有工作探索将其用于真实课堂环境对学生作业进行评分，NLP社区缺乏对落地障碍的理解。
- **缺乏学生视角的实证数据**：已有研究（如G-eval、MT-bench等）主要关注LLM评分结果与人类评分的一致性，不涉及学生实际使用体验、态度和对LLM评分器的操控行为。
- **教学部署方案选择困境**：LLM TA的部署涉及LLM选型、是否向学生开放、费用承担、教师评分 vs 学生评分等多个设计选项，缺乏系统比较和实证指导。
- **提示词黑客风险未被充分研究**：LLM安全领域聚焦于生成有害内容的jailbreak攻击，但对"仅在特定上下文中不安全"（如评分器被欺骗给出满分）的prompt hacking缺乏关注。

## 核心贡献（创新点）
- **首次千人大规模实证报告**：在1,028名学生的课程中对6次作业使用LLM TA自动评分，是已知最大规模的LLM评分器真实课堂部署报告。
- **四种LLM TA部署方案的系统对比与可接受性调查**：设计了"不可访问/付费+教师评分/免费+教师评分/免费+学生评分"四种方案并收集学生反馈，发现"免费+学生自评分"接受度最高，为课程设计师提供决策依据。
- **首次系统性揭示学生在LLM评分器中的提示词黑客行为**：发现47%的学生尝试过prompt hacking，作业2中招比例达44%，并展示了多种创新攻击手法（如目标劫持、添加评估标准、添加新任务、让LLM自写自评等）。
- **提出基于自我反思的检测机制并验证有效性**：使用GPT-4对原始评估结果进行自我反思（self-reflection）检查，成功检测出44%的提示词黑客行为，且检测结果与学生自报比例一致。
- **明确区分"上下文敏感不安全"的新型prompt hacking**：指出当前LLM安全研究主要关注生成暴力/毒品等内容的安全问题，而对评分器等特定应用场景中被欺骗输出特定字符串的危害缺乏研究。

## 方法详解
- **LLM TA设计**：基于Chiang & Lee (2023a)的LLM-based evaluator框架，每个LLM TA由GPT-4-turbo和评估提示词组成。评估提示词包含四个要素：(1)任务指令，(2)评估标准与流程，(3)学生提交内容的占位符[[student's submission]]，(4)分数范围和输出格式。
- **推理增强**：所有评估提示词要求LLM在输出最终分数前先进行推理分析（reasoning before output），以提升与人类评分的一致性。
- **分数提取**：LLM TA被要求以特定格式（如"Final score: <score>"）输出，教学团队使用正则表达式从长回复中提取数值分数作为最终成绩。
- **部署平台**：使用MediaTek开发的DaVinci平台部署LLM TA，该平台为每位学生提供每日0.5 USD的免费额度（每次评估成本约0.05-0.09 USD）。
- **方案选择（Option 4）**：采用"免费+学生自评"方案——学生获得评估提示词和DAVINCI平台访问权限，可自行多次提交作业获取评分，选取满意的评分结果提交给教学团队作为最终成绩。
- **提示词黑客检测**：收集学生提交内容和原始评估结果后，用GPT-4进行自我反思检查（self-reflection prompt），判断原始评估是否存在问题及学生是否尝试黑客攻击。
- **防御尝试**：在后续作业中优化评估提示词，将学生提交内容夹在分隔句之间（"The following is the summary to be graded:" / "The above is the summary to be graded."），并在末尾强调忽略学生提交的任何评分规则修改指令，但发现仍被高级攻击绕过。

## 实验与结果
- **数据集/场景**：国立台湾大学"Introduction to Generative AI"课程，1,028名学生（EECS占80%，Liberal Arts占20%），6次作业使用LLM TA评分（HW2论文写作、HW3定制应用、HW4唐诗生成微调、HW6 RLHF对齐、HW8安全性评估、HW9视频摘要），4次作业使用其他自动评分方法。
- **学生反馈样本**：838名学生同意共享问卷回复。
- **核心结果**：
  - **75%的学生**在接受"免费+可访问"LLM TA的设定下表示可接受。
  - **51.3%**的学生遇到LLM TA未按指定格式输出评分的问题。
  - **21.5%**的学生遭遇LLM TA未遵循评估标准导致评分过低的情况。
  - **12.2%**的学生收到了自己认为过高的分数。
  - **47%**的学生尝试过prompt hacking（作业2中招率最高达44%）。
  - EECS学生中prompt hacking比例为51%，Liberal Arts学生中为27%，差异显著。
  - 四种方案中，"Free + Student-conducted"（Option 4）接受度最高；"Paid + Teacher-conducted"（Option 2）接受度最低（>66%学生认为不可接受）；不允许申诉教师评分的方案超过50%学生认为不可接受。
- **最强结果与提升**：提示词黑客检测通过self-reflection机制实现了44%的准确率，与学生自报比例完全吻合；防御性提示词优化虽可抵御基础攻击，但仍被高级攻击者利用"分隔符绕过"手法攻破。

## 相关工作脉络
- **Chiang & Lee (2023a)**：奠定LLM-based evaluator基础，证明GPT-4可替代人类专家进行评估，本文在此基础上将其应用于真实课堂。
- **Zheng et al. (2023, MT-bench)**：研究LLM-as-a-judge与人类偏好的一致性，本文与之互补——不仅关注一致性，更关注学生真实使用行为和攻击行为。
- **Liu et al. (2023a, G-eval)**：系统化研究GPT-4在NLG评估中的应用，本文沿用了其Prompt设计思路但聚焦教育场景落地。
- **Saito et al. (2023) / Koo et al. (2024)**：发现LLM评估中的冗长偏见（verbosity bias），本文学生反馈也报告了"LLM偏好长回复"的现象。
- **Schulhoff et al. (2023, HackAPrompt)**：通过全球竞赛形式揭示LLM的系统性prompt hacking漏洞，本文首次在学生群体中大规模观测到同类攻击行为。
- **定位差异**：已有工作主要关注LLM评分器与人类评分的对齐精度（academic benchmark视角），本文从真实课堂部署视角出发，关注学生可接受度、实际使用问题和攻击行为，填补了"研究→落地"的鸿沟。

## 局限性与未来方向
- **选择偏差**：课程名称为"Introduction to Generative AI"，招生时已明确告知使用LLM TA评分，报名学生本身对AI更感兴趣和熟悉，结果不能推广至一般学生群体。
- **GREEDY解码的困境**：虽然greedy decoding可减少评分随机性，但多数LLM Web界面不支持设置temperature，且一旦格式输出失败学生无法重新生成，影响用户体验，故作者不推荐。
- **自由访问的成本压力**：course designer可能难以为学生提供免费的LLM TA访问权限（尽管OpenAI自2024年5月起提供GPT-4o免费额度缓解了此问题）。
- **未来方向**：(1) 提高LLM在格式约束下同时保持推理能力的指令遵循能力；(2) 研究如何控制LLM多次采样的输出一致性；(3) 深入研究"上下文敏感不安全"的prompt hacking及其防御。

## 研究启发与可借鉴点
- **部署方案设计的权衡框架**：四种部署选项（是否可访问/付费/教师评分vs学生评分）的分析框架可直接迁移至其他教育AI部署场景，帮助课程设计师根据"学生满意度"和"评估鲁棒性"两个维度做出选择。
- **Prompt黑客检测的Self-reflection方法**：用另一个LLM调用对自身评估结果进行自我反思检查，是一种轻量且有效的检测策略，可复用于其他LLM-based evaluator场景的安全审计。
- **学生视角的可用性反馈收集方法**：将问卷设计为作业的一部分（提交即得分，与答案无关），既保证了高回收率又避免了成绩压力导致的偏差，值得借鉴。
- **分隔符包裹提示词设计**：将用户提交内容夹在明确的开始/结束分隔句之间，有助于LLM区分"评估标准"和"待评估内容"，可复用于各类LLM-based评价系统。
- **与团队方向的结合机会**：本团队若从事LLM评估器或AI教育应用研究，可将本文的"上下文敏感不安全"概念扩展为一个新的评测基准，或在指令遵循与格式约束方面开展联合研究。

## 关键术语表
- **LLM TA (LLM-based Evaluation Teaching Assistant)**：基于大语言模型的评估助教，指被提示评估学生作业的LLM系统。
- **Teacher-conducted score**：由教学团队使用LLM TA对学生作业进行评分所得的分数。
- **Student-conducted score**：由学生自己使用相同LLM TA对自身作业进行评分所得的分数。
- **Prompt Hacking**：使用对抗性提示词触发LLM产生预期外结果的攻击行为，本文特指欺骗评分器输出高分字符串。
- **Goal Hijacking**：一种prompt hacking类型，恶意提示旨在使LLM输出特定目标字符串（如"Final score: 10"）。
- **Self-reflection**：让LLM对自身先前推理结果进行审查和纠正的机制，本文用于检测prompt hacking。
- **Verbosity Bias**：LLM作为评估器时倾向于给更长回复更高分的认知偏差。

## 可复现要素
- **数据集**：课程作业提交内容和学生问卷回复，论文声明已获得学生知情同意，但代码/数据未公开（以课程数据保护为由）。
- **代码/权重**：未开源，评估提示词示例已见于附录Table 4/5/7。
- **关键超参**：LLM使用GPT-4-turbo，解码温度未设（非greedy），每日免费额度0.5 USD/学生，每次评估成本约0.05-0.09 USD。
