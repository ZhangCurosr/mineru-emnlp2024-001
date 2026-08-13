# LONGEMBED: Extending Embedding Models for Long Context Retrieval

Dawei Zhu \*♡♠ Liang Wang ♢ Nan Yang ♢ Yifan Song ♡♠ Wenhao Wu ♡♠

Furu Wei ♢ Sujian Li ♡♠♣

♡ School of Computer Science, Peking University

♠ National Key Laboratory for Multimedia Information Processing, Peking University

♣ Jiangsu Collaborative Innovation Center for Language Ability, Jiangsu Normal University ♢ Microsoft Corporation

{dwzhu,lisujian}@pku.edu.cn wangliang@microsoft.com https://github.com/dwzhu-pku/LongEmbed

## Abstract

Embedding models play a pivotal role in modern NLP applications such as document retrieval. However, existing embedding models are limited to encoding short documents of typically 512 tokens, restrained from application scenarios requiring long inputs. This paper explores context window extension of existing embedding models, pushing their input length to a maximum of 32,768. We begin by evaluating the performance of existing embedding models using our newly constructed LONGEM-BED benchmark, which includes two synthetic and four real-world tasks, featuring documents of varying lengths and dispersed target information. The benchmarking results highlight huge opportunities for enhancement in current models. Via comprehensive experiments, we demonstrate that training-free context window extension strategies can effectively increase the input length of these models by several folds. Moreover, comparison of models using Absolute Position Encoding (APE) and Rotary Position Encoding (RoPE) reveals the superiority of RoPE-based embedding models in context window extension, offering empirical guidance for future models. Our benchmark, code and trained models will be released to advance the research in long context embedding models.

## 1 Introduction

Text embeddings are vector representations of natural language that encode its semantic information. They play a pivotal role in various natural language processing (NLP) tasks, including information retrieval (IR) and retrieval-augmented generation (RAG). However, embedding models for producing these vector representations still operates within a very narrow context window, many supporting only 512 input tokens (Wang et al., 2022; Xiao et al., 2023; Ni et al., 2022). This narrow context window has greatly hindered their application in scenarios requiring long inputs, such as long Wikipedia articles and meeting scripts (Saad-Falcon et al., 2024).

Previous efforts that train a long context embedding modelfrom scratch suffer significant computational overhead, due to the combined demand for large batch sizes and long sequences. For example, Chen et al. (2024) utilized 96 A100 GPUs to train BGE-M3 which supports 8k context. Meanwhile, there have been many successes in extending context window of existing LLMs in a plug-and-play way or via efficient fine-tuning, pushing their context from 4k to 128k (Xiong et al., 2023) and even 2 million tokens (Ding et al., 2024). Motivated by this, instead of training long context embedding models from scratch, this paper explores context window extension of existing embedding models.

First, we examine the capability of existing embedding models in processing long context. Retrieval is selected as the proxy task, as it closely mirrors real-world application scenarios. While there have been some retrieval benchmarks such as BEIR (Thakur et al., 2021) and LoCo (Saad-Falcon et al., 2024), we identify two major limitations with these existing benchmarks: 1) limited document length, 2) biased distribution of target information. To overcome this, we introduce the LONGEMBED benchmark that integrates two synthetic tasks to enable flexible control over document length, and four real tasks featuring dispersed target information. Results on LONGEMBED indicates huge room for improvement in current embedding models.

Based on this, we explore plug-and-play strategies to extend embedding models, including parallel context windows, reorganizing position ids, and position interpolation. Comprehensive experiments show that these strategies can effectively extend the context window of existing embedding models by several folds, regardless of their original context being 512 or beyond 4k. Furthermore, for models employing absolute position encoding (APE), we show the possibility of harvesting further improvements via fine-tuning while strictly preserving original behavior within the short context. In this way, we have extended $\mathrm { E 5 _ { B a s e } }$ (Wang et al., 2022) from 512 to 4k (See Figure 1c).

![](images/ecc52afe6126ce71b3f7ee1ab1ed30bb086e7db7abad255cebcdac8a35a12b93.jpg)  
(a)

![](images/c16f1667c9750579a44910786e81abe10c6758195db3ff659fa923fb33fb8f71.jpg)  
(b)

![](images/4784ee7c69c39e097b51ba3a7afc069b16463ccf17c61116254109b79c363605.jpg)  
(c)  
Figure 1: (a) Overview of the LONGEMBED benchmark. (b) Performance of current embedding models on passkey retrieval, with evaluation length ranging from 256 to 32,768 <sup>1</sup>. ▲ / ♦ denotes embedding models with $5 1 2 / \geq$ 4k context. The greener a cell is, the higher retrieval accuracy this model achieves on the corresponding evaluation length. (c) Effects of context window extension methods on E5, E5-RoPE, E5-Mistral, measured by improvements of Avg. Scores on LONGEMBED. SE / NTK is short for SelfExtend / NTK-Aware Interpolation.

For models utilizing RoPE (Su et al., 2021), substantial enhancements on LONGEMBED are observed when employing methods that fully leverage RoPE’s advantages, such as NTK (Peng and Quesnelle, 2023) and SelfExtend (Jin et al., 2024). As illustrated in Figure 1b and 1c, leveraging NTK extends the context window of E5-Mistral to 32k, achieving close-to-perfect accuracy on passkey retrieval and state-of-the-art performance on LONGEMBED. Further, for fair comparison of APE / RoPE-based embedding models, we pretrain E5-RoPE following the training procedure and data of E5. Thorough comparison of E5 and E5-RoPE reveals the superiority of RoPE-based embedding models in context window extension. To sum up, our contributions are as follows:

• We construct LONGEMBED to benchmark long context retrieval, which includes two synthetic and four real-world tasks, featuring documents of varying lengths and dispersed target information.

• We have conducted comprehensive experiments on training-free context window extension, extending the input length of existing embedding models by several folds.

• We reveal the superiority of RoPE-based embedding models in context window extension via thorough comparison of models adopting APE and RoPE, offering empirical guidance for future embedding models.

• Our benchmark and trained models $( \mathrm { E 5 _ { B a s e ^ { - 4 k } } } .$ $_ { \mathrm { E 5 - R o P E _ { B a s e } ) } }$ will be released to advance the research in long context embedding models.

## 2 Related Work

Text Embedding Models. Text embeddings encode semantic information of text as lowdimensional vectors, enabling numerous NLP applications. Early attempts on embeddings models include latent semantic indexing (Deerwester et al., 1990) and weighted average of word embeddings (Mikolov et al., 2013). Modern embedding models (Wang et al., 2022; Xiao et al., 2023; Neelakantan et al., 2022) exploit supervision from labeled query-document pairs, adopting a multi-stage training paradigm that pre-trained on large-scale raw text pairs using contrastive loss, then fine-tuned on small scale but high-quality datasets.

Existing efforts in developing long-context embedding models typically involve first obtaining a long-context backbone model, either by pretraining with long inputs from scratch (Günther et al., 2023; Nussbaum et al., 2024; Chen et al., 2024) or using existing ones (Wang et al., 2023b), followed by training the backbone model to produce embeddings. Instead, this paper endows existing embedding models with the ability to handle long context through context window extension.

Context Window Extension for LLMs. Due to the high cost of pre-training an LLM from scratch, there have been many efforts towards extending the context window of existing LLMs in a plug-andplay manner. We categorize these efforts as follows: 1) Divide-and-conquer, which involves segmenting long inputs into short chunks, processing each chunk with the model, and aggregating the results, as demonstrated by PCW (Ratner et al., 2023); 2) Position reorganization, which reorganizes position ids to accommodate longer inputs, as exemplified by SelfExtend (Jin et al., 2024), DCA (An et al., 2024). 3) Position interpolation, which introduces new position embeddings by interpolating existing ones, includes PI (Chen et al., 2023), NTK (Peng and Quesnelle, 2023), YaRN (Peng et al., 2023), and Resonance RoPE (Wang et al., 2024a). Our paper thoroughly investigates these three lines of methods on embedding models. We also acknowledge other efforts in context extension, such as token compression (Jiang et al., 2023; Ge et al., 2023; Zhang et al., 2024a) and memory-based transformers (Wang et al., 2024b; Xiao et al., 2024). However, the former is not applicable for bidirectional attention, and the latter requires complex mechanisms for accessing encoded content, hence we do not experiment with these two categories.

In addition to their plug-and-play usability, further fine-tuning on top of these methods with long training samples has been proven to yield better performance (Xiong et al., 2023; Fu et al., 2024; Zhang et al., 2024b; Yen et al., 2024). Addressing the overhead of training on long inputs and the scarcity of extremely long training data, a line of research investigates simulating long inputs within short context, including Randomized Positions (Ruoss et al., 2023), Positional Skip-wise (PoSE) training (Zhu et al., 2023), and SkipAlign (Wu et al., 2024). This paper also leverage these efforts to synthesize long training samples from the original training data, facilitating further fine-tuning on top of plug-and-play methods.

## 3 The LONGEMBED benchmark

In this section, we first identify two limitations of existing retrieval benchmarks for evaluating longcontext capabilities (§ 3.1). Then, we introduce the retrieval tasks adopted in our LONGEMBED, including both synthetic ones (§ 3.2) and real ones (§ 3.3).

![](images/4dcf4beefc642c981977a166b1d6a46c9acce2edc0f39f3a8f63020f784f1f75.jpg)  
Figure 2: Results of $\mathrm { E 5 _ { B a s e } }$ on 8 LoCo tasks that are publicly available.

## 3.1 Examing Existing Retrieval Benchmarks

There are two main desiderata for curating a long context retrieval benchmark. First, the candidate documents should be long enough. Second, the target information to answer user query should be as uniformly distributed across the document as possible. This prevents embedding models from solely focusing on specific parts, such as the beginning (Coelho et al., 2024), to achieve unreasonably high scores. Based on these criteria, we examine existing retrieval benchmarks as follows:

BEIR Benchmark (Thakur et al., 2021) is a collection of 18 information retrieval datasets, ranging across ad-hoc web search, question answering, fact verification, etc. However, documents in this benchmark contains fewer than 300 words on average (See Table 5 in Appendix), making it unsuitable for measuring long context retrieval that usually involves documents of thousands or tens of thousands of words.

LoCo Benchmark (Saad-Falcon et al., 2024) consists 12 retrieval tasks that requires long context reasoning, spanning diverse domains such as law and finance. However, it still suffers from biased distribution of key information, as demonstrated in Figure 2. With only 512 context length, E5<sub>Base</sub> achieves >85% nDCG scores on 3 out of 8 publiclyavailable LoCo tasks. This severely biased distribution of target information undermines its ability to reflect model performance as context increases.

## 3.2 Synthetic Tasks in LONGEMBED

First, we introduce the passkey and needle retrieval task for embedding models as follows:

Personalized Passkey Retrieval. Passkey retrieval (Mohtashami and Jaggi, 2023) requires LLMs to recover a random passkey hidden within a long document comprising garbage information. For embedding models, we adopt the personalized passkey retrieval (Wang et al., 2023b), where each document contains a unique person name and his/her passkey at random position. The goal is to retrieve the document containing the given person’s passkey from all candidates documents.

<table><tr><td>Dataset</td><td>Domain</td><td># Queries</td><td># Docs</td><td>Avg. Query Words</td><td>Avg. Doc Words</td></tr><tr><td colspan="6">Real Tasks</td></tr><tr><td>NarrativeQA</td><td>Literature, Film</td><td>10,449</td><td>355</td><td>9</td><td>50,474</td></tr><tr><td>QMSum</td><td>Meeting</td><td>1,527</td><td>197</td><td>71</td><td>10,058</td></tr><tr><td>2WikiMultihopQA</td><td>Wikipedia</td><td>300</td><td>300</td><td>12</td><td>6,132</td></tr><tr><td>SummScreenFD</td><td>ScreenWriting</td><td>336</td><td>336</td><td>102</td><td>5,582</td></tr><tr><td colspan="6">Synthetic Tasks</td></tr><tr><td>Passkey</td><td>Synthetic</td><td>400</td><td>800</td><td>11</td><td>十十</td></tr><tr><td>Needle</td><td>Synthetic</td><td>400</td><td>800</td><td>7</td><td></td></tr></table>

Table 1: Overview of the LONGEMBED benchmark. Average word number is rounded to the nearest integer. † For needle and passkey test, we have 8 groups of queries and candidate documents, with the documents averaging 0.25, 0.5, 1, 2, 4, 8, 16, 32 0.75k words, respectively.

![](images/611ab343ef3fae6aa6d4342a4d61dc9662368b34675e0b938eddb79f8346d59f.jpg)  
Figure 3: Example for the passkey and needle test. For the passkey test, the <prefix / suffix> are repeats of "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again." For the needle test, the <prefix> and <suffix> form a long essay.

Needle-in-a-haystack Retrieval. While passkey retrieval surrounds key information with garbage sentences, needle-in-a-haystack retrieval (Kamradt, 2023; Liu et al., 2024) randomly inserts key information into an arbitrary position of a long essay, making the task more challenging. To tailor this task for embedding models, we instruct GPT-4 to generate 100 facts covering a variety of domains including physics, history, geometry, art, etc, and 100 queries correspondingly. The facts are subsequently treated as needles and randomly inserted into the PaulGrahamEssay to form 100 candidate documents. Our task is to correctly retrieve the document that contains corresponding needle given the query.

The advantage of synthetic data is that we can flexibly control context length and distribution of target information. For both tasks, we evaluate a broad context range of 0.25, 0.5, 1, 2, 4, 8, 16, 32 1, 024 tokens <sup>2</sup>. For each context length, we include 50 test samples, each comprising 1 query and 100 candidate documents. <sup>3</sup> In this way, we can measure the effective context size of embedding models for up to 32k tokens. Examples for both tasks are in Figure 3.

## 3.3 Real Tasks in LONGEMBED

While synthetic tasks offer flexibility in manipulating context length and distributing target information, they still differ from real-world scenarios. To conduct a comprehensive evaluation, we have tailored following long-form QA and summarization tasks for long context retrieval. For QA datasets, we use the questions as queries, the set of all input documents as candidate documents. For summarization datasets, we use the summaries as queries, and the set of all input documents as candidate documents.

NarrativeQA (Kociský et al.ˇ , 2018) is a QA dataset comprising long stories and corresponding questions about specific content such as characters, events. As these details are dispersed throughout the story, models must process the entire long context to get the correct answers.

2WikiMultihopQA (Ho et al., 2020) is a multi-hop QA dataset featuring questions with up to 5 hops, synthesized through manually designed templates to prevent shortcut solutions. This necessitates the ability to process and reason over long context, ensuring that answers cannot be obtained by merely focusing on a short span within the document.

QMSum (Zhong et al., 2021) is a query-based meeting summarization dataset that requires selecting and summarizing relevant segments of meetings in response to queries. Due to the involvement of multiple participants and topics in the meeting, summarization regarding specific queries naturally requires aggregating information dispersed throughout the entire text.

SummScreenFD (Chen et al., 2022) is a screenplay summarization dataset comprising pairs of TV series transcripts and human-written summaries. Similar to QMSum, its plot details are scattered throughout the transcript and must be integrated to form succinct descriptions in the summary.

Table 1 presents the overall statistics of LONGEMBED. Considering the computational complexity that increases quadratically with input length, we intentionally restrict the number of candidate documents in each task to to not exceed $1 0 ^ { 3 }$ In this way, we can efficiently evaluate the basic long context capabilities of embedding models. For further elaboration on the source and examples for each dataset, please refer to Appendix C.

## 4 Methodology

## 4.1 Preliminary: APE & RoPE

Absolute Position Embedding (APE) stands as the predominant positional encoding strategy for embedding models, as majority of them follows the BERT architecture (Devlin et al., 2019). APEbased models first embed absolute position ids into position vectors and add token embeddings to their corresponding position vectors, before feeding them to a stack of transformer layers.

Rotary Position Embedding (RoPE) is the most pervasive position embedding strategy in the era of LLMs, including LLaMA (Touvron et al., 2023), QWen (Bai et al., 2023a), etc. It encodes position information of tokens with a rotation matrix that naturally incorporates explicit relative position dependency. To elucidate, given a hidden vector $\pmb { h } = [ h _ { 0 } , h _ { 1 } , . . . , h _ { d - 1 } ]$ of dimension $d ,$ and a position index m, RoPE operates as follows:

$$
\begin{array} { c } { { f ( h , m ) = [ ( h _ { 0 } + \mathrm { i } h _ { 1 } ) e ^ { \mathrm { i } m \theta _ { 0 } } , ( h _ { 2 } + \mathrm { i } h _ { 3 } ) e ^ { \mathrm { i } m \theta _ { 1 } } , . . . , } } \\ { { { \left( h _ { d - 2 } + \mathrm { i } h _ { d - 1 } \right) e ^ { \mathrm { i } m \theta _ { d / 2 - 1 } } ] } } } \end{array}
$$

where $\theta _ { j } = 1 0 0 0 0 ^ { - 2 j / d } , j \in \{ 0 , 1 , . . . , d / 2 - 1 \}$ $\mathrm { i } = \sqrt { - 1 }$ is the imaginary unit. Unlike APE that is directly applied to the input vector $^ { x , }$ RoPE is employed on the query and key vectors at each layer. The attention score $\boldsymbol { a } ( \boldsymbol { q } , \boldsymbol { k } )$ between a query q at position m and a key k at position n is:

$$
\begin{array} { l } { { a ( { \pmb q } , { \pmb k } ) = \mathrm { R e } \langle f ( { \pmb q } , m ) , f ( { \pmb k } , n ) \rangle } } \\ { { \ = \mathrm { R e } \left[ \displaystyle \sum _ { j = 0 } ^ { d / 2 - 1 } ( q _ { 2 j } + \mathrm { i } q _ { 2 j + 1 } ) ( k _ { 2 j } - \mathrm { i } k _ { 2 j + 1 } ) e ^ { \mathrm { i } ( m - n ) \theta _ { j } } \right] } } \\ { { \ : = g ( { \pmb q } , { \pmb k } , ( m - n ) { \pmb \theta } ) } } \end{array}\tag{1}
$$

where g(·) is an abstract mapping function exclusively dependent on q, k and $( m - n ) \theta$

## 4.2 Extending APE-based Models

As delineated in Section 2, training-free context extension strategies applicable to embedding models can be classified into 3 categories: 1) Divideand-conquer; 2) Position reorganization; 3) Position interpolation. In this section, we introduce methods from each of these categories to assess their applicability to embedding models. Further fine-tuning on top of these methods is also included. Let $L _ { o }$ represent the original context length, $\mathcal { D } = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { L _ { t } } \}$ denote a long document of target context length $L _ { t }$ , and $s = \lceil L _ { t } / L _ { o } \rceil$ indicate the context scaling factor. The context extension methods we investigated are described below:

Parallel Context Windows (PCW). To process a long document with a short-context model, PCW divides the long document into multiple short chunks, processes each chunk in parallel, and aggregates their results (Ratner et al., 2023; Yen et al., 2024). In practice, we first segment into chunks of $L _ { o }$ tokens, then average over each chunk’s embeddings to represent . For simplicity, we set the overlap between adjacent chunks to 0, except for the last chunk, to ensure it contains $L _ { o }$ tokens.

Grouped & Recurrent Positions (GP & RP). Dividing inputs into chunks and processing them separately sacrifices their interaction in between. By contrast, position reorganization accommodates longer context by reusing the original position ids. To be specific, we experiment with two simple strategies: Grouped Positions and Recurrent $P o -$ sitions. The former groups the original position ids as such: $f _ { g p } ( p i d )  \lfloor p i d / s \rfloor$ , while the latter assigns the position ids recurrently, formulated as: $f _ { r p } ( p i d )  p i d$ mod $L _ { o }$

![](images/73a24c82142ed0ce88b4b251a995be8b1e378051f3a73b980fe59cf9bf874f87.jpg)  
Figure 4: (Left) Arrangement of pids for extending APE-based models from 512 to 1,024. (Right) Illustration of learnable ( ) and frozen ( ) position vectors when further tuning on RP / PI.

Linear Position Interpolation (PI). Instead of reusing position ids, Chen et al. (2023) introduces new position embeddings via linear interpolation of existing ones. To apply PI on APE-based models, we map the positions ids as such: $f _ { p i } ( p i d ) $ $p i d / s$ , and assign embeddings for non-integers as linear interpolation of that of neighboring integers. In practice, we first extend the original position embedding matrix $E _ { o } \in \mathbb { R } ^ { L _ { o } \times d }$ into $E _ { t } \in \mathbb { R } ^ { L _ { t } \times d }$ where d stands for hidden size. Next, we assign $E _ { t } [ i \cdot s ] = E _ { o } [ i ] , i \in \{ 0 , 1 , . . . , L _ { o } - 1 \}$ . For noninteger position id $j$ between i and $i + 1$ , we determine their embeddings as follows: $E _ { t } [ s \cdot j ] =$ $( ( i + 1 - j ) E _ { t } [ i \cdot s ] + ( j - i ) E _ { t } [ ( i + 1 ) \cdot s ] )$

Further Tuning. Except for PCW, which divides long texts into smaller blocks and processes separately, GP, RP, and PI can all be seen as extending the position embedding matrix. Since APE-based models assign an independent vector to each position, we can freeze the original model parameters while updating only the newly added position embeddings. In this way, we can strictly maintain model ability within 512 context, while harvesting further performance gains in handling long context as free lunch. Specifically, further finetuning on top of RP and PI is explored in this paper, as illustrated in Figure 4 (Right). Since the traditional training data for embedding models are short queries and passages not exceeding 512 tokens, we manipulate position ids to simulate long training samples, as proposed in Zhu et al. (2023). See Appendix B for details of further fine-tuning.

## 4.3 Extending RoPE-based Models

For RoPE-based models, we further explore Self Extend and NTK, which respectively advances over GP and PI, harnessing the inherent advantages of RoPE. Since there is no simple strategy for further training while exactly maintaining original performance like APE, we leave comprehensive exploration of training-based context window extension for RoPE-based models for future work.

Self Extend (SE). Compared with APE, RoPE operates on the query and key vectors at each layer to encode relative positions, offering enhanced flexibility for position reorganization. For each token, instead of assigning grouped relative positions to all other tokens, SelfExtend (Jin et al., 2024) re-introduces normal relative positions within the nearest neighbor window w, achieving improved performance. For example, consider a document of 10 tokens $\{ x _ { 0 } , x _ { 1 } , . . . , x _ { 9 } \}$ with a neighbor window size w = 4 and a group size $g = 2 .$ The relative positions to x<sub>0</sub> are $\{ 0 , 1 , 2 , 3 , 4 , 4 , 5 , 5 , 6 , 6 \}$ . For x<sub>4</sub>, the relative positions of the other tokens are $\{ - 4 , - 3 , - 2 , - 1 , 0 , 1 , 2 , 3 , 4 , 4 \}$

NTK-Aware Interpolation (NTK). Given a scaling factor s, PI proportionally down-scales position index m to $m / s .$ In this way, the attention score $\boldsymbol { a } ( \boldsymbol { q } , \boldsymbol { k } )$ defined in Equation 1 becomes $g ( \pmb { q } , \pmb { k } , ( m - n ) \pmb { \theta } / s )$ This is also equivalent to reducing the frequencies θ uniformly, which may prevent the model from learning high-frequency features, as shown by the Neural Tangent Kernel (NTK) theory (Jacot et al., 2018). To remedy this, NTK-Aware interpolation (Peng and Quesnelle, 2023) scales high frequencies less and low frequencies more to spread out the interpolation pressure across multiple dimensions. This is achieved by directly altering the original $\theta _ { j } = 1 0 0 0 0 ^ { - 2 j / d }$ into $\theta _ { j } ^ { \prime } = ( 1 0 0 0 0 \lambda ) ^ { - 2 j / d }$ , where λ is conventionally chosen to be slightly greater than s.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Param.</td><td rowspan="2">CTX Len.</td><td colspan="2">Synthetic (Acc@1)</td><td colspan="4">Real (nDCG@10)</td><td rowspan="2">Avg.</td></tr><tr><td>Passkey</td><td>Needle</td><td>NQA</td><td>QMS</td><td>SFD</td><td>WQA</td></tr><tr><td colspan="10">512 Context Models</td></tr><tr><td> $\mathrm { E 5 _ { B a s e } }$  (Wang et al., 2022)</td><td>110M</td><td>512</td><td>38.0</td><td>28.5</td><td>25.3</td><td>23.8</td><td>74.7</td><td>55.8</td><td>41.0</td></tr><tr><td> $\mathrm { E 5 - R o P E _ { B a s e } }$ </td><td>110M</td><td>512</td><td>38.5</td><td>31.5</td><td>24.6</td><td>23.2</td><td>66.6</td><td>58.8</td><td>40.5</td></tr><tr><td>GTEBase (Li et al., 2023)</td><td>110M</td><td>512</td><td>31.0</td><td>24.5</td><td>28.6</td><td>21.8</td><td>55.8</td><td>47.3</td><td>34.8</td></tr><tr><td> $\mathrm { B G E _ { B a s e } }$  (Xiao et al., 2023)</td><td>110M</td><td>512</td><td>18.0</td><td>25.3</td><td>25.6</td><td>22.4</td><td>60.3</td><td>51.7</td><td>33.9</td></tr><tr><td>Contriever (Izacard et al., 2021)</td><td>110M</td><td>512</td><td>38.5</td><td>29.0</td><td>26.7</td><td>25.5</td><td>73.5</td><td>47.3</td><td>40.1</td></tr><tr><td> $\mathrm { G T R } _ { \mathrm { B a s e } } \ ( \mathrm { N i } \ \mathrm { e t } \ \mathrm { a l } . , 2 0 2 2 )$ </td><td>110M</td><td>512</td><td>38.5</td><td>26.3</td><td>26.5</td><td>18.3</td><td>63.7</td><td>52.2</td><td>36.5</td></tr><tr><td colspan="10">≥ 4k Context Models</td></tr><tr><td>E5-Mistral (Wang et al., 2023b)</td><td>7B</td><td>4,096</td><td>71.0</td><td>48.3</td><td>44.6</td><td>43.6</td><td>96.8</td><td>82.0</td><td>64.4</td></tr><tr><td>Jina-V2 (Günther et al., 2023)</td><td>137M</td><td>8,192</td><td>50.3</td><td>54.5</td><td>37.9</td><td>38.9</td><td>93.5</td><td>74.0</td><td>58.2</td></tr><tr><td>Nomic-V1(Nussbaum et al., 2024)</td><td>137M</td><td>8,192</td><td>60.7</td><td>39.5</td><td>41.2</td><td>36.7</td><td>93.0</td><td>73.8</td><td>57.5</td></tr><tr><td>BGE-M3 (Chen et al., 2024)</td><td>568M</td><td>8,192</td><td>59.3</td><td>40.5</td><td>45.8</td><td>35.5</td><td>94.0</td><td>78.0</td><td>58.9</td></tr><tr><td>OpenAI-Ada-002</td><td></td><td></td><td>50.8</td><td>36.8</td><td>41.1</td><td>40.0</td><td>91.8</td><td>80.1</td><td>56.8</td></tr><tr><td colspan="10">Our Extended Models</td></tr><tr><td> $\mathrm { E 5 _ { B a s e } + T u n i n g \ ( 4 k ) }$ </td><td>110M</td><td>4,096</td><td>67.3</td><td>41.5</td><td>30.4</td><td>35.7</td><td>95.2</td><td>69.2</td><td>56.6</td></tr><tr><td> $\mathrm { E 5  – R o P E _ { B a s e } + S e l f E x t e n d ( 4 k ) }$ </td><td>110M</td><td>4,096</td><td>73.5</td><td>53.5</td><td>32.3</td><td>39.1</td><td>91.9</td><td>74.6</td><td>60.8</td></tr><tr><td> $\mathrm { E 5 - M i s t r a l + N T K } \left( 3 2 \mathrm { k } \right)$ </td><td>7B</td><td>32,768</td><td>93.8</td><td>66.8</td><td>49.8</td><td>49.2</td><td>97.1</td><td>95.2</td><td>75.3</td></tr></table>

Table 2: Results (%) of existing and extended embedding models on LONGEMBED. NQA, QMS, SFD, WQA is short for NarrativeQA, QMSum, SummScreenFD, 2WikiMultihopQA, respectively. We show that context window extension can effectively improve existing embedding models in processing long context.

## 5 Experiments

## 5.1 Experimental Setup

Benchmarked Models. We evaluate both opensourced and proprietary models on LONGEMBED, including E5, GTE, BGE, Contriever, GTR, E5- Mistral, Jina-V2, Nomic-V1, BGE-M3, OpenAIada-002. M2 (Saad-Falcon et al., 2024) is not included in our evaluation, given its training data partly overlaps with test samples in LONGEMBED.

Candidate Models for Extension. From each of the APE-based and RoPE-based category, we select 2 candidate models for comprehensive study. The former includes $\mathrm { E 5 _ { B a s e } }$ and $\mathrm { G T E } _ { \mathrm { B a s e } }$ . The latter includes the 4,096-context E5-Mistral, and a newly trained E5- $\cdot { \mathrm { R o P E } } _ { \mathrm { B a s e } } ,$ which supports 512 context (See Appendix A for its training details and BEIR results). Note that E5- $\mathrm { . R o P E _ { B a s e } }$ employs the same training procedure and training data as $\mathrm { E 5 _ { B a s e } }$ , only with APE substituted with RoPE. This facilitates fair comparison of APE / RoPE-based models in context window extension, as presented in Section 5.4. For implementation details of each context window extension strategies on each model, please refer to Appendix B.

## 5.2 Main Results

Table 2 demonstrates the performance of existing embedding models on our LONGEMBED benchmark. Among the 512-context models, $\mathrm { E 5 _ { B a s e } }$ achieves the highest average score of 41.0 points, closely followed by ${ \mathrm { E 5 - R o P E } } _ { \mathrm { B a s e } }$ and Contriever. As the supported context length increases beyond 4k, exemplified by E5-Mistral and Jina-V2, a discernible increase in scores is observed. This verifies both the efficacy of these long-context models and the validity of LONGEMBED to assess longcontext retrieval. Note that even the best performing model attains only 64.4 pts on average, indicating huge room for improvement in current models.

In the last row block of Table 2, we further include the best results achieved by $\mathrm { E 5 _ { B a s e } }$ , E5- $\mathrm { R o P E _ { B a s e } }$ and E5-Mistral after context window extension. For $\mathrm { E 5 _ { B a s e } }$ and ${ \mathrm { E } } 5 { \mathrm { - } } { \mathrm { R o P E } } _ { \mathrm { B a s e } }$ , we extend their contexts from 512 to 4,096. For E5-Mistral, we extend its context from 4,096 to 32,768. Compared to the original versions, the extended models achieve an average score increase of +15.6 / +20.3 / +10.9 points. This indicates the efficacy of these context extension strategies on embedding models, enabling them to handle inputs of several folds longer. Detailed performance comparison of different extension strategies on APE & RoPE-based embedding models is presented in Section 5.3.

![](images/54337fa0e31fa09e72f55cba39321b37d744cbe30795f975b3672be45cdb64a0.jpg)

![](images/fa55b063952e2a25599194050b2017716b0313749143b6ccc0b32f7587a56ff1.jpg)  
Figure 5: Effects of different context window extension methods on $\mathrm { E 5 _ { B a s e } }$ and $\mathrm { G T E } _ { \mathrm { B a s e } }$ . We show that further tuning yields the best results.

![](images/57f1618726a18510d537fd2abe0d71e06dabd4ca0acd0eb0ff9bbb4172680a60.jpg)  
(a)

![](images/3942540dfc4df9fcdf25d11995c3905172287af318d60ddae70cb68ea51fe163.jpg)  
(b)  
Figure 6: (a) Performance gain after tuning on PI / RP, compared with the original model. (b) Best results achieved by extended versions of $\mathrm { E 5 _ { B a s e } / E 5  – R o P E _ { B a s e } } .$

## 5.3 Comparison of Extension Methods

APE-based Models. Figure 5 illustrates the impact of various context extension strategies on $\mathrm { E 5 _ { B a s e } }$ and $\mathrm { G T E } _ { \mathrm { B a s e } }$ across different target context lengths. We observe that plug-and-play methods including GP, RP, PI and PCW strategies yield comparable results with no significant disparities. On the other hand, further tuning consistently yields additional performance gains for both models, across all target context lengths. Particularly noteworthy is $\mathrm { G T E } _ { \mathrm { B a s e } }$ , which showcases a substantial average score increase of approximately 5 points after further tuning. This suggests that freezing the original model weights and fine-tuning exclusively the added position embeddings can effectively extend the model’s context window while strictly maintaining model’s original ability.

RoPE-based Models. Table 3 depicts the outcomes of ${ \mathrm { E 5 - R o P E _ { B a s e } } }$ and E5-Mistral on each dataset of LONGEMBED after context window extension via PCW, GP, PI, SE and NTK. It is observed that RoPE-specific methods including NTK and SE yield significant improvements for both models across all datasets, surpassing PCW, PI and GP by a large margin.

<table><tr><td rowspan="2">Model</td><td colspan="2">Synthetic</td><td rowspan="2">Avg. WQA</td></tr><tr><td>P N</td><td>NQA QMS SFD</td></tr><tr><td> $E 5  – R o P E _ { B a s e }$ </td><td>38.5 31.5 24.6</td><td>23.2 66.6 58.8</td><td>40.5 69.3 52.9</td></tr><tr><td>+PCW (4k) +GP (4k) +PI (4k) +SE (4k) +NTK (4k)</td><td>42.5 50.8 25.1 68.0 38.8 25.9 68.3 36.0 25.9 73.5 53.5 32.3 66.3 46.5 25.5</td><td>34.9 94.9 30.9 85.8 30.8 84.9 39.1 91.9 35.8 90.8</td><td>65.8 52.5 65.3 51.9 74.6 60.8 71.7 56.1</td></tr><tr><td>E5-Mistral</td><td>71.0 48.3 44.6</td><td>43.6 96.8</td><td>82.0 64.4</td></tr><tr><td>+PCW (32k)</td><td>63.5 49.5 59.3 93.8 66.8 49.8 49.2</td><td>51.3 97.3</td><td>91.2 68.7</td></tr><tr><td>+GP (32k) +PI (32k)</td><td>81.0 48.8 37.0</td><td>42.9 90.6</td><td>88.1 64.7 59.4</td></tr><tr><td>+SE (32k) +NTK (32k)</td><td>89.8 48.5 37.8 90.8 52 49.3</td><td>40.4 76.8 48.7 97.2</td><td>63.0 96.4 72.4</td></tr></table>

Table 3: Results (%) of context window extension methods on ${ \mathrm { E 5 - R o P E } } _ { \mathrm { B a s e } }$ and E5-Mistral. For datasets, P, N, NQA, QMS, SFD, WQA is short for Passkey, Needle, NarrativeQA, QMSum, SummScreenFD, 2WikiMultihopQA. For extension methods, PCW, GP, PI, SE, NTK are short for Parallel Context Windows, Grouped Positions, Linear Position Interpolation, SelfExtend, and NTK-Aware Interpolation, respectively.

## 5.4 Analysis

Tuning on PI vs. RP. Figure 6a compares further tuning on top of RP vs. PI. In the former approach, the initial 512 position embeddings are frozen while the remaining embeddings are tuned, whereas for the latter, the frozen / learnable embedding vectors are arranged in an interleaved manner. We observe that tuning on PI consistently produces superior results on both $\mathrm { G T E } _ { \mathrm { B a s e } }$ and $\mathrm { E 5 _ { B a s e } }$ . A possible explanation is that fixed vectors in PI serve intrinsically as anchors, preventing the learnable vectors from converging to suboptimal values.

RoPE vs. APE. We further discuss the potential of APE / RoPE-based models for context window extension. $\mathrm { E 5 _ { B a s e } }$ and ${ \mathrm { E 5 - R o P E } } _ { \mathrm { B a s e } }$ are selected as the comparison subjects thanks to their shared training process, training data, and comparable performance on BEIR and LONGEMBED benchmarks. At each target context length ( 1k, 2k, 4k ), we report the best scores achieved by each model on LONGEMBED, as illustrated in Figure 6b. Without requiring further training, ${ \bf E } 5 { \mathrm { - } } \mathrm { R o P E } _ { \mathrm { B a s e } }$ consistently demonstrates superior performance compared to $\mathrm { E 5 _ { B a s e } }$ across all target lengths. Furthermore, as the target window length increases, this superiority becomes more pronounced, even surpassing the fine-tuned version of $\mathrm { E 5 _ { B a s e } }$ by a large margin. This suggests that RoPE-based models can better extrapolate to to longer context. Consequently, we advocate for the use of RoPE in future embedding models.

## 6 Conclusion

This paper explores context window extension of existing embedding models. Through extensive experiments on our LONGEMBED benchmark, we show that training-free context window extension strategies can effectively increase the input length of these models by several folds. Further, our analysis reveals the superiority of RoPE-based embedding models over APE-based ones in context window extension. Hence, we advocate for the use of RoPE for future embedding models.

## Limitations

As a pioneering work in applying context window extension on embedding models, this paper is still limited in several aspects, particularly in that most of the context extension strategies explored in this paper are training-free. As evidenced by previous findings (Xiong et al., 2023; Fu et al., 2024; Zhang et al., 2024b; Yen et al., 2024), and the additional performance gain achieved via tuning on $\mathrm { E 5 _ { B a s e } }$ and $\mathrm { G T E } _ { \mathrm { B a s e } }$ , we believe further fine-tuning on top of plug-and-play methods can bring even better extension results. In the future, we will make comprehensive exploration of training-based context window extension for embedding models, especially for RoPE-based ones.

## Ethics Statement

This work fully complies with the ACL Ethics Policy. We declare that there are no ethical issues in this paper, to the best of our knowledge.

## Acknowledgement

We thank the anonymous reviewers for their helpful comments on this paper. We thank Xueguang Ma, Niklas Muennighoff, and Kenneth Enevoldsen for their thoughtful discussion and assistance in integrating LongEmbed into MTEB. This work was partially supported by National Natural Science Foundation of China (No. 62476010).

## References

Chenxin An, Fei Huang, Jun Zhang, Shansan Gong, Xipeng Qiu, Chang Zhou, and Lingpeng Kong. 2024. Training-free long-context scaling of large language models. arXiv preprint arXiv:2402.17463.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023a. Qwen technical report. arXiv preprint arXiv:2309.16609.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. 2023b. Longbench: A bilingual, multitask benchmark for long context understanding. arXiv preprint arXiv:2308.14508.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216.

Mingda Chen, Zewei Chu, Sam Wiseman, and Kevin Gimpel. 2022. Summscreen: A dataset for abstractive screenplay summarization. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8602–8615.

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. 2023. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595.

David Chiang and Peter Cholak. 2022. Overcoming a theoretical limitation of self-attention. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7654–7664, Dublin, Ireland. Association for Computational Linguistics.

João Coelho, Bruno Martins, João Magalhães, Jamie Callan, and Chenyan Xiong. 2024. Dwell in the beginning: How language models embed long documents for dense retrieval. arXiv preprint arXiv:2404.04163.

Scott Deerwester, Susan T Dumais, George W Furnas, Thomas K Landauer, and Richard Harshman. 1990. Indexing by latent semantic analysis. Journal of the American societyfor information science, 41(6):391– 407.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, and Mao Yang. 2024. Longrope: Extending llm context window beyond 2 million tokens. arXiv preprint arXiv:2402.13753.

Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, and Hao Peng. 2024. Data engineering for scaling language models to 128k context. arXiv preprint arXiv:2402.10171.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. Simcse: Simple contrastive learning of sentence embeddings. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894–6910.

Tao Ge, Jing Hu, Xun Wang, Si-Qing Chen, and Furu Wei. 2023. In-context autoencoder for context compression in a large language model. arXiv preprint arXiv:2307.06945.

Michael Günther, Jackmin Ong, Isabelle Mohr, Alaeddine Abdessalem, Tanguy Abel, Mohammad Kalim Akram, Susana Guzman, Georgios Mastrapas, Saba Sturua, Bo Wang, et al. 2023. Jina embeddings 2: 8192-token general-purpose text embeddings for long documents. arXiv preprint arXiv:2310.19923.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing a multihop QA dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th International Conference on Computational Linguistics, pages 6609–6625, Barcelona, Spain (Online). International Committee on Computational Linguistics.

Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2021. Towards unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118, 2(3).

Arthur Jacot, Franck Gabriel, and Clément Hongler. 2018. Neural tangent kernel: Convergence and generalization in neural networks. Advances in neural information processing systems, 31.

Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023. Llmlingua: Compressing prompts for accelerated inference of large language models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 13358–13376.

Hongye Jin, Xiaotian Han, Jingfeng Yang, Zhimeng Jiang, Zirui Liu, Chia-Yuan Chang, Huiyuan Chen, and Xia Hu. 2024. Llm maybe longlm: Self-extend llm context window without tuning. arXiv preprint arXiv:2401.01325.

Greg Kamradt. 2023. Needle in a haystack - pressure testing llms. https://github.com/gkamradt/ LLMTest\_NeedleInAHaystack.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for opendomain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781.

Tomáš Kociský, Jonathan Schwarz, Phil Blunsom, Chrisˇ Dyer, Karl Moritz Hermann, Gábor Melis, and Edward Grefenstette. 2018. The NarrativeQA reading comprehension challenge. Transactions ofthe Associationfor Computational Linguistics, 6:317–328.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. 2019. Natural questions: A benchmark for question answering research. Transactions ofthe Association for Computational Linguistics, 7:452– 466.

Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, Daniel Haziza, Luca Wehrstedt, Jeremy Reizenstein, and Grigory Sizov. 2022. xformers: A modular and hackable transformer modelling library. https: //github.com/facebookresearch/xformers.

Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. 2023. Towards general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions ofthe Association for Computational Linguistics, 12:157–173.

Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.

Amirkeivan Mohtashami and Martin Jaggi. 2023. Landmark attention: Random-access infinite context length for transformers. arXiv preprint arXiv:2305.16300.

Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy, et al. 2022. Text and code embeddings by contrastive pretraining. arXiv preprint arXiv:2201.10005.

Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. Ms marco: A human-generated machine reading comprehension dataset.

Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith Hall, Ming-Wei Chang, et al. 2022. Large dual encoders are generalizable retrievers. In Proceedings

of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9844–9855.

Zach Nussbaum, John X Morris, Brandon Duderstadt, and Andriy Mulyar. 2024. Nomic embed: Training a reproducible long context text embedder. arXiv preprint arXiv:2402.01613.

Bowen Peng and Jeffrey Quesnelle. 2023. Ntkaware scaled rope allows llama models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation. https://www.reddit.com/r/LocalLLaMA/ comments/14lz7j5/ntkaware\_scaled\_rope\_ allows\_llama\_models\_to\_have.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. 2023. Yarn: Efficient context window extension of large language models. arXiv preprint arXiv:2309.00071.

Nir Ratner, Yoav Levine, Yonatan Belinkov, Ori Ram, Inbal Magar, Omri Abend, Ehud Karpas, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. 2023. Parallel context windows for large language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6383–6402.

Anian Ruoss, Grégoire Delétang, Tim Genewein, Jordi Grau-Moya, Róbert Csordás, Mehdi Bennani, Shane Legg, and Joel Veness. 2023. Randomized positional encodings boost length generalization of transformers. In Proceedings ofthe 61st Annual Meeting ofthe Association for Computational Linguistics (Volume 2: Short Papers), pages 1889–1903.

Jon Saad-Falcon, Daniel Y Fu, Simran Arora, Neel Guha, and Christopher Ré. 2024. Benchmarking and building long-context retrieval models with loco and m2-bert. arXiv preprint arXiv:2402.07440.

Uri Shaham, Elad Segal, Maor Ivgi, Avia Efrat, Ori Yoran, Adi Haviv, Ankit Gupta, Wenhan Xiong, Mor Geva, Jonathan Berant, and Omer Levy. 2022. SCROLLS: Standardized CompaRison over long language sequences. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 12007–12021, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Jianlin Su. 2021. Understanding attention scaling from the perspective of entropy invariance. https: //spaces.ac.cn/archives/8823.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. 2021. Roformer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864.

Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022. Text embeddings by weaklysupervised contrastive pre-training. arXiv preprint arXiv:2212.03533.

Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2023a. Simlm: Pre-training with representation bottleneck for dense passage retrieval. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2244–2258.

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2023b. Improving text embeddings with large language models. arXiv preprint arXiv:2401.00368.

Suyuchen Wang, Ivan Kobyzev, Peng Lu, Mehdi Rezagholizadeh, and Bang Liu. 2024a. Resonance rope: Improving context length generalization of large language models. arXiv preprint arXiv:2403.00071.

Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, and Furu Wei. 2024b. Augmenting language models with long-term memory. Advances in Neural Information Processing Systems, 36.

Wenhao Wu, Yizhong Wang, Yao Fu, Xiang Yue, Dawei Zhu, and Sujian Li. 2024. Long context alignment with short instructions and synthesized positions. arXiv preprint arXiv:2405.03939.

Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, Song Han, and Maosong Sun. 2024. Infllm: Unveiling the intrinsic capacity of llms for understanding extremely long sequences with training-free memory. arXiv preprint arXiv:2402.04617.

Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighof. 2023. C-pack: Packaged resources to advance general chinese embedding. arXiv preprint arXiv:2309.07597.

Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi Rungta, Karthik Abinav Sankararaman, Barlas Oguz, et al. 2023. Effective long-context scaling of foundation models. arXiv preprint arXiv:2309.16039.

Howard Yen, Tianyu Gao, and Danqi Chen. 2024. Longcontext language modeling with parallel context encoding. Preprint, arXiv:2402.16617.

Peitian Zhang, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, and Zhicheng Dou. 2024a. Soaring from 4k to 400k: Extending llm’s context with activation beacon. arXiv preprint arXiv:2401.03462.

Yikai Zhang, Junlong Li, and Pengfei Liu. 2024b. Extending llms’ context window with 100 samples. arXiv preprint arXiv:2401.07004.

Ming Zhong, Da Yin, Tao Yu, Ahmad Zaidi, Mutethia Mutuma, Rahul Jha, Ahmed Hassan Awadallah, Asli Celikyilmaz, Yang Liu, Xipeng Qiu, and Dragomir Radev. 2021. QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization. In North American Association for Computational Linguistics (NAACL).

Dawei Zhu, Nan Yang, Liang Wang, Yifan Song, Wenhao Wu, Furu Wei, and Sujian Li. 2023. Pose: Efficient context window extension of llms via positional skip-wise training. In The Twelfth International Conference on Learning Representations.

A Training Details for $\mathbf { E 5 - R o P E _ { B a s e } }$
<table><tr><td rowspan="2">Params</td><td colspan="2">Pre-training</td><td colspan="2">Fine-tuning</td></tr><tr><td> $\mathrm { E 5 _ { B a s e } }$ </td><td> $\mathrm { E 5 - R o P E _ { B a s e } }$ </td><td> $\mathrm { E 5 _ { B a s e } }$ </td><td> $\mathrm { E 5 - R o P E _ { B a s e } }$ </td></tr><tr><td>learning rate</td><td> $\vert 2 \times 1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>GPUs (V100)</td><td>32</td><td>32</td><td>8</td><td>8</td></tr><tr><td>warmup steps</td><td>1000</td><td>1000</td><td>400</td><td>400</td></tr><tr><td>max length</td><td>128</td><td>512</td><td>192</td><td>192</td></tr><tr><td>batch size</td><td>32k</td><td>16k</td><td>256</td><td>256</td></tr><tr><td>max steps</td><td>20k</td><td>20k</td><td>n.a.</td><td>n.a.</td></tr><tr><td>epochs</td><td>n.a.</td><td>n.a.</td><td>3</td><td>3</td></tr><tr><td>T</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td>α</td><td>n.a.</td><td>n.a.</td><td>0.2</td><td>0.2</td></tr><tr><td>weight decay</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td>hard negatives</td><td>0</td><td>0</td><td>7</td><td>7</td></tr><tr><td>pos embedding</td><td>APE</td><td>RoPE</td><td>APE</td><td>RoPE</td></tr></table>

Table 4: Hyperparameters for contrastive pre-training and fine-tuning of $\mathrm { E 5 _ { B a s e } }$ and $\mathrm { E 5 - R o P E _ { B a s e } }$

In this section, we describe the training details of ${ \bf E } 5 { \mathrm { - } } \mathrm { R o P E } _ { \mathrm { B a s e } }$ . Our training procedure and data exactly follows that of E5 (Wang et al., 2022), where we first perform contrastive pre-training on their collected CCPairs, then perform finetuning on the concatenation of 3 datasets: MS-MARCO passage ranking (Nguyen et al., 2016), NQ (Karpukhin et al., 2020; Kwiatkowski et al., 2019), and NLI (Gao et al., 2021). Each example is paired with 7 hard negatives. We leverage the mined hard negatives and re-ranker scores from SimLM (Wang et al., 2023a) for the first two datasets. As the NLI dataset only provides 1 hard negative per example, we randomly sample 6 sentences from the entire corpus. xFormers (Lefaudeux et al., 2022) is used for memory efficient training. As presented in Table 4, training hyperparameters for $\mathrm { E 5 _ { B a s e } }$ and ${ \mathrm { E 5 - R o P E _ { B a s e } } }$ are identical, except in two aspects:

• Initialization. Before contrastive pre-training, $\mathrm { E } 5 _ { \mathrm { B a s e } }$ is initialized on $\mathbf { B E R T _ { B a s e } }$ (Devlin et al., 2019), which employs absolute position embeddings (APE). For the initialization of E5- ${ \mathrm { R o P E } } _ { \mathrm { B a s e } } .$ , we simply replace the APE part of $\mathbf { B E R T _ { B a s e } }$ with RoPE. It’s worth noting that the $\mathbf { B E R T _ { B a s e } }$ model after this replacement cannot function properly. We count on the subsequent pre-training phase to adapt the model to RoPE.

• Pre-training length and batch size. $\mathrm { E 5 _ { B a s e } }$ does not update its position embedding matrix during the training phase, i.e., it utilizes the same position embedding matrix as $\mathbf { B E R T _ { B a s e } }$ . This allows it to generalize to input sequences of up to 512 tokens, while being trained with a max training length of 192. As for E5-RoPE, replacing APE with RoPE during initialization prevents us from directly inheriting the original model’s capability in handling 512 tokens. Consequently, in the pre-training phase of E5-RoPE, we set the maximum training length to 512, and reduce the batch size to 16k according to memory constraints.

<table><tr><td>Tasks</td><td>#W/Q.</td><td># W/D.</td><td> $\mathbf { E 5 _ { B a s e } }$ </td><td> $\mathbf { E 5 - R o P E _ { B a s e } }$ </td></tr><tr><td>MS MARCO</td><td>6.0</td><td>56.0</td><td>41.8</td><td>42.4</td></tr><tr><td>Trec-Covid</td><td>10.6</td><td>160.8</td><td>69.6</td><td>73.3</td></tr><tr><td>NFCorpus</td><td>3.3</td><td>232.3</td><td>35.4</td><td>34.9</td></tr><tr><td>NQ</td><td>9.2</td><td>78.9</td><td>58.2</td><td>60.1</td></tr><tr><td>HotpotQA</td><td>17.6</td><td>46.3</td><td>69.1</td><td>61.0</td></tr><tr><td>FiQA</td><td>10.8</td><td>132.3</td><td>39.8</td><td>36.4</td></tr><tr><td>ArguAna</td><td>193.0</td><td>166.8</td><td>44.6</td><td>54.2</td></tr><tr><td>Touche-2020</td><td>6.6</td><td>292.4</td><td>26.4</td><td>26.6</td></tr><tr><td>CQADupStack</td><td>8.6</td><td>129.1</td><td>37.4</td><td>36.5</td></tr><tr><td>Quora</td><td>9.5</td><td>11.4</td><td>86.6</td><td>87.7</td></tr><tr><td>DBPedia</td><td>5.4</td><td>49.7</td><td>42.2</td><td>40.0</td></tr><tr><td>Scidocs</td><td>9.4</td><td>176.2</td><td>18.7</td><td>18.1</td></tr><tr><td>Fever</td><td>8.1</td><td>84.8</td><td>85.0</td><td>68.0</td></tr><tr><td>Climate-Fever</td><td>20.1</td><td>84.8</td><td>26.6</td><td>19.0</td></tr><tr><td>Scifact</td><td>12.4</td><td>213.6</td><td>72.0</td><td>71.0</td></tr><tr><td>Average</td><td>&lt; 200</td><td>&lt;300</td><td>50.23</td><td>48.61</td></tr></table>

Table 5: Statistics and performance comparison of $\mathrm { E 5 _ { B a s e } }$ and ${ \mathrm { E 5 - R o P E } } _ { \mathrm { B a s e } }$ on 15 publicly available BEIR tasks. # W/Q. and # $\mathrm { W } / \mathrm { D }$ . stands for word number per query and per document, respectively.

Table 5 demonstrates results of $\mathrm { E 5 _ { B a s e } }$ and E5- $\mathrm { R o P E } _ { \mathrm { B a s e } }$ on 15 publicly available BEIR tasks. We observe comparable overall scores between both models. This comparable performance, along with their shared training process and training data, facilitates fair comparison of APE and RoPE-based models’s capabilities in length extrapolation. Note that the slight performance loss of ${ \mathrm { E 5 - R o P E _ { B a s e } } }$ could possibly be attributed to the replacement of position embedding in the initialization phase, or the reduced batch size in the pre-training phase, as mentioned before.

## B Implementation Details for Context Extension Strategies

This section describes implementation details for the explored context extension stratgies. For plugand-play methods including PCW, RP, GP, PI, NTK and SE, Table 6 summarizes their hyperparameters under each condition.

<table><tr><td>Extension</td><td>PCW &amp; GP &amp; RP &amp; PI</td><td>NTK</td><td>SE</td></tr><tr><td colspan="4"> $G T E _ { B a s e } \ \& \ E 5 _ { B a s e }$ </td></tr><tr><td> $5 1 2  1 , 0 2 4$ </td><td> $L _ { o } = 5 1 2 , L _ { t } = 1 , 0 2 4 , s = 2$ </td><td>1</td><td></td></tr><tr><td> $5 1 2  2 { , } 0 4 8$ </td><td> $L _ { o } = 5 1 2 , L _ { t } = 2 , 0 4 8 , s = 4$ </td><td>1</td><td></td></tr><tr><td> $5 1 2  4 { , } 0 9 6$ </td><td> $L _ { o } = 5 1 2 , L _ { t } = 4 , 0 9 6 , s = 8$ </td><td>1</td><td></td></tr><tr><td colspan="4"> $E 5  – R o P E _ { B a s e }$ </td></tr><tr><td> $5 1 2  1 , 0 2 4$ </td><td> $L _ { o } = 5 1 2 , L _ { t } = 1 , 0 2 4 , s = 2$ </td><td> $\lambda = 3 ( 1 0 , 0 0 0  3 0 , 0 0 0 )$ </td><td> $g = 3 , w = 2 5 6$ </td></tr><tr><td> $5 1 2  2 { , } 0 4 8$ </td><td> $L _ { o } = 5 1 2 , L _ { t } = 2 , 0 4 8 , s = 4$ </td><td> $\lambda = 5 ( 1 0 , 0 0 0  5 0 , 0 0 0 )$ </td><td> $g = 5 , w = 1 2 8$ </td></tr><tr><td> $5 1 2  4 { , } 0 9 6$ </td><td> $L _ { o } = 5 1 2 , L _ { t } = 4 , 0 9 6 , s = 8$ </td><td> $\lambda = 1 0 \left( 1 0 , 0 0 0 \AA > 1 0 0 , 0 0 0 \right)$ </td><td> $g = 9 , w = 6 4$ </td></tr><tr><td colspan="4">E5-Mistral</td></tr><tr><td> $4 { , } 0 9 6 {  } 8 { , } 1 9 2$ </td><td> $L _ { o } = 4 , 0 9 6 , L _ { t } = 8 , 1 9 2 , s = 2$ </td><td> $\lambda = 3 ( 1 0 , 0 0 0  3 0 , 0 0 0 )$ </td><td> $g = 3 , w = 2 , 0 4 8$ </td></tr><tr><td> $4 , 0 9 6  1 6 , 3 8 4$ </td><td> $L _ { o } = 4 , 0 9 6 , L _ { t } = 1 6 , 3 8 4 , s = 4$ </td><td> $\lambda = 5 ( 1 0 , 0 0 0  5 0 , 0 0 0 )$ </td><td> $g = 5 , w = 1 , 0 2 4$ </td></tr><tr><td> $4 , 0 9 6  3 2 , 7 6 8$ </td><td> $L _ { o } = 4 , 0 9 6 , L _ { t } = 3 2 , 7 6 8 , s = 8$ </td><td> $\lambda = 1 0 \left( 1 0 , 0 0 0 \AA > 1 0 0 , 0 0 0 \right)$ </td><td> $g = 9 , w = 5 1 2$ </td></tr></table>

Table 6: Hyperparameters for plug-and-play context extension strategies.

Further Tuning. On top of PI and RP, we perform further tuning on both $\mathrm { E 5 _ { B a s e } }$ and ${ \mathrm { G T E } } _ { \mathrm { B a s e } } ,$ utilizing the fine-tuning dataset mentioned in $\mathsf { A p - }$ pendix A. Following the practice of PoSE (Zhu et al., 2023), we manipulate position ids to simulate long training samples. Concretely, given an input document $\mathcal { D } = \{ x _ { 0 } , x _ { 1 } , . . . , x _ { L _ { o } - 1 } \}$ of original context length $L _ { o } ,$ , we introduce a skipping bias term u at the beginning of , transferring the original position ids into $\{ 0 , 1 , . . . , L _ { o } - 1 \}$ into $\{ u , u + 1 , . . . , u + L _ { o } - 1 \} . ^ { 4 }$ For every piece of training data, u is re-sampled from the discrete uniform distribution $\mathcal { U } ( \{ 0 , 1 , . . . , L _ { t } - L _ { o } \} )$ . In this way, we ensure comprehensive coverage of target context window. The training procedure spans 3 epochs on 2 A100 GPUs, with a learning rate of $5 e ^ { - 4 }$ , a batch size of 512, and 100 steps for warmup. Other hyperparameters are same as Table 4.

Inference. In inference time, attention scaling (Su, 2021; Chiang and Cholak, 2022) is used by default for all tested models for better length extrapolation ability. Especially for $\mathrm { G T E _ { B a s e } }$ and $\mathrm { E 5 _ { B a s e } }$ tuned on PI, we use the original position ids when input length not exceeds 512. This is achived by mapping the position ids $\{ 0 , 1 , . . . , l \}$ into $\{ 0 , s , . . . , l \times s \}$ , where s is the scaling factor, $l < 5 1 2 \AA$

## C Further details on LONGEMBED

Figure 7 presents source and examples for each dataset included in LONGEMBED. For QA datasets including NarrativeQA and 2WikiMultihopQA, we adopt their test splits. Note that for 2WikiMultihopQA, we adopt the length-uniformly sampled version from Bai et al. (2023b) to better assess the model’s capabilities across various context lengths. For summarization datasets including QM-Sum and SummScreenFD, we adopt the version processed by SCROLLS (Shaham et al., 2022). Since SCROLLS does not include ground truth summarization in its test sets, we switch to validation set for these two datasets. Particularly for QMSum, as its validation set only have 60 documents, which is too small for document retrieval, we included the train set as well.

<table><tr><td rowspan="2">Method</td><td colspan="2">Synthetic</td><td colspan="2">Real</td><td rowspan="2"> $\mathbf { A v } \mathbf { g } .$ </td></tr><tr><td>P</td><td>N NQA</td><td>QMS</td><td>SFD WQA</td></tr><tr><td>BM25</td><td>100</td><td>95.3</td><td>71.5 81.3</td><td>97.6 96.5</td><td>90.4</td></tr><tr><td>E5-Mistral +NTK (32k)</td><td>71.0 93.8</td><td>48.3 66.8</td><td>44.6 43.6 49.8 49.2</td><td>96.8 82.0 97.1 95.2</td><td>64.4 75.3</td></tr></table>

Table 7: BM25 Results on LONGEMBED. P, N, NQA, QMS, SFD, WQA is short for Passkey, Needle, NarrativeQA, QMSum, SummScreenFD, 2WikiMultihopQA.

## D BM25 Results on LONGEMBED

Table 7 shows the scores of BM25 on LONGEM-BED, along with those of the best-performing long context embedding model, E5-Mistral. The significant gap between BM25 and E5-Mistral highlights substantial room for improvement in current long context embedding models.

<table><tr><td>Dataset Name</td><td>Source / Split</td><td>Query Example</td><td>Document Example</td></tr><tr><td>Narrative QA</td><td>- / test</td><td>Why is Bobolink eventually eager to help Martin?</td><td>The Project Gutenberg EBook of The Purple Cloud, by M.P. Shiel\n [...] Title: The Purple Cloud\n\nAuthor: M.P. Shiel\n\nRelease Date: February 22, 2004, [...]</td></tr><tr><td>QMSum</td><td>Scrolls / train + valid</td><td>The team wanted to understand how they could combine different linguistic features to make a more robust recognition model. They were [...]</td><td>Project Manager: Can I close this ?\nUser Interface: Uh we don&#x27;t have any changes , do we ?\nProject Manager: Oh , okay .\nUser Interface: So no . {vocalsound }\nProject Manager: {vocalsound} There we go . Okay , here we are again . Detailed design {disfmarker} oh , come on . Well {disfmarker} Ah {gap} s Forgot to insert the minutes [...]</td></tr><tr><td>2WikiMultihop QA</td><td>LongBench / test</td><td>Where was the director of film The Central Park Five born</td><td>Passage 1:\nMargaret, Countess of Brienne\nMarguerite d&#x27;Enghien (born 1365 - d. after 1394), was the ruling suo jure Countess of Brienne and of Conversano, suo jure Lady of Enghien, and Lady of Beauvois from 1394 until an unknown date. […..] Passage 2:\nNocher II, Count of Soissons\nNocher II (died 1019), Count of Bar-sur-Aube, Count of Soissons. He was the son of Nocher I, Count of Bar-sur-Aube. Nocher&#x27;s brother Beraud (d. 1052) was Bishop of Soissons.Nocher became Count of Soissons, jure uxoris, upon his marriage to Adelise,</td></tr><tr><td>D</td><td></td><td>Penny gets a new chair, which Sheldon enjoys until he finds out that she picked it up from the street. He constantly pesters Penny to dispose of it, to no avail. Note: Melissa Rauch is absent in this episode.</td><td>[PREVIOUSLY_ON]\nYou make jumps you can&#x27;t explain, Will. The evidence explains. Then help me find some evidence. I wouldn&#x27;t put him out there! Should he get too close, I need you to make sure he&#x27;s not out there alone. I don&#x27;t think the Shrike killed that girl in the field. This girl&#x27;s killer thought that she was a pig. You think this was a copycat? I think I can help good Will, see his face. Hello? They know.\n(gunshots)\nYou said he wouldn&#x27;t get too close. See?\n(gunshots)\n(knocking)\nJack: We&#x27;re here!\n(police radio chatter)\nWill: Could be a permanent installation in</td></tr><tr><td>Passkey</td><td>-/ -</td><td>what is the passkey for Kyree Mays?</td><td>[..] The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. The grass is green. The sky is blue.\nMalayah Graves&#x27;s pass key is 41906. Remember it. 41906 is the pass key for Malayah Graves.\nThe sun is yellow. Here we go. There and back again. The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. [...]</td></tr><tr><td>Needle</td><td>-/ -</td><td>What is the best thing to do in San Francisco?</td><td>Aaron Swartz created a scraped feed of the essays page. November 2021(This essay is derived from a talk at the Cambridge Union. ) [...] The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\nThere&#x27;s a narrow sense in which it refers to aesthetic judgements and a broader one in which it refers to</td></tr></table>

Figure 7: Source and examples for each dataset in LONGEMBED.