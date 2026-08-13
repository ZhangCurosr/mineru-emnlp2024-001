# In-context Contrastive Learning for Event Causality Identification

Chao Liang <sup>1</sup> Wei Xiang <sup>2</sup> Bang Wang <sup>1</sup> ∗

<sup>1</sup> School of Electronic Information and Communications, Huazhong University of Science and Technology, Wuhan, China {liangchao111, wangbang}@hust.edu.cn <sup>2</sup> Faculty of Artificial Intelligence in Education, Central China Normal University, Wuhan, China. xiangwei@ccnu.edu.cn

## Abstract

Event Causality Identification (ECI) aims at determining the existence of a causal relation between two events. Although recent prompt learning-based approaches have shown promising improvements on the ECI task, their performance are often subject to the delicate design of multiple prompts and the positive correlations between the main task and derivate tasks. The in-context learning paradigm provides explicit guidance for label prediction in the prompt learning paradigm, alleviating its reliance on complex prompts and derivative tasks. However, it does not distinguish between positive and negative demonstrations for analogy learning. Motivated from such considerations, this paper proposes an In-Context Contrastive Learning (ICCL) model that utilizes contrastive learning to enhance the effectiveness of both positive and negative demonstrations. Additionally, we apply contrastive learning to event pairs to better facilitate event causality identification. Our ICCL is evaluated on the widely used corpora, including the EventStoryLine and Causal-TimeBank, and results show significant performance improvements over the state-of-the-art algorithms. 1

## 1 Introduction

Event Causality Identification (ECI) is to detect whether there exists a causal relation between two event mentions in a document. It is of great importance for many Natural Language Processing (NLP) applications, such as question answer (Breja and Jain, 2020), machine reading comprehension (Berant et al., 2014), and etc. Furthermore, It also has many practical applications in real-world scenarios, such as event prediction (Preethi et al., 2015; Radinsky et al., 2012) and strategy optimization (Balgi et al., 2022). Fig. 1 illustrates an event causality example from the Event StoryLine Corpus (ESC). We concatenated two causal demonstrations and two non-causal demonstrations before the query to be predicted, and enhanced the analogy between the query and demonstrations through contrastive. Ultimately, our ICCL model determined the causality between the two events, "died" and "shield", in the query.

![](images/cbafc51087eee16a4a0c74fcffd34adeb011aa8c338d85561cabca46bf0b6a49.jpg)  
Figure 1: Illustration of our motivation. The event pairs are highlighted in different colors.

Some graph-based methods have been proposed for the ECI task (Zhao et al., 2021; Phu and Nguyen, 2021; Pu et al., 2023), which apply a graph structure to represent events and their potential relations. For example, Zhao et al. (2021) initialize event nodes’ embeddings using a documentlevel encoder and employ a graph inference mechanism to update their embeddings. Pu et al. (2023) incorporate causal label information and event pair interaction information to enhance the representation learning for event nodes in the graph. These methods follow the traditional representation learning for classification yet on a graph structure.

Recently the prompt learning paradigm (Liu et al., 2023) has shown its great successes in many NLP tasks, as it can well leverage the potentials of a pre-trained language model (PLM). Some researchers have applied the prompt learning for the ECI task (Liu et al., 2021b; Shen et al., 2022). For example, the DPJL model (Shen et al., 2022) designs a main cloze task but also designs two derivative prompt tasks. Although the DPJL has achieved new state-of-the-art performance, it involves the delicate design of multiple prompts and relies on the positive correlations between the main task and derivative tasks.

The in-context learning paradigm (Dong et al., 2022) includes some demonstrations with their ground-truth labels into the query prompt to learn some patterns hidden in demostrations when making its prediction. However, it does not distinguish between positive and negative demonstrations for analogy. Motivated from such considerations, we propose to use contrastive learning on the in-context demonstrations to enhance the effectiveness of analogy, as illustrated in Fig. 1. Besides, we also argue that the semantic of event mentions are the most important for the causal relation identification between them. As such we apply contrastive learning to the representation of event mentions in in-context demonstrations, so as to distinguishing the semantic between causal and non-causal event pairs and facilitating event causality predictions.

In this paper, we propose an In-Context Contrastive Learning (ICCL) model for the ECI task. The ICCL model contains three modules. The prompt learning module reformulates an input event pair and some retrieved demonstrations into a prompt template, as the input for PLM encoding. The in-context contrastive module optimizes the representation of event mention by simultaneously maximizing its agreement with positive demonstrations and minimizing with negative ones, via a contrastive loss. The causality prediction module predicts answer word to identify causal relations. Experiments are conducted on the widely used EventStoryLine and Causal-TimeBank corpora, and results have shown that our ICCL achieves the new state-of-the-art performance for the ECI task.

## 2 Related work

## 2.1 Event Causality Identification

Event Causality Identification (ECI) is an essential task in information extraction, attracting significant attention due to its wide range of potential applications. Early methods mainly relied on designing task-oriented neural network models (Liu et al., 2021b; Zuo et al., 2021a). For example, Liu et al. (2021b) improve the capability of their neural model to identify previously unseen causal relations by incorporating event-agnostic and context-specific patterns derived from the ConceptNet (Speer et al., 2017). With further exploration of graph structures and the emergence of large-scale PLMs, recent studies have increasingly adopted graph-based and prompt-based learning approaches to address the ECI task.

Graph-based approaches usually model the ECI task as a node classification problem, employing graph neural networks to learn event node representations based on contextual semantics at the document level (Phu and Nguyen, 2021; Cao et al., 2021; Fan et al., 2022). For example, Fan et al. (2022) establish explicit connections between events, mentions and contexts to construct a cooccurrence graph for node representation learning and causal relation identification. In addition to node classification, some studies approach the ECI task as a graph-based edge prediction problem (Zhao et al., 2021; Chen et al., 2022). For example, Zhao et al. (2021) initialize event node embeddings using a document-level encoder based on a PLM and employ a graph inference mechanism to predict causal edges through graph updating.

## 2.2 Prompt-based Causality Identification

Recently, with the help of large-scale PLMs, such as the BERT (Devlin et al., 2018), RoBERTa (Liu et al., 2019) and etc, prompt learning has emerged as a new paradigm for various NLP tasks (Xiang et al., 2022; Ding et al., 2021). It converts downstream tasks into the similar form as pretraining task, which aligns objectives between the two stages. This alignment helps bridging the gap between PLM and task and can directly enhance the performance of a downstream task. Moreover, researchers have also devised appropriate prompts to reframe ECI task as a cloze task (Shen et al., 2022; Liu et al., 2021b). For example, Shen et al. (2022) propose a derivative prompt joint learning model that leverages potential causal knowledge within PLMs based on the causal cue words detection. Liu et al. (2021b) use an event mention masking generalization mechanism to encode some event causality patterns for causal relation reasoning. Although prompt-based methods are constrained by complex prompts and derivate tasks, these prompt-based models effectively leverage the implicit knowledge of PLMs to address the ECI task.

![](images/d0254a7dbb0701f1aae7e6a80be97cf4985a02427901a10860ff0b25e61cd23f.jpg)  
Figure 2: Illustration of our ICCL framwork.

## 3 Method

Fig. 2 illustrates our ICCL model, including the prompt learning module, the in-context contrastive module, and the causality prediction module.

## 3.1 Task Formulation

We apply the prompt learning paradigm to transform the ECI task into a causal relation cloze task, utilizing a PLM to predict answer words for causal relation identification. As the event mentions are annotated by a few words in a sentence, we use the event mentions $E _ { 1 }$ and $E _ { 2 }$ of an event pair as well as their raw sentences $S _ { 1 }$ and $S _ { 2 }$ , as the input $x = \{ E _ { 1 } , E _ { 2 } , S _ { 1 } , S _ { 2 } \}$ , where $E _ { 1 } \in S _ { 1 }$ and $E _ { 2 } \in S _ { 2 }$ . The virtual answer words <causal> and <none> indicating whether there is a causal relation between the input event pair, are used as the output $y \in \{ < \mathtt { c a u s a l } > , < \mathtt { n o n e } > \}$ . We note that in cases where $E _ { 1 }$ and $E _ { 2 }$ originate from the same sentence, $S _ { 1 }$ and $S _ { 2 }$ refer to the same sentence.

## 3.2 Prompt Learning Module

As illustrated in the bottom of Fig. 2, we first reformulate each input instance $x = \{ E _ { 1 } , E _ { 2 } , S _ { 1 } , S _ { 2 } \}$ into a kind of in-context prompt template $T ( x )$

as the input of a PLM for encoding. The incontext prompt input contains a query instance and K retrieved demonstrations. The query instance is the input event instance, denoted as $q =$ $\{ E _ { 1 } ^ { q } , E _ { 2 } ^ { q } , S _ { 1 } ^ { q } , S _ { 2 } ^ { q } \}$ , with the causal relation between two events to be identified. The demonstrations are retrieved from the training dataset, consisting of an event mention pair and their raw sentences, as well as the relation label between them, denoted as $d _ { k } = \{ E _ { 1 } ^ { k } , E _ { 2 } ^ { k } , S _ { 1 } ^ { k } , S _ { 2 } ^ { k } , y ^ { k } \}$ . We randomly select M demonstrations labeled with <causal> relation and N demonstrations labeled with <none> relation, denoted as $d _ { m } ^ { + }$ and $d _ { n } ^ { - }$ , respectively.

We design a prediction prompt template $T _ { p } ( q )$ for the query instance $q$ and an analogy prompt template $T _ { a } ( d _ { k } )$ for its retrieved demonstrations $d _ { k } .$ respectively. Both of them are constructed by concatenating the raw sentences with a simple cloze template, as follows:

$$
\begin{array} { r l } & { T _ { p } ( q ) = S _ { 1 } ^ { q } + S _ { 2 } ^ { q } + } \\ & { \qquad \quad [ \mathsf { s t a r t } ] + E _ { 1 } ^ { q } + [ \mathsf { M A S K } ] + E _ { 2 } ^ { q } + [ \mathsf { e n d } ] . } \\ & { T _ { a } ( d _ { k } ) = S _ { 1 } ^ { k } + S _ { 2 } ^ { k } + } \\ & { \qquad \quad [ \mathsf { s t a r t } ] + E _ { 1 } ^ { k } + y ^ { k } + E _ { 2 } ^ { k } + [ \mathsf { e n d } ] . } \end{array}
$$

where $E _ { 1 } ^ { q } , E _ { 2 } ^ { q } , S _ { 1 } ^ { q } , S _ { 1 } ^ { q }$ are the two event mentions and their raw sentences, and the PLM-specific token [start] and [end] are used to indicate the beginning and ending of the cloze template. For prediction prompt template $T _ { p } ( q )$ , a PLM-specific token [MASK] is inserted between two event mentions for relation prediction; For analogy prompt template $T _ { a } ( d _ { k } )$ , it is replaced by the virtual word of the relation label $y ^ { \bar { k } }$ for each demostrations, i.e. <causal> or <none>.

The in-context prompt template $T ( x )$ is constructed by concatenating the prediction prompt tempalte $T _ { p } ( q )$ and some analogy prompt templates $T _ { a } ( d _ { k } )$ of its retrieved demonstrations, as follows:

$$
\begin{array} { r } { T ( x ) = [ \mathtt { C L S } ] + T _ { a } ( d _ { 1 } ^ { + } ) \ [ \mathtt { S E P } ] \dots T _ { a } ( d _ { M } ^ { + } ) \ [ \mathtt { S E P } ] } \\ { + T _ { a } ( d _ { 1 } ^ { - } ) \ [ \mathtt { S E P } ] \dots T _ { a } ( d _ { N } ^ { - } ) \ [ \mathtt { S E P } ] + T _ { p } ( q ) \ [ \mathtt { S E P } ] . } \end{array}
$$

where the PLM-specific token [CLS] and [SEP] are used to indicate the beginning and ending of an input, and some [SEP] tokens are used as separators between the query and those demonstrations. Note that, the causal demonstrations $d _ { m } ^ { + }$ are positioned before the none causal demonstrations $d _ { n } ^ { - }$ . We provide a specific example of in-context prompt template input in Appendix C.

After the PLM encoding, we obtain a hidden state h $\in \mathbb { R } ^ { d }$ for each input tokens, where d is the dimension of hidden states. We denote the hidden state of input [MASK] token as $\mathbf { h } _ { m a s k }$ for causality prediction. The hidden states of input event pair in query instance, retrieved causal and nonecausal demonstrations are denoted as $[ \mathbf { h } _ { e _ { 1 } } ^ { q } , \mathbf { h } _ { e _ { 2 } } ^ { q } ]$ $[ \mathbf { h } _ { e _ { 1 } } ^ { m ^ { + } } , \mathbf { h } _ { e _ { 2 } } ^ { m ^ { + } } ]$ and $[ \mathbf { h } _ { e _ { 1 } } ^ { n ^ { - } } , \mathbf { h } _ { e _ { 2 } } ^ { n ^ { - } } ]$ , respectively, which are next used for in-context contrastive learning.

## 3.3 In-context Contrastive Module

The in-context contrastive module optimizes the representation of event mention by simultaneously maximizing its agreement with positive demonstration samples and minimizing with negative ones, via a contrastive loss. In the training phase, we use the input query instance as an anchor. The retrieved demonstrations with the same relation label as the query are positive samples, while those with different relation label are negative samples. We assume that the query’s label is <causal>, so the causal demonstrations $d _ { m } ^ { + }$ being treated as positives, and non-causal ones $d _ { n } ^ { - }$ as negatives.

Motivated by the fact that the offsets of pretrained word embeddings can model the relationship between them (Mikolov et al., 2013; Pennington et al., 2014; Chen et al., 2016), such as $\mathbf { h } _ { k i n g } - \mathbf { h } _ { m a n } \approx \mathbf { h } _ { q u e e n } - \mathbf { h } _ { w o m a n }$ . We use the offsets between event mentions’ hidden states to represent their relation for contrastive learning, as

follows:

$$
{ \bf z } ^ { q } = { \bf h } _ { e _ { 1 } } ^ { q } - { \bf h } _ { e _ { 2 } } ^ { q } ,
$$

$$
{ \bf z } _ { m } ^ { + } = { \bf h } _ { e _ { 1 } } ^ { m ^ { + } } - { \bf h } _ { e _ { 2 } } ^ { m ^ { + } } ,\tag{1}
$$

(2)

$$
\mathbf { z } _ { n } ^ { - } = \mathbf { h } _ { e _ { 1 } } ^ { n ^ { - } } - \mathbf { h } _ { e _ { 2 } } ^ { n ^ { - } } ,\tag{3}
$$

where ${ \bf z } ^ { q } , { \bf z } _ { m } ^ { + } , { \bf z } _ { n } ^ { - }$ are the relation vector of event pair in query instance, positive and negative demonstrations, respectively.

We adpot supervised constrastive learning on the relation vector of event pair for its representation optimization (Khosla et al., 2020). Specifically, it pulls together the anchor towards positive samples in embedding space, while simultaneously pushing it apart from negative samples. The supervised contrastive loss is computed as follows:

$$
L _ { c o n } = - \log \sum _ { m = 1 } ^ { M } \frac { \exp ( \sin ( \mathbf { z } ^ { q } , \mathbf { z } _ { m } ^ { + } ) / \tau ) } { \sum _ { d \in \mathcal { D } } \exp ( \sin ( \mathbf { z } ^ { q } , d ) / \tau ) } ,\tag{4}
$$

where $\mathcal { D } = \{ \mathbf { z } _ { m } ^ { + } \} _ { m = 1 } ^ { M } \cup \{ \mathbf { z } _ { n } ^ { - } \} _ { n = 1 } ^ { N }$ , M and N represent the number of positive and negative demonstrations, respectively.

## 3.4 Causality Prediction Module

The causality prediction module uses the [MASK] token of input query instance to predict an answer word for causal relation identification. Specifically, we input the hidden state $\mathbf { h } _ { m a s k }$ into the masked language model classifier, and estimate the probability of each word in its vocabulary dictionary for the [MASK] token, as follows:

$$
P ( \mathbb { M A S K } ] = v \in \mathcal { V } \mid T ( x ) ) ,\tag{5}
$$

We add two virtual words into PLM’s vocabulary dictionary as the answer space, viz. <causal> and <none> , to indicate whether a causal relation exists or not. Then a softmax layer is applied on the prediction scores of the two virtual answer words to normalize them into probabilities:

$$
P _ { i } ( v _ { i } \in \mathcal { V } _ { a } | T ( x ) ) = \frac { \exp ( p _ { v _ { i } } ) } { \sum _ { j = 1 } ^ { n } \exp ( p _ { v _ { j } } ) } ,\tag{6}
$$

where $\mathcal { V } _ { a } = \{ < \mathtt { c a u s a l } >$ , <none>}.

In the training phase, we tune parameters of PLM and MLM classifier based on in-context prompt and newly added vitual words. We adopt the cross entropy loss as the loss function:

$$
L _ { p r e } = - \frac { 1 } { L } \sum _ { l = 1 } ^ { L } \mathbf { y } ^ { ( l ) } \log ( \hat { \mathbf { y } } ^ { ( l ) } ) + \lambda \| \boldsymbol { \theta } \| ^ { 2 } ,\tag{7}
$$

where $\mathbf { y } ^ { ( l ) }$ and $\hat { \mathbf { y } } ^ { ( l ) }$ are answer label and predicted label of the l-th training instance respectively. λ and θ are the regularization hyper-parameters. We use the AdamW optimizer (Loshchilov and Hutter, 2017) with L2 regularization for model training.

## 3.5 Training strategy

We jointly train the in-context contrastive module and the causality prediction module. The loss function of our ICCL model is optimized as follows:

$$
L _ { t o t a l } = L _ { p r e } + \beta * L _ { c o n } ,\tag{8}
$$

where $\beta$ is the weight coefficient to balance the importance of contrastive loss and prediction loss. We conduct some experiments to explore the impact of different $\beta$ values on model performance. The experimental results and analysis are presented in Appendix D.

## 4 Experiment Setting

## 4.1 Datasets

Our experiments are conducted on two widely used datasets for the ECI task: EventStory-Line 0.9 Corpus (ESC) (Caselli and Vossen, 2017) and Causal-TimeBank Corpus (CTB) (Mirza and Tonelli, 2014).

EventStoryLine contains 22 topics and 258 documents collected from various news websites. In total, there are 5,334 event mentions in ECS dataset. Among them, 5,625 event pairs are annotated with causal relations. Specifically, 1,770 causal relations are intra-sentence causalities, while 3,855 ones are cross-sentence causalities. Following the standard data splitting Gao et al. (2019), we use the last two topics as the development set, and conduct 5-fold cross-validation on the remaining 20 topics. The average results of precision (P), recall (R), and F1 score are adopted as performance metrics.

Causal-TimeBank comprises 184 documents sourced from English news articles, with a total of 7,608 annotated event pairs. Among them, 318 are annotated with causal relations. Specifically, 300 causal relations are intra-sentence causalities, while only 18 ones are cross-sentence causalities. Following the standard data splitting (Liu et al., 2021a), we employ a 10-fold cross-validation and the average results of precision (P), recall (R), and F1 score are adopted as performance metrics. Following Phu and Nguyen (2021), we only conduct intra-sentence event causality identification experiments on CTB, as the number of cross-sentence event causal pairs is quite small.

## 4.2 Parameter Setting

We use the pre-trained RoBERTa (Liu et al., 2019) model with 768-dimension base version provided by the HuggingFace transformers<sup>2</sup> (Wolf et al., 2020). Our implementation is based on PyTorch framework<sup>3</sup>, running on NVIDIA GTX 3090 GPUs. The training process costs approximately 5 GPU hours on average. We set the learning rate to 1e-5, batch size to 16. The contrastive loss ratio $\beta$ is set to 0.5, the temperature parameter $\tau$ is set to 1.0, and the number of demonstrations is set to $^ { 4 , }$ viz. $( M , N ) = ( 2 , 2 )$ . All trainable parameters are randomly initialized from normal distributions.

## 4.3 Competitors

We compare our ICCL with the following competitors: ILP (Gao et al., 2019), KnowMMR (Liu et al., 2021b), RichGCN (Phu and Nguyen, 2021), CauSeRL (Zuo et al., 2021a), LSIN (Cao et al., 2021), LearnDA (Zuo et al., 2021b), GESI (Fan et al., 2022), ERGO (Chen et al., 2022), DPJL (Shen et al., 2022), SemSln (Hu et al., 2023). The detailed introduction of competitors can be found in Appendix B.

## 5 Result and Analysis

## 5.1 Overall Result

Table 1 compares the overall performance between our ICCL and the competitors on the ESC and CTB corpus. We can observe that the ILP cannot outperform other competitors, including the RichGCN, GESI, ERGO, and SemSln. This can be attributed to their utilization of some graph neural networks for document structure encoding, enabling them to learn global contextual semantic for causality prediction. We can also observe that the DPJL adopting a kind of derivative prompt learning can significantly outperform the other competitors in intra-sentence causality identification. The outstanding performance can be attributed to its applying the prompt learning paradigm that transforms the ECI task to directly predict a PLM vocabulary word, other than fine-tuning a task-specific neural model upon a PLM. Although some other competitors have used external knowledge bases for relation identification, the prompt learning paradigm can better leverages potential causal knowledge in PLMs.

<table><tr><td rowspan="3">Model</td><td rowspan="3">PLM</td><td colspan="9">EventStoryLine</td><td colspan="3">Causal-TimeBank</td></tr><tr><td colspan="2">Intra</td><td colspan="2"></td><td colspan="2">Cross</td><td colspan="3">Intra and Cross</td><td colspan="3">Intra</td></tr><tr><td>P(%)</td><td>R(%)</td><td>F1(%)</td><td>P(%)</td><td>R(%)</td><td>F1(%)</td><td>P(%)</td><td>R(%)</td><td>F1(%)</td><td>P(%)</td><td>R(%)</td><td>F1(%)</td></tr><tr><td>ILP (Gao et al., 2019)</td><td></td><td>38.8</td><td>52.4</td><td>44.6</td><td>35.1</td><td>48.2</td><td>40.6</td><td>36.2</td><td>49.5</td><td>41.9</td><td>一</td><td></td><td>-</td></tr><tr><td>LearnDA (Zuo et al., 2021b)</td><td>BERT</td><td>42.2</td><td>69.8</td><td>52.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td>41.9</td><td>68.0</td><td>51.9</td></tr><tr><td>RichGCN (Phu and Nguyen, 2021)</td><td>BERT</td><td>49.2</td><td>63.0</td><td>55.2</td><td>39.2</td><td>45.7</td><td>42.2</td><td>42.6</td><td>51.3</td><td>46.6</td><td>39.7</td><td>56.5</td><td>46.7</td></tr><tr><td>DPJL (Shen et al., 2022)</td><td>RoBERTa</td><td>65.3</td><td>70.8</td><td>67.9</td><td></td><td></td><td></td><td>-</td><td></td><td></td><td>63.6</td><td>66.7</td><td>64.6</td></tr><tr><td>GESI (Fan et al., 2022)</td><td>BERT</td><td></td><td></td><td>50.3</td><td></td><td></td><td>49.3</td><td></td><td></td><td>49.4</td><td></td><td></td><td></td></tr><tr><td>ERGO (Chen et al., 2022)</td><td>Longformer</td><td>57.5</td><td>72.0</td><td>63.9</td><td>51.6</td><td>43.3</td><td>47.1</td><td>48.6</td><td>53.4</td><td>50.9</td><td>62.1</td><td>61.3</td><td>61.7</td></tr><tr><td>SemSln (Hu et al., 2023)</td><td>BERT</td><td>64.2</td><td>65.7</td><td>64.9</td><td></td><td></td><td></td><td>，</td><td></td><td></td><td>52.3</td><td>65.8</td><td>58.3</td></tr><tr><td rowspan="4">ICCL</td><td>BERT</td><td>64.9</td><td>69.6</td><td>67.1</td><td>56.3</td><td>58.4</td><td>57.2</td><td>59.0</td><td>61.9</td><td>60.4</td><td>60.5</td><td>58.4</td><td>59.1</td></tr><tr><td>ERNIE</td><td>66.8</td><td>68.5</td><td>67.5</td><td>63.7</td><td>56.2</td><td>59.5</td><td>64.8</td><td>60.0</td><td>62.1</td><td>64.8</td><td>66.0</td><td>64.7</td></tr><tr><td>DeBERTa</td><td>67.6</td><td>73.7</td><td>70.4</td><td>61.8</td><td>58.4</td><td>59.9</td><td>61.7</td><td>63.2</td><td>63.3</td><td>66.7</td><td>64.4</td><td>64.9</td></tr><tr><td>RoBERTa</td><td>67.5</td><td>73.7</td><td>70.4</td><td>60.3</td><td>62.7</td><td>61.3</td><td>62.6</td><td>66.1</td><td>64.2</td><td>63.7</td><td>68.8</td><td>65.4</td></tr></table>

Table 1: Comparison of overall results on the ESC and CTB corpus.

Finally, our ICCL with different PLMs has achieved significant performance improvements overall competitors in terms of much higher F1 score with all intra-sentence, inter-sentence, and overall event causality identification on both ESC and CTB corpus. We attribute its outstanding performance to applying contrastive learning on incontext demonstrations, by which our ICCL can better distinguish the semantic of causal and noncausal event pairs for causality prediction. Furthermore, we can also observe that using different PLMs do result in some performance variations, which are further discussed in Appendix A. Finally the ICCL based on RoBERTa has achieved the best performance, as such we implement the remaining ablation experiments with RoBERTa.

## 5.2 Ablation Study

To examine the effectiveness of contrastive learning and in-context learning, we design the following ablation study. Table 2 compares their perfomance.

Prompt is prompt learning model, without demonstrations or contrastive module.

In-context is in-context learning model, including retrieved demonstrations but without contrastive module.

ProCon w/o Demos is prompt based contrastive model, but without demonstrations. We select positive and negative samples within batch insted of demonstrations, and use hidden state of [MASK] as input to contrastive module.

ProCon w/ Demos is in-context based contrastive model with retrieved demonstrations, but still use the hidden state of [MASK] as input to contrastive module.

EvtCon is event based prompt contrastive model, the only difference with ProCon w/o Demos is using hidden states of event pairs as contrastive module inputs.

In-context learning: The first observation is that models incorporating in-context learning perform better. For example, the three models, In-context, ProCon w/ Demos and ICCL outperform Prompt, ProCon w/o Demos and Evt-Con, respectively. This indicates the inclusion of demonstrations to explicitly guide the label prediction is highly effective in improving model performance. Furthermore, models with in-context learning show notable performance gains in challeging cross-sentence causality identification. That’s because randomly selected demonstrations are predominantly composed of cross-sentence samples, which are more abundant in datasets. Therefore, PLMs develop a more comprehensive understanding of cross-sentence causality.

Contrastive learning: We can observe that models with a contrastive module exhibit better performance. For example, both ProCon w/ Demos and EvtCon preform bette than Prompt. Additionally, both ProCon w/o Demos and ICCL preform bette than In-context. This can be attributed to the utilization of the contrastive learning paradigm, which enables the PLM to concentrate on event pairs or [MASK] and enhances PLM’s ability to model them. Furthermore, it also helps discriminatively model positive and negative demonstrations, strengthening analogy between the query and all demonstrations. Additionally, we also observe that EvtCon usually outperformes ProCon w/o Demos. That’s because hidden state of [MASK] serves as input for both contrastive and prediction module in the case of ProCon w/o Demos, yet the optimization directions of two modules do not

<table><tr><td rowspan=3 colspan=3>Model</td><td rowspan=1 colspan=9>EventStoryLine</td><td rowspan=1 colspan=3>Cause-TimeBank</td></tr><tr><td rowspan=1 colspan=3>Intra</td><td rowspan=1 colspan=3>Cross</td><td rowspan=1 colspan=3>Intra and Cross</td><td rowspan=1 colspan=3>Intra</td></tr><tr><td rowspan=1 colspan=3>p (%)r (%) f1 (%)</td><td rowspan=1 colspan=3>p (%)r (%)f1 (%)</td><td rowspan=1 colspan=3>p (%)r (%)f1 (%)</td><td rowspan=1 colspan=3>p (%)r (%) f1 (%)</td></tr><tr><td rowspan=1 colspan=3>Prompt</td><td rowspan=1 colspan=3>67.2 69.7  68.2</td><td rowspan=1 colspan=1>58.6</td><td rowspan=1 colspan=2>59.8  59.0</td><td rowspan=1 colspan=1>61.3</td><td rowspan=1 colspan=2>62.9  61.7</td><td rowspan=1 colspan=3>58.9  55.3  56.6</td></tr><tr><td rowspan=1 colspan=3>In-context</td><td rowspan=1 colspan=1>66.0</td><td rowspan=1 colspan=1>72.4</td><td rowspan=1 colspan=1>68.9</td><td rowspan=1 colspan=1>57.7</td><td rowspan=1 colspan=1>60.9</td><td rowspan=1 colspan=1>59.1</td><td rowspan=1 colspan=1>60.4</td><td rowspan=1 colspan=2>64.5  62.2</td><td rowspan=1 colspan=1>60.3</td><td rowspan=1 colspan=2>58.0  58.7</td></tr><tr><td rowspan=1 colspan=2>ProCon w/o Demos</td><td rowspan=1 colspan=1>emos</td><td rowspan=1 colspan=1>os</td><td rowspan=1 colspan=1>60.8</td><td rowspan=1 colspan=1>77.9</td><td rowspan=1 colspan=1>68.2</td><td rowspan=1 colspan=1>54.2</td><td rowspan=1 colspan=1>65.6</td><td rowspan=1 colspan=1>59.3</td><td rowspan=1 colspan=1>56.4</td><td rowspan=1 colspan=1>69.4</td><td rowspan=1 colspan=1>62.1</td><td rowspan=1 colspan=1>51.5</td><td rowspan=1 colspan=1>71.8</td></tr><tr><td rowspan=1 colspan=2>ProCon w/ Demos</td><td rowspan=1 colspan=1>mos</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>67.1</td><td rowspan=1 colspan=1>73.5</td><td rowspan=1 colspan=1>70.1</td><td rowspan=1 colspan=1>58.0</td><td rowspan=1 colspan=1>61.9</td><td rowspan=1 colspan=1>59.8</td><td rowspan=1 colspan=1>60.9</td><td rowspan=1 colspan=1>64.5</td><td rowspan=1 colspan=1>63.1</td><td rowspan=1 colspan=1>66.9</td><td rowspan=1 colspan=1>60.2</td></tr><tr><td rowspan=1 colspan=3>EvtCon</td><td rowspan=1 colspan=1>62.1</td><td rowspan=1 colspan=1>78.2</td><td rowspan=1 colspan=1>69.0</td><td rowspan=1 colspan=1>52.3</td><td rowspan=1 colspan=1>68.9</td><td rowspan=1 colspan=1>59.1</td><td rowspan=1 colspan=1>55.3</td><td rowspan=1 colspan=1>71.8</td><td rowspan=1 colspan=1>62.1</td><td rowspan=1 colspan=2>55.8 65.6</td><td rowspan=1 colspan=1>59.8</td></tr><tr><td rowspan=1 colspan=3>ICCL</td><td rowspan=1 colspan=1>67.5</td><td rowspan=1 colspan=1>73.7</td><td rowspan=1 colspan=1>70.4</td><td rowspan=1 colspan=1>60.3</td><td rowspan=1 colspan=1>62.7</td><td rowspan=1 colspan=1>61.3</td><td rowspan=1 colspan=1>62.6</td><td rowspan=1 colspan=1>66.1</td><td rowspan=1 colspan=1>64.2</td><td rowspan=1 colspan=2>63.7  68.8</td><td rowspan=1 colspan=1>65.4</td></tr></table>

Table 2: Results of ablation study on the ESC and CTB corpus.

completely align.

## 5.3 Numbers of demonstrations

To further investigate the impact of demonstrations, we conducted an experiment that compared the performance of In-context and ICCL with varying numbers of causal and non-causal demonstrations. The results are showcased in Fig. 3.

With more demonstrations, F1-score of both models initially exhibited improved performance, further validating the effectiveness of using demonstrations as explicit guidance. However, as the input length becomes too long, performance of Incontext declines, while the performance of ICCL continues to improve. This can be attributed to the effectiveness of contrastive module used in ICCL, which aids the PLM in better focusing on event pairs, even with longer input. Additionally, the causal/non-causal ratio of 2/1 performs better compared to that of 1/2. That’s because the dataset contains a limited number of causal samples. Increasing the number of causal demonstrations helps the model better learn the features of causal examples, mitigating the data imbalance issue.

We can also observe that performance metrics of In-context model, particularly precision, exhibit minimal changes when the number of demonstrations varies. While as for our ICCL model, the precision and recall vary based on the ratio of causal and non-causal demonstrations. More non-causal demonstrations results in higher recall, while the opposite scenario leads to higher precision. These findings emphasize that the critical role of the contrastive module in enhancing analogy and enabling the PLM to effectively utilize positive and negative demonstrations.

## 5.4 Few shot

Some researchers have reported the robustness of prompt paradigm in using fewer training data (Wang et al., 2021; Ding et al., 2021). Since our ICCL also employs a prompt-based method to predict the label, we examine its performance in low-resource scenarios and replicate the performance of ERGO as a benchmark for comparison. Fig. 4 shows the performance comparison between ERGO and our ICCL on ESC corpus.

![](images/53bf2fc354a7511a1110814bca2407abb7158eb712f85abd2d0d3900bc559870.jpg)

(a) F1 score  
![](images/09762ba13e8f86949127eaaf83ed37df217376d8d4f8934bce2813b43352b2fa.jpg)  
(b) Recall

![](images/e77fead7c35f0a3dc8280c1b966497de76cc3975ef6e97d7893e74275e74f424.jpg)  
(c) Precision  
Figure 3: Comparision of ICCL and In-context model when using differenr numbers of causal and non-causal demonstrations on ESC corpus.

As expected, the performance of ICCL gradually decreases as the amount of training data decreases. However, the decrease in performance is relatively slow, with an F1 score decrease of about 10% when training data is reduced by 80%, whereas the performance of ERGO declined by nearly 25%. Notably, even with only 20% of the training data, ICCL (F1: 51.9%) outperformes ERGO (F1: 50.9%) and many other competitors with full training data. These results confirm the effectiveness of ICCL even with fewer training data.

We also showcase the intra-sentence causality identification performance among different PLMs and several zero-shot models in the Table 3. We can not only find that our fine-tuned generative model, T5 (Our implementation), perform significantly worse than autoencoder models like BERTbase (Gao et al., 2023) and RoBERTa-base (Gao et al., 2023), which confirms the conclusion drawn by Gao et al. (2023) that generative models may not be well-suited for causal reasoning tasks like ECI. We can also observe that although the ChatGPT models, such as gpt-3.5-turbo and gpt-4, have more comprehensive pre-training and larger model scales, these zero-shot models exhibit a significant performance gap compared to fine-tuned models like T5-base and et al. This demonstrates the importance of fine-tune, indicating that it is challenging to address causal reasoning tasks like ECI in a zeroshot scenario. For more detailed analysis, please refer to Appendix A.

![](images/dcef2499f3055fa6e63c82fa041aab00382f481a214ef02cbea23d3801e65dec.jpg)

Figure 4: Results of few shot on ESC corpus. We replicated ERGO and get its few-shot results in the figure.
<table><tr><td rowspan="2">Model</td><td colspan="3">EventStoryLine</td><td colspan="3">Cause-TimeBank</td></tr><tr><td>P (%)</td><td>R (%)</td><td>F1 (%)</td><td>P (%)</td><td>R (%)</td><td>F1 (%)</td></tr><tr><td>BERT (Gao et al., 2023)</td><td>38.1</td><td>56.8</td><td>45.6</td><td>41.1</td><td>45.8</td><td>43.5</td></tr><tr><td>RoBERTa (Gao et al., 2023)</td><td>42.1</td><td>64.0</td><td>50.8</td><td>39.9</td><td>60.9</td><td>48.2</td></tr><tr><td>T5 (Our implementation)</td><td>36.2</td><td>49.2</td><td>40.7</td><td>7.7</td><td>52.1</td><td>12.1</td></tr><tr><td>gpt-3.5-turbo (Gao et al., 2023)</td><td>27.6</td><td>80.2</td><td>41.0</td><td>6.9</td><td>82.6</td><td>12.8</td></tr><tr><td>gpt-4 (Gao et al., 2023)</td><td>27.2</td><td>94.7</td><td>42.2</td><td>6.1</td><td>97.4</td><td>11.5</td></tr></table>

Table 3: Intra-sentence causality identification results of different PLMs and LLMs on the ESC and CTB corpus.

## 5.5 Embedding Visualization

In order to verify the impact of contrastive module with event pairs as input, we compare the learned event pairs’ embeddings $\left( h _ { e _ { 1 } } - h _ { e _ { 2 } } \right)$ of different models on ESC test dataset by t-distributed stochastic neighbor embedding (t-SNE) (Hinton and Roweis, 2002). In Fig. 5, we color-coded the points to represent True Nagetive (TN), False Positive (FP), False Nagetive (FN) and True Positive (TP) samples.

![](images/fd87f7242a188011b70852d129a8355f7db9a85c69f2c2c9a0f2a963558756be.jpg)  
(a) Prompt

![](images/2f868bb1312d46d3a7331c2cf47d2f387212287ad9ed141c42d6a6abe0d1ad91.jpg)  
(b) In-context

![](images/6615561f47ffa4bcbe7ff48e34156c23af423644ac4412c7b3535528cd9924d6.jpg)  
(c) EvtCon

![](images/bfdcacfecb365f6f20441b5f0d43d27a108b2ea79dca3b68ccc4d8aea0e605fa.jpg)  
(d) ICCL  
Figure 5: Visualization of the event pairs’ embedding encoded by different models on ESC corpus

We can ovserve that models incorporating the contrastive module with event pairs as input exhibit a clear phenomenon of event pairs representations clustering together based on labels in the embedding space, which demonstrates the effective of the contrastive module. Additionally, representations of samples predicted to have the same label tended to cluster together, highlighting the crucial role of event pairs in identifying causality.

## 6 Concluding Remarks

In this paper, we propose an ICCL model and apply it on the ECI task. We leverage the causality knowledge of PLM by introducing explicit guidance through the inclusion of demonstrations, rather than relying on the design of complex prompts. Meanwhile, we employ contrastive learning with event pairs as input to enhance the PLM’s attention to event pairs and strengthen the analogy between query and demonstrations. Experiments on the ESC and CTB corpus have validated that our ICCL can significantly outperform the state-of-the-art algorithms.

In future, we will try to undertake experiments to apply our proposed framework to other NLP tasks in order to explore whether it can exhibit favorable adaptability when applied to different tasks.

## Limitation

Due to the input length limitations of the PLM, the number of demonstrations needs to be kept within a manageable range. However, our ICCL uses demonstrations as positive and negative samples in contrastive learning. This implies that there are limited positive and negative samples, which weakens the effectiveness of contrastive learning.

## Acknowledgements

This work is supported in part by National Natural Science Foundation of China (Grant No: 62172167). The computation is completed in the HPC Platform of Huazhong University of Science and Technology.

## Ethics Statement

This paper has no particular ethic consideration.

## References

Sourabh Balgi, Jose M Pena, and Adel Daoud. 2022. Personalized public policy analysis in social sciences using causal-graphical normalizing flows. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 36, pages 11810–11818.

Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150.

Jonathan Berant, Vivek Srikumar, Pei-Chun Chen, Abby Vander Linden, Brittany Harding, Brad Huang, Peter Clark, and Christopher D Manning. 2014. Modeling biological processes for reading comprehension. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 1499–1510.

Manvi Breja and Sanjay Kumar Jain. 2020. Causality for question answering. In COLINS, pages 884–893.

Pengfei Cao, Xinyu Zuo, Yubo Chen, Kang Liu, Jun Zhao, Yuguang Chen, and Weihua Peng. 2021. Knowledge-enriched event causality identification via latent structure induction networks. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4862–4872.

Tommaso Caselli and Piek Vossen. 2017. The event storyline corpus: A new benchmark for causal and temporal relation extraction. In Proceedings of the Events and Stories in the News Workshop, pages 77– 86.

Meiqi Chen, Yixin Cao, Kunquan Deng, Mukai Li, Kun Wang, Jing Shao, and Yan Zhang. 2022. Ergo: Event relational graph transformer for documentlevel event causality identification. arXiv preprint arXiv:2204.07434.

Qian Chen, Xiaodan Zhu, Zhenhua Ling, Si Wei, Hui Jiang, and Diana Inkpen. 2016. Enhanced lstm for natural language inference. arXiv preprint arXiv:1609.06038.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Ning Ding, Yulin Chen, Xu Han, Guangwei Xu, Pengjun Xie, Hai-Tao Zheng, Zhiyuan Liu, Juanzi Li, and Hong-Gee Kim. 2021. Prompt-learning for fine-grained entity typing. arXiv preprint arXiv:2108.10604.

Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. 2022. A survey for in-context learning. arXiv preprint arXiv:2301.00234.

Chuang Fan, Daoxing Liu, Libo Qin, Yue Zhang, and Ruifeng Xu. 2022. Towards event-level causal relation identification. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1828– 1833.

Jinglong Gao, Xiao Ding, Bing Qin, and Ting Liu. 2023. Is chatgpt a good causal reasoner? a comprehensive evaluation. arXiv preprint arXiv:2305.07375.

Lei Gao, Prafulla Kumar Choubey, and Ruihong Huang. 2019. Modeling document-level causal structures for event causal relation identification. In Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 1808–1817.

Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. 2020. Deberta: Decoding-enhanced bert with disentangled attention. arXiv preprint arXiv:2006.03654.

Geoffrey E Hinton and Sam Roweis. 2002. Stochastic neighbor embedding. Advances in neural information processing systems, 15.

Zhilei Hu, Zixuan Li, Xiaolong Jin, Long Bai, Saiping Guan, Jiafeng Guo, and Xueqi Cheng. 2023. Semantic structure enhanced event causality identification. arXiv preprint arXiv:2305.12792.

Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. 2020. Supervised contrastive learning. Advances in neural information processing systems, 33:18661–18673.

Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. 2021a. What makes good in-context examples for gpt-3? arXiv preprint arXiv:2101.06804.

Jian Liu, Yubo Chen, and Jun Zhao. 2021b. Knowledge enhanced event causality identification with mention masking generalizations. In Proceedings of the Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence, pages 3608–3614.

Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. 2023. Pretrain, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Computing Surveys, 55(9):1–35.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.

Paramita Mirza and Sara Tonelli. 2014. An analysis of causality between events and its relation to temporal information. In Proceedings of COLING 2014, the 25th International Conference on Computational Linguistics: Technical Papers, pages 2097–2106.

Jeffrey Pennington, Richard Socher, and Christopher D Manning. 2014. Glove: Global vectors for word representation. In Proceedings ofthe 2014 conference on empirical methods in natural language processing (EMNLP), pages 1532–1543.

Minh Tran Phu and Thien Huu Nguyen. 2021. Graph convolutional networks for event causality identification with rich document-level structures. In Proceedings ofthe 2021 conference ofthe North American chapter of the association for computational linguistics: Human language technologies, pages 3480–3490.

Peter G Preethi, Vilma Uma, et al. 2015. Temporal sentiment analysis and causal rules extraction from tweets for event prediction. Procedia computer science, 48:84–89.

Ruili Pu, Yang Li, Suge Wang, Deyu Li, Jianxing Zheng, and Jian Liao. 2023. Enhancing event causality identification with event causal label and event pair interaction graph. In Findings of the Association for Computational Linguistics: ACL 2023, pages 10314– 10322.

Kira Radinsky, Sagie Davidovich, and Shaul Markovitch. 2012. Learning causality for news events prediction. In Proceedings of the 21st international conference on World Wide Web, pages 909–918.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal ofMachine Learning Research, 21(1):5485–5551.

Shirong Shen, Heng Zhou, Tongtong Wu, and Guilin Qi. 2022. Event causality identification via derivative prompt joint learning. In Proceedings of the 29th International Conference on Computational Linguistics, pages 2288–2299.

Robyn Speer, Joshua Chin, and Catherine Havasi. 2017. Conceptnet 5.5: An open multilingual graph of general knowledge. In Proceedings ofthe AAAI conference on artificial intelligence, volume 31.

Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xin Tian, Danxiang Zhu, Hao Tian, and Hua Wu. 2019. Ernie: Enhanced representation through knowledge integration. arXiv preprint arXiv:1904.09223.

Chengyu Wang, Jianing Wang, Minghui Qiu, Jun Huang, and Ming Gao. 2021. Transprompt: Towards an automatic transferable prompting framework for few-shot text classification. In Proceedings of the 2021 conference on empirical methods in natural language processing, pages 2792–2802.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. 2020. Transformers: State-of-the-art natural language processing. In Proceedings ofthe 2020 conference on empirical methods in natural language processing: system demonstrations, pages 38–45.

Wei Xiang, Zhenglin Wang, Lu Dai, and Bang Wang. 2022. Connprompt: Connective-cloze prompt learning for implicit discourse relation recognition. In Proceedings ofthe 29th International Conference on Computational Linguistics, pages 902–911.

Kun Zhao, Donghong Ji, Fazhi He, Yijiang Liu, and Yafeng Ren. 2021. Document-level event causality identification via graph inference mechanism. Information Sciences, 561:115–129.

Xinyu Zuo, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao, Weihua Peng, and Yuguang Chen. 2021a. Improving event causality identification via selfsupervised representation learning on external causal statement. arXiv preprint arXiv:2106.01654.

Xinyu Zuo, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao, Weihua Peng, and Yuguang Chen. 2021b. Learnda: Learnable knowledge-guided data augmentation for event causality identification. arXiv preprint arXiv:2106.01649.

## A Study of PLMs

The ICCL model we proposed is a PLM-sensitive model. In order to investigate the performance of our model using different PLMs and select the most suitable one, we conducted PLM ablation experiment to test performance of our model with differenr PLMs. Furthermore, we also cited performance of some baseline methods based on PLMs finetuned on full training datasets from the work of Gao et al. (2023) to evaluate various PLMs and summarized the results in Table 4. The introductions of main PLMs we considered are as follows:

BERT (Devlin et al., 2018): The most representive PLM proposed by Google<sup>4</sup>, which is pretrained using a cloze task and a next sentence prediction task.

RoBERTa (Liu et al., 2019): A BERT enhanced PLM proposed by Facebook<sup>5</sup>, which removes the next sentence prediction objective and is pre-trained on a much larger dataset with some modified key hyper-parameters.

ERNIE (Sun et al., 2019): A knowledge enhaced PLM proposed by Baidu<sup>6</sup>, which uses some knowledgeable masking strategies in pre-training.

DeBERTa (He et al., 2020): The latest masked PLM proposed by Microsoft<sup>7</sup>, which improves BERT and RoBERTa models using a disentangled attention mechanism and an enhanced mask decoder.

T5 (Raffel et al., 2020): A generative language model proposed by Google<sup>8</sup> in 2020, which is pretrained on large-scale unsupervised datasets using an autoregressive approach and fine-tuned on taskspecific annotated data. It has achieved state-ofthe-art performance on multiple NLP tasks such as text generation, summarization, and translation.

As shown in Table 4, according to the research by Gao et al. (2023), it can be observed that,our fine-tuned generative model, T5-base, performs significantly worse than autoencoder models like BERT-base (Gao et al., 2023) and RoBERTa-base (Gao et al., 2023). Moreover, the performance of In-context-T5 is also far inferior to the model Incontext-RoBERTa. This confirms the conclusion drawn by Gao et al. (2023) that generative models may not be well-suited for causal reasoning tasks like ECI. Additionally, although the ChatGPT models, such as gpt-3.5-turbo) and gpt-4, have more comprehensive pre-training and larger model scales, these zero-shot models exhibit a significant performance gap compared to fine-tuned models like T5-base and et al. This demonstrates the importance of fine-tune, indicating that it is challenging to address causal reasoning tasks like ECI in a zeroshot scenario.

Besides, we can observe that our ICCL with all four PLMs has achieved better performance than most of competitors on both ESC and CTB corpus. Even our ICCL-BERT outperformed many competitors with advanced PLMs, such as ERGO based on Longformer(Beltagy et al., 2020). This further demonstrates the effectiveness of our proposed method. Compared to approaches involving complex prompts or joint training across multiple tasks, our approach of utilizing simple explicit guidance and leveraging it for contextual contrastive learning better harnesses the semantic knowledge embedded in PLMs and guides their understanding of causal relationships.

We can also observe that using different PLMs do result in some performance variations. This is not unexpected. It can be attributed to that while all the four PLMs employ a kind of Transformerbased model in pre-training on large-scale corpus, their training strategies or training corpus are not entirely identical. Compared to ICCL-BERT, our ICCL model using ERNIE, DeBERTa, or RoBERTa achieved better performance. This is attributed to the fact that these three PLMs have made some optimizations based on BERT. For example, ERNIE utilizes a strategy of continuous learning in the pre-training stage. Finally, ICCL-RoBERTa achieved the best performance, which removes the next sentence prediction objective and is pre-trained on a much larger dataset with some modified key hyper-parameters. Therefore, we implement the remaining ablation experiments with RoBERTa.

## B Competitors

Table 4 also presents results of more competitors. The introductions of these competitors are as follows:

ILP (Gao et al., 2019) employs integer linear programming to detect causal relationships by incorporating causal constraints at document level.

<table><tr><td rowspan="3">Model</td><td colspan="8">EventStoryLine</td><td colspan="3">Cause-TimeBank</td></tr><tr><td colspan="3">Intra</td><td colspan="3">Cross</td><td colspan="2">Intra and Cross</td><td colspan="3">Intra</td></tr><tr><td>p (%)</td><td>r (%)</td><td>f1 (%)</td><td>p (%)</td><td>r (%)</td><td>f1 (%)</td><td>p (%) r (%)</td><td>f1 (%)</td><td>p (%)</td><td>r (%)</td><td>f1 (%)</td></tr><tr><td>T5</td><td>36.2</td><td>49.2</td><td>40.7</td><td></td><td>1</td><td>一</td><td>一</td><td></td><td>7.7</td><td>52.1</td><td>12.1</td></tr><tr><td>BERT †</td><td>38.1</td><td>56.8</td><td>45.6</td><td></td><td></td><td></td><td></td><td></td><td>41.4</td><td>45.8</td><td>43.5</td></tr><tr><td>RoBERTa †</td><td>42.1</td><td>64.0</td><td>50.8</td><td></td><td></td><td></td><td></td><td></td><td>39.9</td><td>60.9</td><td>48.2</td></tr><tr><td>text-davinci-002 †</td><td>23.2</td><td>80.0</td><td>36.0</td><td></td><td></td><td></td><td></td><td></td><td>5.0</td><td>75.2</td><td>9.3</td></tr><tr><td>text-davinci-003 †</td><td>33.2</td><td>74.4</td><td>45.9</td><td></td><td></td><td></td><td></td><td></td><td>8.5</td><td>64.4</td><td>15.0</td></tr><tr><td>gpt-3.5-turbo †</td><td>27.6</td><td>80.2</td><td>41.0</td><td></td><td></td><td></td><td></td><td></td><td>6.9</td><td>82.6</td><td>12.8</td></tr><tr><td>gpt-4 †</td><td>27.2</td><td>94.7</td><td>42.2</td><td></td><td></td><td></td><td></td><td></td><td>6.1</td><td>97.4</td><td>11.5</td></tr><tr><td>In-context-T5</td><td>63.3</td><td>62.6</td><td>62.7</td><td>53.7</td><td>46.6</td><td>49.3</td><td>57.0 51.5</td><td>53.7</td><td>9.2</td><td>50.4</td><td>14.8</td></tr><tr><td>In-context-RoBERTa</td><td>66.0</td><td>72.4</td><td>68.9</td><td>57.7</td><td>60.9</td><td>59.1</td><td>60.4 64.5</td><td>62.2</td><td>60.3</td><td>58.0</td><td>58.7</td></tr><tr><td>ILP (Gao et al., 2019)</td><td>38.8</td><td>52.4</td><td>44.6</td><td>35.1</td><td>48.2</td><td>40.6</td><td>36.2 49.5</td><td>41.9</td><td></td><td>=</td><td>=</td></tr><tr><td>KnowMMR (Liu et al., 2021b)</td><td>41.9</td><td>62.5</td><td>50.1</td><td></td><td></td><td></td><td></td><td></td><td>36.6</td><td>55.6</td><td>44.1</td></tr><tr><td>RichGCN (Phu and Nguyen, 2021)</td><td>49.2</td><td>63.0</td><td>55.2</td><td>39.2</td><td>45.7</td><td>42.2</td><td>42.6 51.3</td><td>46.6</td><td>39.7</td><td>56.5</td><td>46.7</td></tr><tr><td>CauSeRL (Zuo et al., 2021a)</td><td>41.9</td><td>69.0</td><td>52.1</td><td></td><td></td><td></td><td></td><td></td><td>43.6</td><td>68.1</td><td>53.2</td></tr><tr><td>LSIN (Cao et al., 2021)</td><td>47.9</td><td>58.1</td><td>52.5</td><td></td><td></td><td></td><td></td><td></td><td>51.5</td><td>56.2</td><td>53.7</td></tr><tr><td>LearnDA (Zuo et al., 2021b)</td><td>42.2</td><td>69.8</td><td>52.6</td><td></td><td></td><td></td><td></td><td></td><td>41.9</td><td>68.0</td><td>51.9</td></tr><tr><td>GESI (Fan et al., 2022)</td><td>=</td><td>=</td><td>50.3</td><td></td><td></td><td>49.3</td><td></td><td>49.4</td><td></td><td>=</td><td></td></tr><tr><td>ERGO (Chen et al., 2022)</td><td>57.5</td><td>72.0</td><td>63.9</td><td>51.6</td><td>43.3</td><td>47.1</td><td>48.6 53.4</td><td>50.9</td><td>62.1</td><td>61.3</td><td>61.7</td></tr><tr><td>DPJL (Shen et al., 2022)</td><td>65.3</td><td>70.8</td><td>67.9</td><td></td><td></td><td></td><td></td><td></td><td>63.6</td><td>66.7</td><td>64.6</td></tr><tr><td>SemSln (Hu et al., 2023)</td><td>64.2</td><td>65.7</td><td>64.9</td><td></td><td>=</td><td>–</td><td>= =</td><td>=</td><td>52.3</td><td>65.8</td><td>58.3</td></tr><tr><td>ICCL-BERT</td><td>64.9</td><td>69.6</td><td>67.1</td><td>56.3</td><td>58.4</td><td>57.2</td><td>59.0 61.9</td><td>60.4</td><td>60.5</td><td>58.4</td><td>59.1</td></tr><tr><td>ICCL-ERNIE</td><td>66.8</td><td>68.5</td><td>67.5</td><td>63.7</td><td>56.2</td><td>59.5</td><td>64.8 60.0</td><td>62.1</td><td>64.8</td><td>66.0</td><td>64.7</td></tr><tr><td>ICCL-DeBERTa</td><td>67.6</td><td>73.7</td><td>70.4</td><td>61.8</td><td>58.4</td><td>59.9</td><td>61.7 63.2</td><td>63.3</td><td>66.7</td><td>64.4</td><td>64.9</td></tr><tr><td>ICCL-RoBERTa</td><td>67.5</td><td>73.7</td><td>70.4</td><td>60.3</td><td>62.7</td><td>61.3</td><td>62.6</td><td>66.1 64.2</td><td>63.7</td><td>68.8</td><td>65.4</td></tr></table>

Table 4: Comparison of overall results on the ESC and CTB corpus. Performance of models marked with "†" after the name are cited from the research of Gao et al. (2023). We name our models in the format of Model-PLM, for example, ICCL-BERT is the version of ICCL model based on BERT.

KnowMMR (Liu et al., 2021b) utilizes external knowledge to extract event causality patterns.

RichGCN (Phu and Nguyen, 2021) uses a graph convolutional network to learn contextenriched representations for event pairs based on document-level information.

CauSeRL (Zuo et al., 2021a) employs a contrastive approach to transfer externally learned causal statements.

LSIN (Cao et al., 2021) employs graph induction to acquire external structural and relational knowledge.

LearnDA (Zuo et al., 2021b) utilizes knowledge bases to interactively generate training data.

GESI (Fan et al., 2022) designs a graph convolutional network on an event co-reference graph to model causality.

ERGO (Chen et al., 2022) constructs a relational graph where event pairs serve as nodes, capturing causal transitivity through a transformer-like network.

DPJL (Shen et al., 2022) leverages two derivative prompt tasks to identify causality.

SemSln (Hu et al., 2023) uses a Graph Neural Network (GNN) to learn from event-centric structures for encoding events.

## C In-context input

To help readers gain a better understanding of the in-context input generated by our Prompt module, we provide a specific example in Fig. 6.

As depicted in Fig. 6, we randomly chose two causal demonstrations and two non-causal demonstrations from the training dataset for the query. Each segment in Fig. 6 represents either a prompted demonstration or a prompted query. The initial two segments, highlighted in green font, represents demonstrations labeled as < causal >. The following two segments, highlighted in orange font, represents demonstrations labeled as < none >. Lastly, the final segment, highlighted in purple font, represents the query to predict.

Besides, we have annotated some specific tokens we used with special colors. We utilized three PLM-special tokens: [CLS] to indicate the beginning of the input, [SEP] as a sentence separator, and [MASK] as a placeholder for the label to predict. Furthermore, we have also devised some additional special tokens: [start] and [end] are used to indicate the beginning and end of the cloze tem-

![](images/2e524c7cdf19988c31ec7b252e3bd686c8824624404a23a9efcb39c66f07392c.jpg)

Figure 6: Example of in-context input. The line breaks and the title of each part (ex. Causal Demonstrations) are only to make the input readable, and they are not included in the actual input.  
![](images/71e3c3ad02d8e217e2209314ece76bcfde5e3c791156d84e52dc8017f5d155fe.jpg)  
Figure 7: Comparision of ICCL model with different value of $\beta$ on the ESC corpus.  
plate respectively, [event1], [event1/], [event2], [event2/] are used to highlight the events in the query, while < causal > and < none > respectivaly represent the causal and uncausal labels for the demonstrations.

Additionally, although the contrastive module only works during the training phase, we select appropriate demonstrations for the query in both training and testing phases. Specifically, we randomly select M samples labeled as < causal > and N samples labeled as < none > from training dataset to be demonstrations. And on the contrastive learning process, positive demonstrations are those with the same label as the query, while negative demonstrations have different labels. Furthermore, during training phase, different demonstrations are retrieved for the same query in different epochs to introduce variability and enhance the model’s ability to handle diverse instances of the same query. However, during validation and testing state, demonstrations retrieved for the same query, as well as the permutation order, remain consistent across epochs which ensures fair evaluation.

## D Study of $\beta$

To further explore how to balance the importance of contrastive loss and prediction loss, we investigated the performance of the ICCL model with different values of the hyperparameter $\beta$ on the ESC corpus.

As shown in Fig. 7, we can observe that as $\beta$ increases from 0, the performance of the model initially improves and then starts to decline. The optimal performance on both intra-sentence causality and cross-sentence causality is achieved when $\beta = 0 . 5$ . This indicates that the introduction of contrastive learning loss does indeed help the model better focus on event pairs of the query and demonstrations, understand causalities, and achieve better performance. However, it is important to strike a balance between the contrastive learning loss and the prediction loss. Excessive emphasis on the former should be avoided as it may cause the model to overly prioritize modeling event pairs and overlook the semantic relevance of the context, which can ultimately lead to a decrease in the model’s performance.