# Towards Difficulty-Agnostic Efficient Transfer Learning for Vision-Language Models

Yongjin Yang∗ Jongwoo Ko∗ Se-Young Yun

KAIST AI

{dyyjkd, jongwoo.ko, yunseyoung}@kaist.ac.kr

## Abstract

Vision-language models (VLMs) like CLIP have demonstrated remarkable applicability across a variety of downstream tasks, including zero-shot image classification. Recently, the use of prompts or adapters for efficient transfer learning (ETL) has gained significant attention for effectively adapting to downstream tasks. However, previous studies have overlooked the challenge of varying transfer difficulty of downstream tasks. In this paper, we empirically analyze how each ETL method behaves with respect to transfer difficulty. Our observations indicate that utilizing vision prompts and text adapters is crucial for adaptability and generalizability in domains with high difficulty. Also, by applying an adaptive ensemble approach that integrates task-adapted VLMs with pretrained VLMs and strategically leverages more general knowledge in low-difficulty and less in high-difficulty domains, we consistently enhance performance across both types of domains. Based on these observations, we propose an adaptive ensemble method that combines visual prompts and text adapters with pre-trained VLMs, tailored by transfer difficulty, to achieve optimal performance for any target domain. Upon experimenting with extensive benchmarks, our method consistently outperforms all baselines, particularly on un seen tasks, demonstrating its effectiveness.

## 1 Introduction

Vision-language models (VLMs), such as CLIP (Radford et al., 2021) and ALIGN (Jia et al., 2021), have demonstrated remarkable applicability across various downstream tasks such as image classification. A distinctive feature of these VLMs for image classification is their ability to classify unseen classes that have not been encountered during pre-training through zero-shot inference, which is not possible to traditional vision models.

The primary challenge of VLMs for downstream tasks is to excel in classifying both seen and unseen class sets. In the context of VLM classification tasks, the ability to accurately classify seen class sets is termed adaptability, while the capability to extend this proficiency to unseen class sets is referred to as generalizability. To boost these abilities, recent research has introduced efficient transfer learning (ETL) methods to fine-tune VLMs. One strategy involves the use of soft prompt tuning (Zhou et al., 2022b,a; khattak et al., 2023; Khattak et al., 2023). Another research direction involves adapter-style tuning (Gao et al., 2023; Zhang et al., 2022; Zhu et al., 2023b) either by adjusting specific parameters or employing cache-based techniques. These approaches empower VLMs to swiftly adapt to new tasks using only a few samples (i.e. few-shot image classification task).

However, previous approaches do not consider a significant factor for adapting to downstream tasks: varying transfer difficulty (Yu et al., 2023). This refers to the challenge of adapting pre-trained VLMs according to the target domain. For instance, transferring pre-trained VLMs to specific finegrained domains, such as FGVC Aircraft, is more challenging than transferring to general coarsegrained domains. In a real-world scenario, it is hard to predict the specific target task and domain that will emerge. Therefore, without investigating how each type of ETL behaves in response to different levels of transfer difficulty and applying an adaptive method based on this investigation, the result for each target domain can be suboptimal. Some works manually train models differently for each dataset (Gao et al., 2023; Zhang et al., 2022), but this approach is not feasible in real-world scenarios as prior knowledge for the target task is not given.

To overcome these limitations and apply an adaptive method for tuning adapters and prompts for downstream tasks, we empirically investigate the characteristics of applying different tuning methods for each modality on multiple domains with varying transfer difficulty, revealing four key findings. Firstly, we find that visual prompt tuning (VPT) generalizes better to unseen classes compared to text prompt tuning (TPT) in cases of high-difficulty domains, as TPT tends to overfit on base classes for these domains. (▷ Obs. 1). This occurs because, in high-difficulty domains, the class separability of visual features from a visual encoder is low, causing TPT to overly adapt in classifying these challenging features (▷ Obs. 2). Moreover, text adapter (TA) can significantly boost the adaptability of VPT, resulting in high adaptability and generalizability, especially for highly difficult domains (▷ Obs. 3). However, fine-tuning with adapters could compromise generalizability in easier domains. Our last observation is that combining pre- and post-adapter features to leverage pre-trained VLMs knowledge can address this concern with a proper balance between them. For instance, using more pre-adapter features can maintain generalizability in easier domains. The ideal balance depends on the domain’s difficulty, highlighting the need to adjust the ensemble coefficient accordingly (▷ Obs. 4).

![](images/7c3665d5b9f3772713ccd05a1bd4b966a55ab781562fa9dd6bd11f4a5fc8ca3d.jpg)  
Figure 1: Overview of APEX compared to the conventional ETL methods. APEX exhibits two key differences: (a): Firstly, APEX integrates prompt tuning for the visual encoder and a linear adapter for the text encoder, each tailored to the specific properties of their respective modalities, which performs better on high-difficulty domains. (b): Secondly, APEX integrates an adaptive coefficient within the text encoder to strategically balance pre-adapter and post-adapter features to properly combine task-specific knowledge and general VLMs knowledge based on transfer difficulty. A detailed explanation, including notations and the algorithm, can be found in Section 4 and Appendix B.

Based on our observations, we present a APEX (text Adapter, visual Prompt, and adaptive Ensemble for cross(X)-modality) that utilizes an adaptive ensemble with VPT and TA. Specifically, we use the combination of VPT and TA, which have shown high generalizability and adaptability for high-difficulty domains, as shown in Obs. 1-3 (Fig. 1(a)). Also, motivated by Obs. 4, we employ an adaptive ensemble approach that determines the optimal ensemble coefficient for each domain by using the distances to learned classes in pre-trained VLMs to estimate transfer difficulty (Fig. 1(b)). This adaptive ensemble controls the level of adaptation, by primarily utilizing task-specific knowledge with adapted VLMs for high-difficulty domains but leveraging general knowledge for low-difficulty domains, as pre-trained VLMs already possess sufficient ability and prevent an overfitting from excessive adaptation. With this, our method acts as a difficulty-agnostic solution, enabling the model to effectively adapt to all target domains regardless of transfer difficulty. In summary, our main contributions are:

We investigate prompt tuning and adapter tuning methods to understand their effectiveness across domains with varying transfer difficulties. Our findings reveal that the efficacy of each method with each modality varies across different of transfer difficulty, with notable performance of VPT and TA for high-difficulty domains.

We propose APEX, which utilizes VPT and TA for tuning and employ an adaptive ensemble approach to optimally leverage the general knowledge of VLMs for each domain. The ensemble’s coefficient is adaptively determined by the distances to learned classes, serving as an estimate of transfer difficulty.

We show that APEX achieves state-of-the-art performance across various downstream tasks, with particularly notable improvements in unseen tasks during adaptation.

## 2 Backgrounds

Here, we provide a brief overview of the background related to our method. For a detailed explanation with more related works is in Appendix E.

Zero-shot CLIP. CLIP (Radford et al., 2021) is designed for creating visual features based on natural language guidance. The CLIP model can perform zero-shot inference, classifying an image into one of C possible classes without additional training. This is achieved by calculating the cosine similarity between an visual feature z, derived from the visual encoder, and the text features of each class $\{ \mathbf { t } _ { i } \} _ { i = 1 } ^ { C }$ , which are obtained from the text encoder.

For processing the image, let us define the visual encoder as , which comprises $L _ { \mathcal { V } }$ layers, denoted as $\{ \mathcal { V } _ { i } \} _ { i = 1 } ^ { L \nu }$ . The encoder takes patch embeddings $\mathbf { E } _ { 0 } \in \bar { \mathbb { R } ^ { M \times d _ { v } } }$ as input, which are obtained by dividing the image I into M fixed-size patches. Patch embeddings $\mathbf { E } _ { i }$ is then fed into the $( i + 1 ) ^ { \mathrm { t h } }$ transformer block $( \nu _ { i + 1 } )$ along with a learnable class ([CLS]) tokens $\mathbf { c } _ { i }$ . This process is sequentially carried out through all $L _ { \mathcal { V } }$ transformer blocks, formulated as follows:

$$
[ \mathbf { c } _ { i } , \mathbf { E } _ { i } ] = \mathcal { V } _ { i } \left( [ \mathbf { c } _ { i - 1 } , \mathbf { E } _ { i - 1 } ] \right) i = 1 , \dots , L \nu ,\tag{1}
$$

$$
\mathbf { z } = \mathtt { I m a g e P r o j } ( \mathbf { c } _ { L \nu } ) ,\tag{2}
$$

Here, $[ \cdot , \cdot ]$ denotes the concatenation operation. We can obtain the text features in a similar way with word embeddings $\mathbf { W } _ { 0 } = [ \mathbf { w } _ { 0 } ^ { 1 } , \dots , \mathbf { w } _ { 0 } ^ { N } ] \in \bar { \mathbb { R } } ^ { N \times d _ { l } }$ and text encoder which is consist of $L _ { T }$ layers $\{ \mathcal { T } _ { i } \} _ { i = 1 } ^ { L \tau }$ , as follows:

$$
[ \mathbf { W } _ { i } ] = { \mathcal { T } } _ { i } ( \mathbf { W } _ { i - 1 } ) \ i = 1 , \dots , L \tau\tag{3}
$$

$$
\mathbf { t } _ { i } = \tt T e x t P r o j ( \mathbf { w } _ { L \tau } ^ { N } )\tag{4}
$$

The predicted probability for class i is as:

$$
\operatorname* { P r } ( y = i | \mathbf { z } , \mathbf { t } ) = \frac { \exp { ( \sin ( \mathbf { z } , \mathbf { t } _ { i } ) / \tau ) } } { \sum _ { j = 1 } ^ { C } \exp { ( \sin ( \mathbf { z } , \mathbf { t } _ { j } ) / \tau ) } } ,\tag{5}
$$

where sim( , ) indicates cosine similarity and $\tau$ is the learned temperature of CLIP. We can also interpret the text features as a classifier (Gao et al., 2023; Zhang et al., 2022), where $\mathbf { t } _ { i }$ is the classifier weight for class i.

Prompt Tuning for CLIP. To enable prompt tuning (Zhou et al., 2022a; khattak et al., 2023; Zhu et al., 2023a; Khattak et al., 2023), we replace the Eq. (1) and Eq. (3) by newly introducing b and $b _ { \mathcal { T } }$ learnable tokens $\{ \bar { P } _ { i } ^ { k } \in \bar { \mathbb { R } } ^ { d _ { v } } \} _ { k = 1 } ^ { b _ { v } }$ <sub>=1</sub> and $\{ P _ { i } ^ { k } \in \mathbb { R } ^ { d _ { l } } \} _ { k = 1 } ^ { b _ { T } }$ for $i ^ { \mathrm { t h } }$ layer, and their concatenation $\hat { { \bf P } } _ { i }$ and $\mathbf { P } _ { i }$ . We can introduce the visual prompt for the first $J _ { \nu }$ layers of the visual encoder, then we can compute as follows:

$$
\begin{array} { r l } & { [ \mathbf { c } _ { i } , \mathbf { E } _ { i } , \mathbf { \Theta } _ { - } ] = \mathcal { V } _ { i } ( [ \mathbf { c } _ { i - 1 } , \mathbf { E } _ { i - 1 } , \hat { \mathbf { P } } _ { i - 1 } ] ) , } \\ & { [ \mathbf { c } _ { j } , \mathbf { E } _ { j } , \hat { \mathbf { P } } _ { j } ] = \mathcal { V } _ { j } ( [ \mathbf { c } _ { j - 1 } , \mathbf { E } _ { j - 1 } , \hat { \mathbf { P } } _ { j - 1 } ] ) , } \end{array}\tag{6}
$$

for $i = 1 , \ldots , J _ { \mathcal { V } }$ and $j = J _ { \mathcal { V } } + 1 , \ldots , L _ { \mathcal { V } }$ . Also, we can replace Eq. (3) to belows by introducing text prompt for the fisrt $J _ { T }$ layers of text encoder:

$$
\begin{array} { r l } & { \left[ \mathbf { \Theta } _ { - } , \mathbf { W } _ { i } \right] = { \mathcal { T } } _ { i } ( \left[ \mathbf { P } _ { i - 1 } , \mathbf { W } _ { i - 1 } \right] ) i = 1 , \dots , J _ { T } , \qquad ( 7 ) } \\ & { \left[ \mathbf { P } _ { j } , \mathbf { W } _ { j } \right] = { \mathcal { T } } _ { j } ( \left[ \mathbf { P } _ { j - 1 } , \mathbf { W } _ { j - 1 } \right] ) j = J _ { T } + 1 , \dots , L _ { T } . } \end{array}
$$

Here, we train the visual and text prompt for the first $J _ { \nu }$ and $J _ { T }$ layers of corresponding encoders. Adapter-style Tuning for CLIP. To enable adapter-style tuning, we replace Eq. (2) and Eq. (4) by introducing ImgAdapt and TxtAdapt which are shallow stacking networks upon the frozen CLIP model (Gao et al., 2023; Zhang et al., 2022; Zhu et al., 2023b).

$$
\tilde { \mathbf { z } } = \mathbf { I m g P r o j } ( \mathbf { c } _ { L \nu } ) , ~ \mathbf { z } = \mathbf { I m g A d a p t } ( \tilde { \mathbf { z } }\tag{8}
$$

$$
\tilde { \mathbf { t } } = \mathbf { \bar { r } } \mathbf { x } \mathbf { t } \mathbf { p } \mathbf { r } \mathbf { o } \mathbf { j } ( \mathbf { w } _ { L _ { \mathcal { T } } } ^ { N } ) , \mathbf { t } = \mathbf { \bar { r } } \mathbf { x } \mathbf { t } \mathbf { a } \mathbf { a } \mathbf { p } \mathbf { t } ( \tilde { \mathbf { t } } )\tag{9}
$$

## 3 Motivating Observations

Here, we analyze the behavior of visual and text encoders depending on different tuning methods and transfer difficulty of target domains within the framework of ETL. To accomplish this, we begin by categorizing domains based on their relative transfer difficulty (RTD), which is a metric first defined by Yu et al. (2023).

Definition 1 (Relative Transfer Difficulty (Yu et al., 2023)). Let $f ( \cdot )$ and $g ( \cdot )$ be random classifiers where the precision ofeach equals 1/C, and zeroshot CLIP, respectively. Also, Prec<sub>f</sub> and $P r e c _ { g }$ denote the precision of classifiers f and g. Then, RTD isformulated asfollows:

$$
R T D = \frac { P r e c _ { f } } { P r e c _ { g } } = \frac { 1 / C } { P r e c _ { g } } = \frac { 1 } { C \cdot P r e c _ { g } }
$$

Under this metric, we identify EuroSAT, DTD, and FGVC Aircraft as the three most challenging domains, while ImageNet, SUN397, and Stanford Cars are recognized as the three easiest domains. We will primarily focus on these six domains to clearly demonstrate the impact of RTD on VLMs’ behavior. To assess adaptability and generalizability, we train the CLIP-B/16 utilizing each prompt tuning approach on tasks requiring generalization from base to novel categories. Here, “base category" refers to a subset of classes within the domain learned through few-shot methods, and “novel category" is those not included in the training. Each dataset is split into these categories; the model is trained on base classes with 16 shots and tested on both. Therefore, performance on the “base category" is related to adaptability, and performance in the “novel category" is related to generalizability. More detailed values are present in Appendix D.

![](images/ff0ee2fac72a7455ee7dc7ad84ffd6ab81211900759379556e9f87e38c899da4.jpg)  
Figure 2: Comparison of accuracy differences (%) between base and novel categories across three prompt tuning options (TPT, VPT+TPT, VPT) with varying numbers of shots.

![](images/7cab3287b236c590f20e41b1ca94a9ba4d8079dc72cb020f9ed6116393789d2f.jpg)

Figure 3: Comparison of the accuracy (%) of base and novel categories using TPT, VPT, and their combination (VPT+TPT) on three transfer learning datasets over various training epochs.  
![](images/1b6118bd7e417a7cdfb71e09eb4fde6bdbf68c98674965f46c43e658cddce6fa.jpg)  
Figure 4: t-SNE (Van der Maaten and Hinton, 2008) plots of visual features for novel category with their corresponding labels (left), zero-shot CLIP prediction (middle), and prediction with TPT (right). 50 samples are randomly selected from each class in EuroSAT and SUN397, using all 5 classes in EuroSAT and 5 randomly chosen classes from SUN397. Dotted lines within the t-SNE plot represent the decision boundaries corresponding to each class, indicated by the same color.

Observation 1. VPT offers better generalizability than TPT. While TPT has greater adaptability to seen classes in low-difficulty domains, it is not effectivefor high-difficulty domains and shows overfitting to the base classes.

We commence with an analysis of the separate behavior of visual and text prompts during the tuning process. Fig. 2 illustrates the performance discrepancy between the two categories for each method. Across all domains, VPT consistently shows the smallest performance gap for every shot number, indicating reduced overfitting to base classes. This observation is especially prominent in domains with high RTD though the trend is not as pronounced in domains with low RTD. We also observe that combining VPT and TPT does not consistently mitigate the overfitting of TPT, as evidenced by the larger performance gap in FGVC Aircraft and EuroSAT compared to TPT alone.

Fig. 3 displays the comparative performance of base and novel categories over different epochs. While all prompt tuning methods show an improvement in base category performance at the expense of generalization, VPT consistently exhibits a lesser decline in novel category performance. Notably, for challenging domains like FGVC Aircraft and EuroSAT, VPT exceeds the novel performance of TPT and their combination regardless of epoch.

Observation 2. Low class separability of visual features is the primary reason for the overfitting of TPT on high RTD.

Class separability is a critical factor in determining the transferability of a source model to a target domain (Pándy et al., 2022). To determine the class separability of visual features, we use the ratio of intra- to inter-class cosine similarities (Oh et al., 2021; Zhu et al., 2023b). Fig. 5 demonstrates that the ratio is higher in domains with lower RTD, which are considered easier, and lower in more challenging datasets with higher RTD. These findings suggest that the class separability highly correlates with transfer difficulty, strongly influencing the overfitting risk of TPT on high RTD domains.

To see how class separability affects TPT, we further explore the visual features and predictions of zero-shot CLIP and TPT. As shown in Fig. 4, EuroSAT, which exhibits a high RTD, shows lower class separability compared to SUN397 that has a lower RTD. Furthermore, in EuroSAT, when TPT attempts to classify visual features with low class separability, its performance for novel classes is lower than zero-shot CLIP. This is because TPT tries to fit the decision boundary, represented as dotted lines, to features that are challenging to classify by solely adjusting classifier weights with multiple stacks of learnable prompts. This underscore the significance of separable visual features, a factor closely linked to VPT. Consequently, this leads to significant overfitting, where the decision boundary of one class overlaps with others. Conversely, with visual features that exhibit high class separability, TPT’s predictions are more accurate than those of zero-shot CLIP as it can easily determine the better decision boundary.These results underscore the significance of separable visual features, a factor closely linked to VPT.

![](images/6e04eef87e88f5a63a605ee58b4f2f3b99fd7b361ac36a2bde0335e09ca3ec96.jpg)  
Figure 5: Comparison of intra- and inter-class ratios to show class separability across different datasets with their RTD, arranged from low to high RTD.

Observation 3. TA effectively enhances adaptability with a low risk of overfitting when employed with VPT, especially on higher RTD datasets.

Fig. 6 shows that while TA and VPT each exhibit less adaptability than TPT alone, together they outperform across all categories, signifying both high adaptability and generalizability. This advantageous combination is particularly significant for higher RTD, while the performance improvement in novel categories with lower RTD is marginal.

This synergy occurs because VPT enhances the class separability in visual features, allowing the linear transformation of classifier weights to suffice for adaptation, as depicted in Fig. 7. TA simply modifies the features of the pre-trained text encoder, preventing overconfidence in the decision boundary, especially for domains with high RTD and low class separability. In addition, we conduct experiments using a combination of TPT and a visual adapter (VA). However, this combination proves less effective than integrating VPT and TA, further emphasizing the importance of visual feature separability.

![](images/fa94763866ca515be9d9d41e40ce0a7bd0e78f43b55e1a4cf168cafff3fd73c7.jpg)

Figure 6: Comparison of the combined effectiveness of prompt tuning and adapter-style tuning. “Easy" refers to three domains with low RTD, and “Challenge" refers to three domains with high RTD.  
![](images/adf8924bb164c77d7ed229b8d663c31a60613a58cff25b1f5cb28b942e4c98a5.jpg)  
Figure 7: t-SNE plots of visual features of CLIP with VPT for a novel category with their corresponding labels (left) and prediction with TA (right). 50 samples are randomly selected from each class.

Observation 4. By modulating the influence of TA through an ensemble ofpre-adapter and postadapter features, each with a domain-specific coefficient, we can significantly improve generalization in low RTD domains while maintaining high performance in high RTD domains.

While combining VPT and TA has great synergy in high RTD domains, utilizing TA can result in the loss of some general knowledge from the original CLIP, which is crucial for domains with low RTD. This is evident in Tab. 1, as naïvely using VPT and TA together may lead to a degradation in performance on novel classes in domains with low RTD. This is because for low RTD, a lot of tasks within the domain need to lie in the region of general knowledge, as illustrated in Fig. 1(b). But the training of a TA creates a task-specific boundary which may not be optimal for other tasks within the same domain. In domains with high RTD, task-specific knowledge gained from adapters can also enhance performance on unseen tasks, as the general knowledge is often insufficient for these domains.

This degradation in domains with low RTD can be mitigated by diminishing the influence of TA.

Table 1: Comparison of accuracy (%) on novel classes between zero-shot CLIP, without an ensemble, an ensemble with fixed coefficient, and an ensemble with optimal coefficient. We determine the fixed coefficient as 0.4, based on average novel performance.
<table><tr><td>Dataset</td><td>SUN397</td><td>Stanford Cars</td><td>DTD</td><td>EuroSAT</td></tr><tr><td>ZS CLIP</td><td>75.35</td><td>74.89</td><td>59.90</td><td>64.05</td></tr><tr><td>VPT + TA</td><td>74.52 (-0.83)</td><td>68.40 (-6.49)</td><td>63.05 (+3.15)</td><td>77.73 (+13.68)</td></tr><tr><td>+ Fixed Ens (α = 0.4)</td><td>78.68 (+3.33)</td><td>74.22 (-0.67)</td><td>64.16 (+4.26)</td><td>75.87 (+11.82)</td></tr><tr><td>+ Opt. Ens</td><td>78.90 (+3.55)</td><td>75.19 (+0.30)</td><td>64.32 (+4.42)</td><td>77.73 (+13.68)</td></tr><tr><td>Opt. α</td><td>0.3</td><td>0.0</td><td>0.5</td><td>1.0</td></tr></table>

Inspired by the residual connection in adapter-style tuning methods (Zhang et al., 2022; Gao et al., 2023), we use an ensemble of pre-adapter and postadapter features for the text encoder. This ensemble, defined with coefficient α, can be expressed as:

$$
\mathbf { t } = \alpha \cdot \mathbf { T x } \mathbf { t } \mathbf { a d a p t } ( \tilde { \mathbf { t } } ) + ( 1 - \alpha ) \cdot \tilde { \mathbf { t } } .\tag{10}
$$

As Tab. 1 illustrates, the ensemble method improves performance in domains with low RTD. However, using pre-adapter features can yield suboptimal outcomes in more challenging domains. For instance, performance on EuroSAT drops from 77.73% to 75.87% when α is set as a fixed coefficient, as domains with high RTD demand more from TA. By optimally setting α for each domain, we consistently outperform zero-shot CLIP across all domains by effectively combining general and task-specific knowledge tailored to each domain’s needs. Observing this optimal coefficient, we note that that more challenging domains typically require a higher coefficient. These findings highlight the necessity of a method to calculate an adaptive coefficient of ensemble, which would modulate TA activation according to domain and its RTD.

## 4 Method

Based on our observations, we propose a new method, APEX, which is a difficulty-agnostic approach that utilizes an adaptive ensemble with tuning methods including VPT and TA.

## 4.1 Configuration Design & Training

Due to the need for a combination of VPT and TA to achieve adaptability and generalizability in highly difficult domains, we configure the trainable parameters to include multiple stacks of visual prompts, and a linear text adaptation layer following the pre-trained text encoder. While existing adapter-style methods (Zhang et al., 2022; Zhu et al., 2023b; Gao et al., 2023) rely on manually optimized text prompts for different datasets, we use learnable text prompts just for the input because manually creating prompt templates for each domain in the real world is challenging. The learnable text prompts are unnecessary if manual prompts are already well-formed, which is further explained in Section 5.

![](images/bdef3e3f1ca3a5b8809f1ec0b35194cafe08b14b44c5d0f542023fb5bee9ce8c.jpg)  
Figure 8: The relationship between class distance and optimal α for each domain used in Eq. (10) and Table 1.

We extract the visual feature z using Eq. (6) and Eq. (2) and the text feature t using Eq. (7) with $\begin{array} { r l r } { J _ { \mathcal { T } } } & { { } = } & { 1 } \end{array}$ and Eq. (9). We apply linear adapter parameterized as matrix A and bias b for TextAdapter in Eq. (9) rather than using bottleneck structure (Zhang et al., 2022; Gao et al., 2023) based on our results in Fig. 11. Our adapter can be formulated as follows:

$$
\mathbf { t } = \mathbf { T x t a d a p t } ( \tilde { \mathbf { t } } ) : = \mathbf { A } ^ { \top } \tilde { \mathbf { t } } + \mathbf { b }\tag{11}
$$

During the training procedure, our objective is to maximize the predicted probability $\mathrm { P r } ( y = y _ { \mathrm { g t } } | \mathbf { z } , \mathbf { t } )$ for ground truth label $y _ { \mathrm { g t } }$ by using cross-entropy loss $\ell _ { \mathrm { C E } } ( { \mathbf { z } } , { \mathbf { t } } , y _ { \mathrm { g t } } )$ which is defined as follows:

$$
\ell _ { \mathrm { C E } } ( \mathbf { z } , \mathbf { t } , y _ { \mathrm { g t } } ) = \log \operatorname* { P r } ( y = y _ { \mathrm { g t } } | \mathbf { z } , \mathbf { t } ) ,
$$

where the predicted probability is computed as Eq. (5).

## 4.2 Adaptive Ensemble for Evaluation

Due to the various levels of transfer difficulty encountered during deployment, an adaptive method is necessary to avoid suboptimal results for each target domain. Motivated by our observations, in the evaluation stage, we use an adaptive ensemble approach that combines pre-adapter $( \tilde { \mathbf { t } } _ { \mathrm { e v a l } } )$ and post-adapter text features (Eq. (11)), described as follows:

$$
\mathbf { t } _ { \mathrm { e v a l } } = \alpha _ { \mathrm { e v a l } } \cdot ( \mathbf { A } ^ { \top } \tilde { \mathbf { t } } _ { \mathrm { e v a l } } + \mathbf { b } ) + ( 1 - \alpha _ { \mathrm { e v a l } } ) \cdot \tilde { \mathbf { t } } _ { \mathrm { e v a l } } ,
$$

![](images/4865cd4337db7cc87a828671aa9abe10529f278a6f70eab241f28f6def471b0b.jpg)  
Figure 9: A concept figure for calculating the adaptive coefficient $\alpha _ { \mathrm { e v a l } }$ for ensemble upon its class distance.

where $\alpha _ { \mathrm { e v a l } }$ is the ensemble coefficient for a target class at evaluation and $\mathbf { t } _ { \mathrm { e v a l } }$ is the final representation for that class. With this ensemble approach, for domains with high RTD, the model relies on the adaptability and generalizability of VPT and TA. Conversely, for domains with low RTD, it leverages general knowledge from the pre-trained model to avoid excessive adaptation.

To determine the optimal $\alpha _ { \mathrm { e v a l } }$ for each class, which estimates transfer difficulty and acts as a controller for adaptation, we employ a non-parametric method based on the distance between the text features of the evaluation class and the classes learned during training. This approach is based on the assumption that in domains with high RTD, class features are typically less separable in the text embedding space, similarly to their separability in the image embedding space. Hence, domains like EuroSAT exhibit low class distances, while those with low RTD, such as Stanford Cars, display high class distances. Fig. 8 shows that the optimal α, used in Eq. (10) and Tab. 1, is highly correlated with the distance between class features. This tendency suggests that $\alpha _ { \mathrm { e v a l } }$ based on the distance between class features can effectively represent transfer difficulty.

Moreover, instead of applying a single $\alpha _ { \mathrm { e v a l } }$ for all classes, we adopt a class-wise approach. This is because, within the same domain, target features considered as out-of-task should rely more on the general knowledge of pre-trained VLMs, whereas features closer to the learned classes should leverage more task-specific knowledge. With regard to this, we adaptively set $\alpha _ { \mathrm { e v a l } }$ by comparing the text feature of the evaluation class with the features of the learned classes, as illustrated in Fig. 9. Specifically, we calculate both the average and nearest distances between the evaluation class and the C

Table 2: Accuracy comparison on base-to-novel generalization of APEX with previous methods.
<table><tr><td>Dataset</td><td></td><td>CLIP</td><td>CLIP -Adapter</td><td> $\mathbf { . c _ { o 0 p } }$ </td><td>MaPLe</td><td> $\mathbf { { P r o } } _ { - \mathbf { { G r a d } } }$ </td><td>APEX</td></tr><tr><td rowspan="3">Average on 11 datasets</td><td>Base</td><td>69.34</td><td>83.23</td><td>81.11</td><td>82.52</td><td>82.55</td><td>83.99</td></tr><tr><td>Novel</td><td>74.22</td><td>70.13</td><td>70.55</td><td>74.24</td><td>72.20</td><td>76.76</td></tr><tr><td>HM</td><td>71.70</td><td>75.64</td><td>75.03</td><td>77.86</td><td>76.77</td><td>80.04</td></tr><tr><td rowspan="3">ImageNet</td><td>Base</td><td>72.43</td><td>76.06</td><td>76.47</td><td>77.02</td><td>76.97</td><td>77.12</td></tr><tr><td>Novel</td><td>68.14</td><td>68.40</td><td>69.60</td><td>70.15</td><td>67.20</td><td>71.10</td></tr><tr><td>HM</td><td>70.22</td><td>72.03</td><td>72.87</td><td>73.42</td><td>71.75</td><td>73.99</td></tr><tr><td rowspan="3">Caltech101</td><td>Base</td><td>96.84</td><td>98.00</td><td>97.70</td><td>97.95</td><td>97.88</td><td>98.18</td></tr><tr><td>Novel</td><td>94.00</td><td>93.66</td><td>93.96</td><td>94.60</td><td>93.57</td><td>95.06</td></tr><tr><td>HM</td><td>95.40</td><td>95.78</td><td>95.78</td><td>96.25</td><td>95.68</td><td>96.59</td></tr><tr><td rowspan="3">OxfordPets</td><td>Base</td><td>91.17</td><td>94.86</td><td>95.66</td><td>95.80</td><td>95.00</td><td>95.11</td></tr><tr><td>Novel</td><td>97.26</td><td>94.49</td><td>96.32</td><td>97.82</td><td>97.46</td><td>97.27</td></tr><tr><td>HM</td><td>94.12</td><td>94.67</td><td>95.99</td><td>96.80</td><td>96.21</td><td>96.18</td></tr><tr><td rowspan="3">Stanford Cars</td><td>Base</td><td>63.37</td><td>77.62</td><td>72.92</td><td>74.69</td><td>78.64</td><td>80.53</td></tr><tr><td>Novel</td><td>74.89</td><td>68.53</td><td>71.98</td><td>73.53</td><td>70.23</td><td>75.08</td></tr><tr><td>HM</td><td>68.65</td><td>72.79</td><td>72.45</td><td>74.11</td><td>74.20</td><td>77.71</td></tr><tr><td rowspan="3">Flowers102</td><td>Base</td><td>72.08</td><td>96.88</td><td>94.82</td><td>95.90</td><td>94.83</td><td>97.47</td></tr><tr><td>Novel</td><td>77.80</td><td>69.20</td><td>70.71</td><td>72.96</td><td>74.70</td><td>77.58</td></tr><tr><td>HM</td><td>74.83</td><td>80.73</td><td>81.01</td><td>82.87</td><td>83.57</td><td>86.40</td></tr><tr><td rowspan="3">Food101</td><td>Base</td><td>90.10</td><td>90.02</td><td>90.63</td><td>90.46</td><td>90.40</td><td>89.60</td></tr><tr><td>Novel</td><td>91.22</td><td>89.76</td><td>91.13</td><td>91.71</td><td>90.43</td><td>92.06</td></tr><tr><td>HM</td><td>74.83</td><td>89.89</td><td>90.88</td><td>91.08</td><td>90.41</td><td>90.81</td></tr><tr><td rowspan="3">FGVC Aircraft</td><td>Base</td><td>27.19</td><td>40.14</td><td>36.19</td><td>37.76</td><td>40.77</td><td>42.69</td></tr><tr><td>Novel</td><td>36.29</td><td>31.77</td><td>26.82</td><td>34.67</td><td>30.16</td><td>35.21</td></tr><tr><td>HM</td><td>31.09</td><td>35.47</td><td>30.81</td><td>36.15</td><td>34.67</td><td>38.59</td></tr><tr><td rowspan="3">SUN397</td><td>Base</td><td>69.36</td><td>81.72</td><td>80.55</td><td>81.33</td><td>81.19</td><td>81.17</td></tr><tr><td>Novel</td><td>75.35</td><td>73.54</td><td>75.48</td><td>77.75</td><td>73.42</td><td>78.98</td></tr><tr><td>HM</td><td>72.23</td><td>77.41</td><td>77.93</td><td>79.50</td><td>77.11</td><td>80.06</td></tr><tr><td rowspan="3">DTD</td><td>Base</td><td>53.24</td><td>81.77</td><td>77.34</td><td>79.34</td><td>76.64</td><td>82.45</td></tr><tr><td>Novel</td><td>59.90</td><td>49.02</td><td>48.86</td><td>56.64</td><td>54.23</td><td>63.80</td></tr><tr><td>HM</td><td>56.37</td><td>61.29</td><td>59.89</td><td>66.10</td><td>63.52</td><td>71.94</td></tr><tr><td rowspan="3">EuroSAT</td><td>Base</td><td>56.48</td><td>91.55</td><td>87.05</td><td>93.00</td><td>91.23</td><td>92.83</td></tr><tr><td>Novel</td><td>64.05</td><td>61.10</td><td>61.27</td><td>69.17</td><td>68.58</td><td>79.89</td></tr><tr><td>HM</td><td>60.03</td><td>73.29</td><td>71.92</td><td>79.33</td><td>78.30</td><td>85.88</td></tr><tr><td rowspan="3">UCF101</td><td>Base</td><td>70.53</td><td>86.87</td><td>82.86</td><td>84.43</td><td>84.54</td><td>86.74</td></tr><tr><td>Novel</td><td>77.50</td><td>71.94</td><td>69.92</td><td>77.64</td><td>74.24</td><td>78.37</td></tr><tr><td>HM</td><td>73.85</td><td>78.70</td><td>75.84</td><td>80.89</td><td>79.06</td><td>82.34</td></tr></table>

learned classes in the following manner:

$$
\begin{array} { r } { d _ { \mathrm { e v a l } } ^ { \mathrm { a v g } } = 1 . 0 - \frac { 1 } { C } \sum _ { j = 1 } ^ { C } \sin ( \mathbf { t ^ { \prime } } _ { \mathrm { e v a l } } , \mathbf { t ^ { \prime } } _ { j } ) , } \\ { d _ { \mathrm { e v a l } } ^ { \mathrm { n n } } = 1 . 0 - \underset { \forall j \in \{ 1 , \dots , C \} } { \operatorname* { m i n } } \ s i m ( \mathbf { t ^ { \prime } } _ { \mathrm { e v a l } } , \mathbf { t ^ { \prime } } _ { j } ) , } \end{array}
$$

where $\mathbf { t } _ { \mathrm { \ e v a l } } ^ { \prime }$ and $\mathbf { t } _ { j } ^ { \prime }$ indicate text feature of evaluation class and learned class $j \in \{ 1 , \ldots , C \}$ from pre-trained VLMs and sim denotes cosine similarity. Using these distance metrics, we compute the coefficient $\alpha _ { \mathrm { e v a l } }$ as follows:

$$
\alpha _ { \mathrm { e v a l } } = \exp \left( - \beta \cdot ( d _ { \mathrm { e v a l } } ^ { \mathrm { a v g } } ) \cdot { \bf 1 } _ { ( d _ { \mathrm { e v a l } } ^ { \mathrm { n n } } > \epsilon ) } \right) ,
$$

where $\beta$ is a scaling factor. The equation indicates a preference for pre-adapter features when the text feature distance from learned classes is large, and for trained TA when it is small. The condition of $d _ { \mathrm { { e v a l } } } ^ { \mathrm { { n n } } } > \epsilon .$ , where ϵ is a small value set at 0.05, serves to treat an evaluation class that is very similar to the base class as identical. This adaptive $\alpha _ { \mathrm { e v a l } }$ enables flexible use of general and task-specific knowledge. Moreover, since text embeddings are usually pre-calculated (Radford et al., 2021), this adaptive coefficient incurs only a minor computational overhead.

Vision Ensemble. Additionally, to further improve the performance by leveraging more general knowledge of the pretrained VLMs, we can also employ an ensemble technique for the visual encoder that combines the visual feature of the pretrained VLM (z′) with the task-adapted VLMs (z) as follows:

$$
{ \bf z } = \bar { \alpha } \cdot { \bf z } ^ { \prime } + ( 1 - \bar { \alpha } ) \cdot { \bf z } ,
$$

α¯, the mean value of $\alpha _ { \mathrm { e v a l } }$ , is used for image ensemble since class-specific $\alpha _ { \mathrm { e v a l } }$ cannot be applied at the image level.

## 5 Experiments

We describe our experimental setup and results for verifying superiority of our method. Additional experimental results are described in Appendix C.

## 5.1 Experimental Setup

Datasets. We evaluate APEX on the three most commonly used transfer learning tasks: base-tonovel generalization, cross-dataset evaluation, and domain generalization. For all the few-shot experiments except domain generalization, we follow CoCoOp (Zhou et al., 2022a) which uses 11 image recognition datasets. The datasets cover multiple recognition tasks including ImageNet (Deng et al., 2009) and Caltech101 (Fei-Fei et al., 2004) which consists of generic objects; Oxford-Pets (Parkhi et al., 2012), Stanford Cars (Krause et al., 2013), Flowers102 (Nilsback and Zisserman, 2008), Food101 (Bossard et al., 2014), and FGVC Aircraft (Maji et al., 2013) for fine-grained classification, SUN397 (Xiao et al., 2010) for scene recognition, UCF101 (Soomro et al., 2012) for action recognition, DTD (Cimpoi et al., 2013) for texture classification, and EuroSAT (Helber et al., 2017) which consists of satellite images. For the domain generalization benchmark, we use ImageNet as a source dataset and use ImageNet-A (Hendrycks et al., 2019), ImageNet-R (Hendrycks et al., 2020), ImageNet-Sketch (Wang et al., 2019), and ImageNetV2 (Recht et al., 2019) as out-of-domain datasets.

Experimental Details. We use multiple baselines for comparison with our methods in experiments. These include the standard zero-shot CLIP (Radford et al., 2021), CLIP-Adapter (Gao et al., 2023), CoCoOp (Zhou et al., 2022a) and MaPLe (khattak et al., 2023). We also consider ProGrad (Zhu et al., 2023a), which uses gradient alignment for prompt learning. When reporting results, we have reproduced all the experiments, as we observe that the values are highly dependent on the random seed. Instead of taking the average results from three seeds, as done in previous works (khattak et al., 2023), we use the average of 20 seeds to determine the final value for base-to-novel and the average of 5 seeds for cross-evaluation and domain-generalization. Additionally, we found that using the Adadelta optimizer (Zeiler, 2012) yields better results, so we have reproduced the experiments with Adadelta. More experimental details can be found in the Appendix A.

Table 3: Comparison of accuracy on cross-dataset of APEX with previous methods.
<table><tr><td colspan="2">Dataset</td><td>C-Adapter</td><td>CoCoOp</td><td>MaPLe</td><td>ProGrad</td><td>APEX</td></tr><tr><td>Source</td><td>ImageNet</td><td>70.12</td><td>71.46</td><td>70.58</td><td>71.73</td><td>72.00</td></tr><tr><td rowspan="10"></td><td>Caltech101</td><td>92.94</td><td>93.24</td><td>93.46</td><td>93.30</td><td>94.46</td></tr><tr><td>OxfordPets</td><td>86.80</td><td>90.38</td><td>90.28</td><td>89.95</td><td>90.06</td></tr><tr><td>Cars</td><td>64.22</td><td>64.08</td><td>65.22</td><td>65.25</td><td>65.46</td></tr><tr><td>Flower102</td><td>69.06</td><td>70.50</td><td>71.80</td><td>69.34</td><td>71.58</td></tr><tr><td>Food101</td><td>85.20</td><td>85.64</td><td>86.24</td><td>86.22</td><td>86.44</td></tr><tr><td>Aircraft</td><td>24.24</td><td>21.58</td><td>23.62</td><td>21.22</td><td>24.44</td></tr><tr><td>SUN397</td><td>64.36</td><td>66.30</td><td>67.32</td><td>65.32</td><td>67.20</td></tr><tr><td>DTD</td><td>43.44</td><td>43.68</td><td>45.04</td><td>42.19</td><td>45.70</td></tr><tr><td>EuroSAT</td><td>47.66</td><td>45.48</td><td>46.24</td><td>45.33</td><td>47.58</td></tr><tr><td>UCF101</td><td>65.52</td><td>67.42</td><td>68.26</td><td>67.62</td><td>68.80</td></tr><tr><td colspan="2">Average</td><td>64.34</td><td>64.83</td><td>65.75</td><td>64.57</td><td>66.16</td></tr></table>

Table 4: Comparison of accuracy on domain generalization of APEX with previous methods.
<table><tr><td rowspan="2"></td><td>Source</td><td colspan="5">Target</td></tr><tr><td>ImageNet</td><td>-V2</td><td>-S</td><td>-A</td><td>-R</td><td>Avg.</td></tr><tr><td>C-Adapter</td><td>70.12</td><td>61.78</td><td>46.70</td><td>48.56</td><td>74.00</td><td>57.76</td></tr><tr><td>CoCoOp</td><td>71.46</td><td>64.44</td><td>48.58</td><td>50.20</td><td>75.64</td><td>59.72</td></tr><tr><td>MaPLe</td><td>70.58</td><td>63.95</td><td>48.78</td><td>50.53</td><td>76.78</td><td>59.90</td></tr><tr><td>ProGrad</td><td>71.73</td><td>64.54</td><td>48.59</td><td>50.38</td><td>75.87</td><td>59.85</td></tr><tr><td>APEX</td><td>72.00</td><td>64.70</td><td>48.48</td><td>50.68</td><td>76.76</td><td>60.16</td></tr></table>

## 5.2 Main Results

Base-to-Novel Generalization. In this scenario, the datasets are evenly divided into base and novel categories. The model is trained on the base classes using 16 shots and is subsequently tested on both the base and novel classes. As indicated in Table 2, APEX consistently outperforms the best of the previous methods in average accuracy across all datasets, with a margin of 1 6%. In particular, our method exhibits superior performance in novel classes on all datasets, demonstrating APEX’s enhanced generalizability. The exceptions are Oxford Pets and FGVC Aircraft, where the performance is already exceptionally high and low, respectively. This improvement is especially notable in domains with high RTD, such as EuroSAT (+15.84%) and

Table 5: Comparison of the effect of adaptive ensemble technique between text and visual encoder by RTD.
<table><tr><td>Text</td><td>Visual</td><td>Easy</td><td>Challenge</td><td>All</td></tr><tr><td>x</td><td>x</td><td>70.67</td><td>58.25</td><td>74.61</td></tr><tr><td>√</td><td>x</td><td> $7 4 . 5 1 \ ( + 3 . 8 4 )$ </td><td> $5 8 . 6 6 ( + 0 . 4 1 ) $ </td><td> $7 6 . 1 9 \left( + 1 . 5 8 \right)$ </td></tr><tr><td>x</td><td>√</td><td> $7 0 . 7 9 \ ( + 0 . 1 2 )$ </td><td> $5 8 . 6 5 \ : ( + 0 . 4 0 )$ </td><td>74.83 (+0.22)</td></tr><tr><td>√</td><td>√</td><td> $7 5 . 0 5 \ ( + 4 . 3 8 )$ </td><td> ${ \pmb 5 9 . 6 3 \left( + 1 . 3 8 \right) }$ </td><td> $7 6 . 7 6 \ : ( + 2 . 1 5 )$ </td></tr></table>

DTD (+3.90%). Additionally, the APEX method also shows superior performance in base categories, highlighting the high adaptability of our approach.

Cross-dataset Evaluation. We train the model to generalize across different domains by using a cross-dataset evaluation task. Specifically, we first train the model on the ImageNet dataset and then transfer it to the 10 other datasets. Table 3 summarizes that APEX shows the best overall performance compared to existing baselines. Our proposed method achieves the best performance on 7 out of 11 tasks. This demonstrates APEX’s effectiveness, especially in difficult situations where both the task and domain are unseen.

Domain Generalization. We assess the capability of APEX to generalize to out-of-distribution data by training on the source dataset, ImageNet, and subsequently testing on various modified versions of ImageNet. Our method does not achieve a large margin of superiority since our adaptive ensemble is primarily designed to enhance performance in novel classes. Nonetheless, our method still surpasses all baseline models on average accuracy in this domain generalization task.

## 5.3 Ablation Study

In this section, we provide ablation experiments on APEX. Full results are detailed in Appendix C.

Effect of Ensemble. We have conducted a component analysis of two adaptive ensemble techniques of APEX, focusing on (1) the text encoder and (2) the visual encoder. The results, as shown in Table 5, reveal that the ensembling of the text encoder is crucial for enhancing performance. Conversely, ensembling the visual encoder results in a minor yet consistent improvement. The text ensemble notably achieves substantial improvements in domains with low RTD, implying that task-specific knowledge is primarily acquired through TA. Overall, employing both ensemble techniques leads to the most improvement regardless of RTD.

Using Low-Rank Linear Adapter. CLIP-Adapter (Gao et al., 2023) and Tip-Adapter (Zhang et al., 2022) utilize the bottleneck layer (He et al., 2016) which shrinks and re-expands the feature dimensions to improve efficiency. Similarly, we utilize low-rank matrix factorization that $\mathbf { A } = \mathbf { U } \mathbf { V } ^ { \top }$ where V, $\mathbf { U } \in \mathbb { R } ^ { d _ { l } \times d _ { r } }$ with $d _ { r } < d _ { l }$ to improve the parameter efficiency. Fig. 10 shows that although TA’s performance diminishes with decreasing dimension $d _ { r }$ , average accuracy with few parameters $( d _ { r } ~ = ~ 3 2 )$ still achieves performance comparable to ProGrad (Zhu et al. 2023a; +0.72%). Moreover, the linear adapter consistently outperforms the non-linear adapter (Gao et al., 2023) across all values of $d _ { r }$ , motivating us to use a linear adapter in our proposed APEX.

![](images/295f7bd19fedd17a23a009366a28fa98301084b4998906854f7ef4b262011d7a.jpg)

![](images/ea6a7d6fb605b38a256ac38c63b76ed45a83c94a5d8368ecabac15e2e8ae6c1f.jpg)  
Linear Non-linear (CLIP-Adapter)

![](images/8671e651e1a7b3185ebcbb4a3132c5455befafd71c727afde4927327775d08f1.jpg)  
Figure 10: Comparison of the accuracy of base, novel, and their harmonic mean using low-rank linear adapter and bottleneck layer of non-linear adapter (Gao et al., 2023).

## 6 Conclusion

We propose APEX to address the challenges of conventional prompt and adapter-style ETL methods for VLMs. Our approach incorporates two key components based on our observations: (1) using VPT and TA for exploiting the property of each modality and (2) adaptive ensemble coefficient in the inference stage. We empirically demonstrate the superior performance of APEX, consistently achieving a better performance than the previous methods.

## Acknowledgements

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.2019-0-00075, Artificial Intelligence Graduate School Program (KAIST), 5%), Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. RS-2024- 00457882, AI Research Hub Project), 5%), and Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2022-0-00641, XVoice: Multi-Modal Voice Meta Learning, 90%).

## Limitation

We focus on two types of ETL, prompt tuning and adapter-style tuning, for VLMs for vision-language understanding tasks such as CLIP, EVA-CLIP, and CoCA-CLIP. While our extensive analyses provide valuable insights, our paper primarily centers on understanding tasks, with opportunities for further exploration in vision-language generation tasks such as BLIP (Li et al., 2022a) and LLaVA (Liu et al., 2024). Additionally, though we focus on two main representative ETL methods, further analyses could be conducted on other ETL methods like LoRA (Hu et al., 2022) and IA3 (Liu et al., 2022). We leave these aspects for future work but wish to emphasize the comprehensive exploration provided by our study on the two representative ETL methods for VLMs.

## References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. 2022. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736.

Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. 2014. Food-101 - mining discriminative components with random forests. In European Conference on Computer Vision.

Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. 2013. Describing textures in the wild. 2014 IEEE Conference on Computer Vision and Pattern Recognition, pages 3606–3613.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Fei-Fei Li. 2009. Imagenet: a large-scale hierarchical image database. pages 248–255.

Li Fei-Fei, Rob Fergus, and Pietro Perona. 2004. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. 2004 Conference on Computer Vision and Pattern Recognition Workshop, pages 178– 178.

Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang, Hongsheng Li, and Yu Qiao. 2023. Clip-adapter: Better vision-language models with feature adapters. International Journal ofComputer Vision, pages 1–15.

Shashank Goel, Hritik Bansal, Sumit Bhatia, Ryan Rossi, Vishwa Vinay, and Aditya Grover. 2022. Cyclip: Cyclic contrastive language-image pretraining. Advances in Neural Information Processing Systems, 35:6704–6719.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770– 778.

Patrick Helber, Benjamin Bischke, Andreas R. Dengel, and Damian Borth. 2017. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal ofSelected Topics in Applied Earth Observations and Remote Sensing, 12:2217–2226.

Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Lixuan Zhu, Samyak Parajuli, Mike Guo, Dawn Xiaodong Song, Jacob Steinhardt, and Justin Gilmer. 2020. The many faces of robustness: A critical analysis of out-of-distribution generalization. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 8320–8329.

Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Xiaodong Song. 2019. Natural adversarial examples. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15257–15266.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. 2021. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 4904–4916. PMLR.

Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. 2022. Visual prompt tuning. In European Conference on Computer Vision (ECCV).

Muhammad Uzair khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, and Fahad Shahbaz Khan. 2023. Maple: Multi-modal prompt learning. In The IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Muhammad Uzair Khattak, Syed Talal Wasim, Muzammal Naseer, Salman Khan, Ming-Hsuan Yang, and Fahad Shahbaz Khan. 2023. Self-regulating prompts: Foundational model adaptation without forgetting. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 15190–15200.

Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 2013. 3d object representations for fine-grained categorization. 2013 IEEE International Conference on Computer Vision Workshops, pages 554–561.

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022a. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888–12900. PMLR.

Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. 2021. Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems, 34:9694–9705.

Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. 2022b. Grounded language-image pre-training. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10965– 10975.

Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, and Kaiming He. 2023. Scaling language-image pre-training via masking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23390–23400.

Haokun Liu, Derek Tam, Muqeeth Mohammed, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin Raffel. 2022. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. In Advances in Neural Information Processing Systems.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2024. Visual instruction tuning. Advances in neural information processing systems, 36.

Xuejing Liu, Wei Tang, Jinghui Lu, Rui Zhao, Zhaojun Guo, and Fei Tan. 2023. Deeply coupled cross-modal prompt learning. arXiv preprint arXiv:2305.17903.

Yuning Lu, Jianzhuang Liu, Yonggang Zhang, Yajing Liu, and Xinmei Tian. 2022. Prompt distribution learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5206–5215.

Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew B. Blaschko, and Andrea Vedaldi. 2013. Finegrained visual classification of aircraft. ArXiv, abs/1306.5151.

Maria-Elena Nilsback and Andrew Zisserman. 2008. Automated flower classification over a large number of classes. 2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing, pages 722–729.

Jaehoon Oh, Hyungjun Yoo, ChangHwan Kim, and Se-Young Yun. 2021. {BOIL}: Towards representation change for few-shot learning. In International Conference on Learning Representations.

Michal Pándy, Andrea Agostinelli, Jasper Uijlings, Vittorio Ferrari, and Thomas Mensink. 2022. Transferability estimation using bhattacharyya class separability. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9172–9182.

Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. 2012. Cats and dogs. 2012 IEEE Conference on Computer Vision and Pattern Recognition, pages 3498–3505.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR.

Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. 2019. Do imagenet classifiers generalize to imagenet? In International Conference on Machine Learning.

Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. 2012. Ucf101: A dataset of 101 human actions classes from videos in the wild. ArXiv, abs/1212.0402.

Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. 2023. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389.

Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-sne. Journal of machine learning research, 9(11).

Haohan Wang, Songwei Ge, Eric P. Xing, and Zachary Chase Lipton. 2019. Learning robust global representations by penalizing local predictive power. In Neural Information Processing Systems.

Jianxiong Xiao, James Hays, Krista A. Ehinger, Aude Oliva, and Antonio Torralba. 2010. Sun database: Large-scale scene recognition from abbey to zoo. 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, pages 3485– 3492.

Hantao Yao, Rui Zhang, and Changsheng Xu. 2023. Visual-language prompt tuning with knowledgeguided context optimization. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6757–6767.

Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. 2022. FILIP: Finegrained interactive language-image pre-training. In International Conference on Learning Representations.

Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. 2022. Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917.

Tao Yu, Zhihe Lu, Xin Jin, Zhibo Chen, and Xinchao Wang. 2023. Task residual for tuning visionlanguage models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10899–10909.

Yuhang Zang, Wei Li, Kaiyang Zhou, Chen Huang, and Chen Change Loy. 2022. Unified vision and language prompt learning. arXiv preprint arXiv:2210.07225.

Matthew D Zeiler. 2012. Adadelta: an adaptive learning rate method. arXiv preprint arXiv:1212.5701.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023. Sigmoid loss for language image pre-training. arXiv preprint arXiv:2303.15343.

Renrui Zhang, Wei Zhang, Rongyao Fang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, and Hongsheng Li. 2022. Tip-adapter: Training-free adaption of clip for few-shot classification. In Computer Vision – ECCV 2022, pages 493–510, Cham. Springer Nature Switzerland.

Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. 2022a. Conditional prompt learning for vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16816–16825.

Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. 2022b. Learning to prompt for visionlanguage models. International Journal ofComputer Vision, 130(9):2337–2348.

Beier Zhu, Yulei Niu, Yucheng Han, Yue Wu, and Hanwang Zhang. 2023a. Prompt-aligned gradient for prompt tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15659–15669.

Xiangyang Zhu, Renrui Zhang, Bowei He, Aojun Zhou, Dong Wang, Bin Zhao, and Peng Gao. 2023b. Not all features matter: Enhancing few-shot clip with adaptive prior refinement. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2605–2615.

## A Implementation Details

As explained in Section 5, we utilize the ViT-B/16 model as the CLIP image encoder and a standard GPT2-like structure with an End Of Text (EOT) token as the classification token for the text encoder. To implement APEX, we use visual prompts for all layers, setting $J _ { V } = 1 2$ for base-to-novel generalization and $J _ { V } = 3$ for cross-evaluation and domain generalization. The text prompt is applied only to the shallow prompt, and therefore, $J _ { V } = 1$ for all experiments. The number of prompts for each layer, $b _ { \nu }$ and $b _ { \mathcal { T } }$ , is set to 2. The initial text prompt is fixed as $^ { * } a$ photo $o f a ^ { \prime \prime } { . }$ , and the visual prompts are initialized with a zero-mean Gaussian distribution with a standard deviation of 0.02. The matrix term of the text adapter is initialized with an identity matrix, and the bias vector is initialized with a zero vector.

For training, we use the Adadelta optimizer (Zeiler, 2012) with a learning rate of 0.15 and a cosine learning rate scheduler. The batch size is set to 16, and we train for 15 epochs, except for ImageNet, where we train for 5 epochs. As in previous works, we apply augmentation techniques of random cropping and flipping. The scaling factor $\beta ,$ used for calculating $\alpha _ { e v a l } .$ , is set to 4.0. In the SGD experiments presented in Appendix C, we adopt a batch size of 16 and epochs of 30 and 5 for ImageNet, along with a learning rate of 0.0015 and a cosine learning rate scheduler. The augmentation and scaling factors are set the same as in the Adadelta experiments.

For reproducing baselines, we use the Adadelta optimizer with a learning rate of 0.25, selected after a grid search with values [0.1, 0.15, 0.2, 0.25, 0.3]. The rest of the settings remain the same as in the original papers. Results with their original configurations using SGD optimizer are listed in $\mathsf { A p - }$ pendix C. All our experiments were conducted on a single NVIDIA RTX 3090.

## B Notation and Algorithm

In this section, we present the notation and algorithm of our method, APEX. The notation is detailed in Table 6. The training algorithm for APEX is outlined in Algorithm 1, and the adaptive inference algorithm is presented in Algorithm 2.

Table 6: The notation table for Section 3
<table><tr><td>Notation</td><td>Description</td></tr><tr><td colspan="2">The notation for VLMs</td></tr><tr><td> $\nu$ </td><td>The visual encoder of VLMs</td></tr><tr><td> $\tau$ </td><td>The text encoder of VLMs</td></tr><tr><td> $L _ { \nu }$ </td><td>The number of layers of visual encoder</td></tr><tr><td> $L \tau$ </td><td>The number of layers of text encoder</td></tr><tr><td> $\nu _ { \ell }$ </td><td>The  $\ell ^ { \mathrm { { t h } } }$  Transformer layer of visual encoder</td></tr><tr><td> $\tau _ { \ell }$ </td><td>The  $\ell ^ { \mathrm { { t h } } }$  Transformer layer of text encoder</td></tr><tr><td> $\mathbf { E } _ { \ell }$ </td><td>The patch embeddings of  $\ell ^ { \mathrm { { t h } } }$  layer of visual encoder</td></tr><tr><td> $\mathbf { W } _ { \ell }$ </td><td>The word embeddings of  $\ell ^ { \mathrm { { t h } } }$  layer of text encoder</td></tr><tr><td colspan="2">The inputs for VLMs or prompt tuning</td></tr><tr><td> $J _ { \mathcal { V } }$ </td><td></td></tr><tr><td> $J _ { T }$ </td><td>The number of layers of VPT The number of layers of TPT</td></tr><tr><td> $b _ { \mathcal { V } }$ </td><td>The context length of  $\mathrm { { V P T } }$ </td></tr><tr><td> $b \tau$ </td><td>The context length of TPT</td></tr><tr><td> $\hat { \mathbf { P } } _ { \ell }$ </td><td>The visual prompt of  $\ell ^ { \mathrm { { t h } } }$  layer of visual encoder</td></tr><tr><td> $\mathbf { P } _ { \ell }$ </td><td>The text prompt of  $\ell ^ { \mathrm { { t h } } }$  layer of text encoder</td></tr><tr><td colspan="2">The outputs for VLMs</td></tr><tr><td> $\mathbf { c } _ { \boldsymbol { \ell } }$ </td><td>The embedded features of  $\ell ^ { \mathrm { { t h } } }$  layer for [CLS] token</td></tr><tr><td> $\mathbf { t } _ { i }$ </td><td>The text features of  $i ^ { \mathrm { t h } }$  class</td></tr><tr><td> $\mathbf { z }$ </td><td>The visual features from visual encoder</td></tr><tr><td colspan="2">The outputs for VLMs related to APEX</td></tr><tr><td> $\mathbf { z } ^ { \prime }$ </td><td>The visual features from visual encoder of pretrained VLMs for adaptive ensemble</td></tr><tr><td> $\mathbf { t } ^ { \prime }$ </td><td>The text features from text encoder of pretrained VLMs for adaptive ensemble</td></tr><tr><td></td><td>The pre-adapter text features of text encoder of</td></tr><tr><td>it</td><td>adapted VLMs</td></tr></table>

Algorithm 1 Pseudo-Algorithm for Training of   
APEX   
Require: Pretrained visual encoder , Pretrained   
text encoder , Learnable vision prompts $\hat { \mathbf { P } }$   
Shallow text prompts $\mathbf { P _ { 0 } } ,$ , Adapter parameter  
ized by matrix A and b   
Require: Training Samples , Initial Text Embed  
dings $\mathbf { W } _ { 0 }$   
1: Randomly initialize $\phi = [ \hat { \mathbf { P } } , \mathbf { A } , \mathbf { b } ]$   
2: while not done do   
3: Sample Batch $\boldsymbol { B } = ( I , y _ { g t } )$   
4: ${ \bf E } _ { 0 } = \mathtt { P a t h I }$ Embedding(I)   
5: for $i = 1 , \ldots , J _ { \mathcal { V } }$ do   
6: $[ \mathbf { c } _ { i } , \mathbf { E } _ { i } , \_ ] \longleftarrow \mathcal { V } _ { i } ( [ \mathbf { c } _ { i - 1 } , \mathbf { E } _ { i - 1 } , \hat { \mathbf { P } } _ { i - 1 } ] )$   
7: end for   
8: for $i = J _ { \mathcal { V } } + 1 , \dotsc , L \gamma$ do   
9: $[ \mathbf { c } _ { i } , \mathbf { E } _ { i } , \hat { \mathbf { P } } _ { i } ] \gets \mathcal { V } _ { i } ( [ \mathbf { c } _ { i - 1 } , \mathbf { E } _ { i - 1 } , \hat { \mathbf { P } } _ { i - 1 } ] )$   
10: end for   
11: $\mathbf { z } \gets \mathtt { I m a g e P r o j } ( \mathbf { c } _ { L _ { \nu } } )$   
12: $\tilde { \mathbf { t } } = \mathcal { T } ( [ \mathbf { W _ { 0 } } , \mathbf { P _ { 0 } } ] )$   
13: $\mathbf { t } = \mathbf { A } ^ { \top } \tilde { \mathbf { t } } + \mathbf { b }$   
14: $/ *$ Calculate the probability for class $i ^ { * } /$   
15: $\begin{array} { r } { \operatorname* { P r } ( y = i | \mathbf { z } , \mathbf { t } ) = \frac { \exp ( \sin ( \mathbf { z } , \mathbf { t } _ { i } ) / \tau ) } { \sum _ { j = 1 } ^ { C } \exp ( \sin ( \mathbf { z } , \mathbf { t } _ { j } ) / \tau ) } } \end{array}$   
16: $\ell _ { \mathrm { C E } } ( \mathbf { z } , \mathbf { t } , y _ { \mathrm { g t } } ) = \log \mathbf { \tilde { P r } } ( y = y _ { \mathrm { g t } } | \mathbf { z } , \mathbf { t } )$   
17: $\phi = \phi - \gamma \nabla _ { \phi } \ell _ { \mathrm { C E } } ( \mathbf { z } , \mathbf { t } , y _ { \mathrm { g t } } ; \phi )$   
18: end while

## C Additional Experiments

## C.1 Ablation on Adaptive Ensemble

Table 7 illustrates the complete results of the component analysis of the adaptive ensemble. We only display results for novel classes, as these ensemble components do not affect the results for base classes, given that $\alpha _ { e v a l }$ is set to 1.0 for seen classes. AThe ensemble of the text encoder is crucial as its removal leads to a significant performance drop in domains with low RTD, such as Stanford Cars and SUN397. This demonstrates that moderating TA with an adaptive ensemble helps to leverage both task-specific knowledge and general VLMs knowledge effectively. The ensemble on the visual encoder offers marginal improvement, but combining both still yields the most superior performance on average.

## C.2 Results on Low-Rank Experiments

Figure 11 presents detailed results for each dataset using low-rank methods. The result demonstrates that our linear adapter provides better overall results, particularly for novel classes across most datasets. This parameter-efficient approach exhibits relative robustness in performance, even outperforming MaPLe (khattak et al., 2023) for rank 64 (+0.32%) on average. These encouraging results have led us to adopt the linear adapter for the text encoder. Furthermore, we observe that initializing the adapter with an identity matrix improves performance, a strategy that can be explored more thoroughly in future work.

Table 7: Comparison of the effect of adaptive ensemble technique between text and visual encoder by RTD.
<table><tr><td>Visual Text</td><td>x x</td><td>x √</td><td> $\checkmark$  x</td><td>√(APEX) √(APEX)</td></tr><tr><td>ImageNet</td><td>69.08</td><td>70.09</td><td>69.22</td><td>71.10</td></tr><tr><td>Caltech101</td><td>94.91</td><td>94.80</td><td>95.01</td><td>95.06</td></tr><tr><td>OxfordPets</td><td>97.24</td><td>97.39</td><td>97.07</td><td>97.27</td></tr><tr><td>Cars</td><td>68.40</td><td>74.46</td><td>68.32</td><td>75.08</td></tr><tr><td>Flower102</td><td>73.71</td><td>76.40</td><td>74.43</td><td>77.58</td></tr><tr><td>Food101</td><td>90.70</td><td>91.83</td><td>90.82</td><td>92.06</td></tr><tr><td>Aircraft</td><td>33.97</td><td>33.89</td><td>33.87</td><td>35.21</td></tr><tr><td>SUN397</td><td>74.52</td><td>78.98</td><td>74.82</td><td>78.98</td></tr><tr><td>DTD</td><td>63.05</td><td>63.05</td><td>63.82</td><td>63.80</td></tr><tr><td>EuroSAT</td><td>77.73</td><td>79.04</td><td>78.25</td><td>79.89</td></tr><tr><td>UCF101</td><td>77.39</td><td>78.17</td><td>77.55</td><td>78.37</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Average</td><td>74.61</td><td>76.19</td><td>74.83</td><td>76.76</td></tr></table>

## C.3 Full Results on Manual Text Prompts

Table 8 presents the detailed results for each dataset using manual prompts, which are summarized in Table 14. The manual prompts, designed for each dataset as described in (Gao et al., 2023; Zhang et al., 2022), appear to underperform compared to other methods. This suggests that they may not be the optimal choice for every dataset, and that designing these prompts manually is challenging. In contrast, just ensembling multiple manual prompts (Radford et al., 2021) works significantly better, indicating that optimal prompts may exist among these manual options. This finding also implies that utilizing improved manual prompts can substantially enhance performance, potentially replacing shallow prompts. Shallow prompt tuning for the text input yields the best results, demonstrating its effectiveness and flexibility. Therefore, we adopt this approach for our main results.

## C.4 Baseline Results with SGD

Table 9 displays the reproduced results using the SGD optimizer, in contrast to the Adadelta optimizer presented in Table 2. As observed, the results with SGD are slightly lower compared to those with Adadelta. This difference is likely due to the adaptive learning rate of Adadelta, which facilitates training in this unstable few-shot scenario. Nonetheless, even with the SGD optimizer, our method significantly outperforms all baselines, particularly in domains with high RTD, maintaining the same trend observed with the Adadelta optimizer.

![](images/df4a1336c39d256a235a1fddd8c8377c0f7998a33f01da4bc3782e3d05924187.jpg)  
Figure 11: Results for the performance of the low-rank approach with different ranks.

## C.5 Comparison with More Baselines

Due to the page limit, we present a comparison with additional baselines for base-to-novel generalization experiments in Table 10, which are not included in Table 2. These include training with VPT, TPT, and a combination of VPT and TPT. We also compare our method with the recently proposed PromptSRC (Khattak et al., 2023), which employs various regularization techniques such as self-consistency loss and Gaussian averaging. Our method outperforms all these baselines in terms of harmonic mean and demonstrates particularly high performance for novel classes. Compared to

PromptSRC, our method significantly outperforms in novel classes of high RTD domains, such as EuroSAT (+8.39%) and DTD (+4.22%), while maintaining comparable performance in other domains. Notably, our method achieves these results with a simpler training approach, without the need for numerous manual prompts for SRC loss, and with fewer hyperparameters, unlike the many required by PromptSRC’s regularization techniques. Additionally, our method surpasses the simpler baselines of naive training using VPT, TPT, and their combination, highlighting the effectiveness of our configuration design and adaptive ensemble.

## C.6 Ablation on Configuration

To further analyze the optimal configuration in combination with an adaptive ensemble, we conduct additional ablation studies on configurations. The results, present in Table 11, show that utilizing VPT and TA yields the best outcomes, confirming their effectiveness when paired with the adaptive ensemble. However, adding TPT to VPT and TA does not enhance performance, especially in high RTD scenarios, as evidenced by decreased performance in DTD (-4.98%) and EuroSAT (-6.78%) compared to configurations without TPT. While combining TPT with VA demonstrates reasonable performance, it is not as effective as the combination of VPT and TA. This highlights the importance of class separability of visual features achieved through multiple stacks of prompts. Overall, the configuration of APEX outperforms the other setups.

Table 8: Full results on each dataset of Table 14
<table><tr><td></td><td>Average on 11 datasets</td><td colspan="5">ImageNet Caltech101 OxfordPets</td><td colspan="2"> $\mathbf { \Delta _ { C a r s } ^ { S t a n f o r d } }$  Flowers102 Food101</td><td colspan="2"> $\mathbf { \Pi } _ { \mathbf { A i r c r a f t } } ^ { \mathbf { F G V C } }$ </td><td colspan="4">SUN397 DTD EuroSAT UCF101</td></tr><tr><td></td><td>Base</td><td>84.15</td><td>76.64</td><td>98.15</td><td>95.05</td><td>80.75</td><td>97.45</td><td>89.35</td><td></td><td>42.92 81.24</td><td>83.02</td><td>93.93</td><td></td><td>87.10</td></tr><tr><td>Opt. manual prompt (Zhang et al., 2022)</td><td>Novel</td><td>75.24</td><td>69.00</td><td>94.33</td><td>97.04</td><td>75.32</td><td></td><td>77.66</td><td>91.28 90.30</td><td>36.42 39.40</td><td>77.60</td><td>57.59</td><td>71.74</td><td>79.70</td></tr><tr><td></td><td>HM</td><td>79.17</td><td>72.62</td><td>96.20</td><td>96.03</td><td>77.94</td><td>86.44</td><td></td><td></td><td></td><td>79.38</td><td>68.01</td><td>81.35</td><td>83.24</td></tr><tr><td></td><td>Base</td><td>84.02</td><td>76.48</td><td>98.15</td><td>95.09</td><td>80.70</td><td>97.37</td><td></td><td>89.56</td><td>42.56</td><td>81.46</td><td>82.62</td><td>93.01</td><td>87.18</td></tr><tr><td>Ens. (60 manual prompts) (Radford et al., 2021) Novel</td><td></td><td>76.17</td><td>70.24</td><td>93.93</td><td>96.44</td><td>75.88</td><td>77.16</td><td></td><td>91.20</td><td>35.64</td><td>78.36</td><td>59.45</td><td>80.35</td><td>79.21</td></tr><tr><td></td><td>HM</td><td>79.70</td><td>73.23</td><td>95.99</td><td>95.76</td><td>78.22</td><td>86.09</td><td></td><td>90.37</td><td>38.79</td><td>79.88</td><td>69.15</td><td>86.22</td><td>83.00</td></tr><tr><td></td><td>Base</td><td>83.99</td><td>77.12</td><td>98.18</td><td>95.11</td><td>80.53</td><td>97.47</td><td></td><td>89.60</td><td>42.69</td><td>81.17</td><td>82.45</td><td>92.83</td><td>86.74</td></tr><tr><td>Shallow prompt (APEX)</td><td>Novel</td><td>76.76</td><td>71.10</td><td>95.06</td><td>97.27</td><td>75.08</td><td>77.58</td><td></td><td>92.06</td><td>35.21</td><td>78.98</td><td>63.80</td><td>79.89</td><td>78.37</td></tr><tr><td></td><td>HM</td><td>80.04</td><td>73.99</td><td>96.59</td><td>96.18</td><td>77.71</td><td>86.40</td><td></td><td>90.81</td><td>38.59</td><td>80.06</td><td>71.94</td><td>85.88</td><td>82.34</td></tr></table>

## C.7 Ablation on $\beta$

Table 12 presents the results of an ablation study on the hyperparameter $\beta _ { ; }$ , which is used to calculate $\alpha _ { \mathrm { e v a l } }$ . A higher $\beta$ leads to a lower $\alpha _ { \mathrm { e v a l } }$ , indicating greater reliance on the general knowledge of VLMs, which is beneficial for domains with low RTD, and vice versa. As observed, the performance in domains with low RTD, such as Stanford Cars and SUN397, tends to improve with a higher $\beta .$ However, the optimal performance for difficult domains like Aircraft and DTD is achieved with $\beta$ values between 1.0 and 3.0. Not all domains follow this tendency since $\alpha _ { \mathrm { e v a l } }$ is calculated on a class-wise basis, as demonstrated in the case of EuroSAT. Interestingly, except for the value of 2.0, our method demonstrates robustness to variations in $\beta ,$ as it does not significantly affect the average performance. Overall, setting $\beta$ to 4.0 yields the best performance, and therefore, this value has been selected for the final results.

## C.8 Ablation on α

Table 13 presents the comprehensive results of the ablation study on a fixed $\alpha ,$ which is used in Table 1 and Eq. (10). The same α is applied uniformly across all classes and is set as a fixed value for both the visual and text encoders. This is done to determine the correlation between α and the domain, along with its transfer difficulty. Similar to Section C.7, domains with high RTD, such as EuroSAT, require a higher α value to perform well compared to domains with low RTD, like Stanford Cars. These findings support the necessity for an adaptive ensemble that is closely aligned with RTD.

## C.9 Shallow Prompt

Although we observe that TPT leads to overfitting, we employ one-layer learnable text prompts to enhance real-world practicality. Table 14 compares the performance of manually optimized prompts (Gao et al., 2023; Zhang et al., 2022), the ensemble of manual prompts (Radford et al., 2021), and shallow prompts. The shallow prompt method outperforms manual prompts, proving its effectiveness. However, manual prompts, particularly when ensembled, also show comparable performance to shallow prompts, suggesting that well-designed manual prompts can be an effective alternative.

## C.10 Results on Different VLMs

We validate our approach using different backbones: EVA-CLIP (Sun et al., 2023) and CoCa (Yu et al., 2022). Table 15 displays the results using these two backbones, where we compare our method with both zero-shot and naive prompt tuning approaches that combine VPT and TPT. As observed, APEX consistently outperforms the average results in terms of harmonic mean, regardless of the model used. Specifically, with EVA-CLIP, our method demonstrates superior performance for both base and novel classes. In the case of the most challenging domain, EuroSAT, our method significantly enhances performance compared to the zero-shot accuracy for novel classes (+18.46%). A similar improvement of 8.85% on EuroSAT is observed with CoCa. However, in terms of novel classes, the average performance of zero-shot tuning is superior for CoCa. This could be attributed to the larger patch size of this backbone, which might increase the risk of overfitting on the vision side when setting two learnable prompts. Nonetheless, our method shows comparable performance on novel classes to zero-shot CoCa, with a significant improvement in base classes. This results in superior performance in harmonic mean, demonstrating our method’s effectiveness across various VLMs.

Table 9: Comparison of baselines using their own configuration (SGD optimizer) with our method.
<table><tr><td>Dataset</td><td></td><td>CLIP</td><td>CLIP -Adapter</td><td> $\mathbf { \sigma _ { \mathbf { \sigma } } } . \mathbf { c _ { o 0 p } ^ { C _ { 0 } } }$ </td><td>MaPLe</td><td> $\mathbf { . } \mathbf { { \dot { G r a d } } }$ </td><td>APEX</td></tr><tr><td rowspan="3">Average on 11 datasets</td><td>Base</td><td>69.34</td><td>81.81</td><td>80.28</td><td>81.74</td><td>81.78</td><td>84.04</td></tr><tr><td>Novel</td><td>74.22</td><td>71.43</td><td>72.03</td><td>73.89</td><td>69.42</td><td>75.67</td></tr><tr><td>HM</td><td>71.70</td><td>75.93</td><td>75.60</td><td>77.30</td><td>74.80</td><td>79.42</td></tr><tr><td rowspan="3">ImageNet</td><td>Base</td><td>|72.43</td><td>74.40</td><td>75.99</td><td>76.81</td><td>76.93</td><td>76.93</td></tr><tr><td>Novel</td><td>68.14</td><td>68.63</td><td>70.39</td><td>70.66</td><td>69.51</td><td>69.61</td></tr><tr><td>HM</td><td>70.22</td><td>71.40</td><td>73.08</td><td>73.61</td><td>73.03</td><td>73.09</td></tr><tr><td rowspan="3">Caltech101</td><td>Base</td><td>96.84</td><td>97.61</td><td>97.64</td><td>95.61</td><td>95.41</td><td>98.18</td></tr><tr><td>Novel</td><td>94.00</td><td>93.72</td><td>94.52</td><td>94.71</td><td>94.05</td><td>95.02</td></tr><tr><td>HM</td><td>95.40</td><td>95.63</td><td>96.05</td><td>96.18</td><td>95.90</td><td>96.57</td></tr><tr><td rowspan="3">OxfordPets</td><td>Base</td><td>|91.17</td><td>95.06</td><td>95.56</td><td>95.61</td><td>95.41</td><td>95.21</td></tr><tr><td>Novel</td><td>97.26</td><td>95.02</td><td>97.52</td><td>97.63</td><td>90.56</td><td>97.74</td></tr><tr><td>HM</td><td>94.12</td><td>95.04</td><td>96.53</td><td>96.61</td><td>92.92</td><td>96.46</td></tr><tr><td rowspan="3">Stanford Cars</td><td>Base</td><td>|63.37</td><td>76.18</td><td>70.97</td><td>72.49</td><td>77.41</td><td>80.44</td></tr><tr><td>Novel</td><td>74.89</td><td>69.30</td><td>73.44</td><td>73.46</td><td>70.92</td><td>74.76</td></tr><tr><td>HM</td><td>68.65</td><td>72.58</td><td>72.18</td><td>72.97</td><td>74.02</td><td>77.50</td></tr><tr><td rowspan="3">Flowers102</td><td>Base</td><td>|72.08</td><td>96.27</td><td>93.88</td><td>95.49</td><td>95.34</td><td>97.73</td></tr><tr><td>Novel</td><td>77.80</td><td>69.92</td><td>72.56</td><td>72.55</td><td>76.84</td><td>76.67</td></tr><tr><td>HM</td><td>74.83</td><td>81.01</td><td>81.85</td><td>82.45</td><td>85.10</td><td>85.93</td></tr><tr><td rowspan="3">Food101</td><td>Base</td><td>90.10</td><td>90.32</td><td>90.54</td><td>90.50</td><td>90.17</td><td>89.46</td></tr><tr><td>Novel</td><td>91.22</td><td>90.10</td><td>91.15</td><td>91.71</td><td>85.53</td><td>91.94</td></tr><tr><td>HM</td><td>74.83</td><td>90.21</td><td>90.84</td><td>91.10</td><td>87.79</td><td>90.68</td></tr><tr><td rowspan="3">FGVC Aircraft</td><td>Base</td><td>|27.19</td><td>38.87</td><td>33.64</td><td>36.33</td><td>39.01</td><td>42.96</td></tr><tr><td>Novel</td><td>36.29</td><td>31.95</td><td>26.49</td><td>32.64</td><td>27.77</td><td>34.72</td></tr><tr><td>HM</td><td>31.09</td><td>35.07</td><td>29.64</td><td>34.39</td><td>32.44</td><td>38.40</td></tr><tr><td rowspan="3">SUN397</td><td>Base</td><td>69.36</td><td>76.50</td><td>79.86</td><td>80.65</td><td>81.35</td><td>81.18</td></tr><tr><td>Novel</td><td>75.35</td><td>74.60</td><td>76.51</td><td>78.33</td><td>69.06</td><td>77.08</td></tr><tr><td>HM</td><td>72.23</td><td>75.54</td><td>78.15</td><td>79.47</td><td>74.70</td><td>79.08</td></tr><tr><td rowspan="3">DTD</td><td>Base</td><td>|53.24</td><td>80.46</td><td>76.58</td><td>79.20</td><td>77.45</td><td>82.19</td></tr><tr><td>Novel</td><td>59.90</td><td>52.79</td><td>53.47</td><td>55.01</td><td>51.63</td><td>61.21</td></tr><tr><td>HM</td><td>56.37</td><td>63.75</td><td>62.97</td><td>64.92</td><td>61.96</td><td>70.17</td></tr><tr><td rowspan="3">EuroSAT</td><td>Base</td><td>|56.48</td><td>88.48</td><td>86.18</td><td>90.38</td><td>84.88</td><td>93.48</td></tr><tr><td>Novel</td><td>64.05</td><td>67.12</td><td>63.04</td><td>68.43</td><td>56.66</td><td>75.88</td></tr><tr><td>HM</td><td>60.03</td><td>76.33</td><td>72.82</td><td>77.89</td><td>67.96</td><td>83.77</td></tr><tr><td rowspan="3">UCF101</td><td>Base</td><td>70.53</td><td>85.81</td><td>82.22</td><td>84.02</td><td>83.82</td><td>86.71</td></tr><tr><td>Novel</td><td>77.50</td><td>72.55</td><td>73.22</td><td>77.62</td><td>71.13</td><td>77.77</td></tr><tr><td>HM</td><td>73.85</td><td>78.62</td><td>77.46</td><td>80.69</td><td>76.96</td><td>82.00</td></tr></table>

## D Details about Observation

## D.1 Relative Transfer Difficulty

Here, we report the value of RTD which is defined in Sectionn 3 for 11 transfer datasets. We compute the RTD based on the CLIP-B/16 model.

## D.2 Inter- and Intra-class Cosine Similarity

In addition to presenting relative values in Figure 5, we also report the absolute values for both interand intra-class similarities. We observe a significant correlation between the RTD and the ratio of intra- to inter-class similarity.

Table 10: Extended baselines not presented in Table 2 for comparison between base-to-novel experiments with our method.
<table><tr><td>Dataset</td><td></td><td>CLIP VPT</td><td></td><td>TPT</td><td> $\mathbf { \Pi } _ { + } ^ { \mathrm { V P T } }$ </td><td>Prompt</td><td>APEX</td></tr><tr><td rowspan="3">Average on 11 datasets</td><td>Base</td><td>69.34</td><td>81.01</td><td>82.07</td><td>82.93</td><td>84.36</td><td>83.99</td></tr><tr><td>Novel</td><td>74.22</td><td>73.11</td><td>73.90</td><td>74.15</td><td>75.37</td><td>76.76</td></tr><tr><td>HM</td><td>71.70</td><td>76.55</td><td>77.51</td><td>78.00</td><td>79.39</td><td>80.04</td></tr><tr><td rowspan="3">ImageNet</td><td>Base</td><td>72.43</td><td></td><td>75.9476.81</td><td>77.18</td><td>77.90</td><td>77.12</td></tr><tr><td>Novel</td><td>68.14</td><td>68.74</td><td>69.45</td><td>69.86</td><td>70.26</td><td>71.10</td></tr><tr><td>HM</td><td>70.22</td><td>72.16</td><td>72.94</td><td>73.34</td><td>73.88</td><td>73.99</td></tr><tr><td rowspan="3">Caltech101</td><td>Base</td><td>96.84</td><td>97.79</td><td>97.84</td><td>97.98</td><td>97.81</td><td>98.18</td></tr><tr><td>Novel</td><td>94.00</td><td>93.65</td><td>94.29</td><td>94.38</td><td>93.88</td><td>95.06</td></tr><tr><td>HM</td><td>95.40</td><td>95.68</td><td>96.03</td><td>96.15</td><td>95.80</td><td>96.59</td></tr><tr><td rowspan="3">OxfordPets</td><td>Base</td><td>91.17</td><td>95.11</td><td>95.48</td><td>95.78</td><td>95.69</td><td>95.11</td></tr><tr><td>Novel</td><td>97.26</td><td>96.57</td><td>97.52</td><td>97.65</td><td>97.42</td><td>97.27</td></tr><tr><td>HM</td><td>94.12</td><td>95.83</td><td>96.49</td><td>96.71</td><td>96.55</td><td>96.18</td></tr><tr><td rowspan="3">Stanford Cars</td><td>Base</td><td>63.37</td><td></td><td>70.72 75.18</td><td>75.75</td><td>80.16</td><td>80.53</td></tr><tr><td>Novel</td><td>74.89</td><td>72.78</td><td>72.73</td><td>73.02</td><td>74.52</td><td>75.08</td></tr><tr><td>HM</td><td>68.65</td><td>71.7473.93</td><td></td><td>74.36</td><td>77.24</td><td>77.71</td></tr><tr><td rowspan="3">Flowers102</td><td>Base</td><td>72.08</td><td>91.60</td><td>96.45</td><td>96.26</td><td>96.96</td><td>97.47</td></tr><tr><td>Novel</td><td>77.80</td><td>69.62 74.69</td><td></td><td>72.62</td><td>76.73</td><td>77.58</td></tr><tr><td>HM</td><td>74.83</td><td>79.11</td><td>84.19</td><td>82.79</td><td>85.67</td><td>86.40</td></tr><tr><td rowspan="3">Food101</td><td>Base</td><td>90.10</td><td>90.17</td><td>90.30</td><td>90.36</td><td>90.60</td><td>89.60</td></tr><tr><td>Novel</td><td>91.22</td><td>90.94</td><td>91.42</td><td>91.58</td><td>91.38</td><td>92.06</td></tr><tr><td>HM</td><td>90.66</td><td>90.55</td><td>90.86</td><td>90.97</td><td>90.99</td><td>90.81</td></tr><tr><td rowspan="3">FGVC Aircraft</td><td>Base</td><td>27.19</td><td></td><td>34.7037.86</td><td>38.76</td><td>43.67</td><td>42.69</td></tr><tr><td>Novel</td><td>36.29</td><td>33.53 34.17</td><td></td><td>35.08</td><td>36.42</td><td>35.21</td></tr><tr><td>HM</td><td>31.09</td><td>34.1035.92</td><td></td><td>36.83</td><td>39.72</td><td>38.59</td></tr><tr><td rowspan="3">SUN397</td><td>Base</td><td>69.36</td><td>79.09</td><td>81.70</td><td>81.57</td><td>82.94</td><td>81.17</td></tr><tr><td>Novel</td><td>75.35</td><td>76.85</td><td>77.62</td><td>77.92</td><td>78.37</td><td>78.98</td></tr><tr><td>HM</td><td>72.23</td><td>77.95</td><td>79.61</td><td>79.70</td><td>80.59</td><td>80.06</td></tr><tr><td rowspan="3">DTD</td><td>Base</td><td>53.24</td><td>78.67</td><td>79.81</td><td>80.81</td><td>82.21</td><td>82.45</td></tr><tr><td>Novel</td><td>59.90</td><td>53.78</td><td>55.32</td><td>55.64</td><td>59.58</td><td>63.80</td></tr><tr><td>HM</td><td>56.37</td><td>63.89</td><td>65.35</td><td>65.90</td><td>69.09</td><td>71.94</td></tr><tr><td rowspan="3">EuroSAT</td><td>Base</td><td>56.48</td><td>94.17</td><td>86.98</td><td>92.91</td><td>93.06</td><td>92.83</td></tr><tr><td>Novel</td><td>64.05</td><td>73.26</td><td>69.16</td><td>71.19</td><td>71.60</td><td>79.89</td></tr><tr><td>HM</td><td>60.03</td><td>82.4177.05</td><td></td><td>80.61</td><td>80.93</td><td>85.88</td></tr><tr><td rowspan="3">UCF101</td><td>Base</td><td>70.53</td><td>83.1084.38</td><td></td><td>84.92</td><td>87.05</td><td>86.74</td></tr><tr><td>Novel</td><td>77.50</td><td>74.52</td><td>76.54</td><td>76.75</td><td>78.96</td><td>78.37</td></tr><tr><td>HM</td><td>73.85</td><td>78.5880.27</td><td></td><td>80.63</td><td>82.81</td><td>82.34</td></tr></table>

## D.3 Results on 6 datasets

We also present extended results in Figure 12, which include data from three additional datasets: ImageNet, SUN397, and DTD. For ImageNet and SUN397, which already exhibit high class separability, we note that all methods—TPT, VPT, and their combination—yield similar performance differences. However, the results for DTD indicate a tendency for TPT to overfit to the base classes. This observation is consistent with the findings presented in Figure 2.

## E More Related Work

Vision-Language Models VLMs overcome the limitations of vision-only supervised learning with their robustness and flexibility in zero-shot inference through natural language supervision. CLIP (Radford et al., 2021) facilitates this by adopting contrastive learning with a large-scale dataset of 400 million images. ALIGN(Jia et al., 2021) further improves upon this by scaling up the dataset with more noisy image-text pairs. FILIP (Yao et al., 2022) enables finer-grained alignment between two modalities and GLIP (Li et al., 2022b) improves visual grounding and object detection using VLMs. CoCa (Yu et al., 2022) employs both captioning and contrastive losses, thereby integrating the model capabilities of contrastive approaches like CLIP with those of generative methods. CyCLIP (Goel et al., 2022) employs cyclic loss to ensure geometric consistency, while FLIP (Li et al., 2023) enhances VLMs through masking techniques. EVA-CLIP (Sun et al., 2023) implements various training techniques, such as different attention mechanisms and optimizers, to further improve CLIP’s performance. Additionally, SigLIP (Zhai et al., 2023) replaces the softmax loss with sigmoid loss, enabling more efficient pretraining with smaller batch sizes.

Table 11: Results for additional ablation study on configurations when combined with adaptive ensemble.
<table><tr><td>|Average on</td><td>11 datasets</td><td colspan="3">ImageNet Caltech101 OxfordPets</td><td colspan="3">Stanford Cars</td><td colspan="2">Flowers102 Food101 FGVC Aircraft</td><td colspan="2"></td><td>SUN397 DTD EuroSAT UCF101</td><td></td></tr><tr><td></td><td>Base</td><td>83.51</td><td>76.43</td><td>98.00</td><td>94.76</td><td>79.68</td><td>97.28</td><td>89.24</td><td>42.27</td><td>80.96</td><td>81.49</td><td>92.27</td><td>86.24</td></tr><tr><td>TPT + VA</td><td>Novel</td><td>75.88</td><td>69.43</td><td>94.49</td><td>97.21</td><td>75.77</td><td>77.50</td><td>91.50</td><td>34.85</td><td>78.20</td><td>62.05</td><td>76.77</td><td>76.90</td></tr><tr><td></td><td>HM</td><td>79.32</td><td>72.76</td><td>96.21</td><td>95.97</td><td>77.68</td><td>86.27</td><td>90.36</td><td>38.20</td><td>79.56</td><td>70.45</td><td>83.81</td><td>81.30</td></tr><tr><td></td><td>Base</td><td>83.56</td><td>76.93</td><td>98.03</td><td>94.77</td><td>79.45</td><td>97.51</td><td>89.26</td><td>42.14</td><td>81.02</td><td>81.72</td><td>92.11</td><td>86.21</td></tr><tr><td>VPT + TA + TPT Novel</td><td></td><td>75.09</td><td>71.30</td><td>94.72</td><td>97.76</td><td>72.98</td><td>76.70</td><td>91.94</td><td>33.80</td><td>78.08</td><td>58.82</td><td>73.11</td><td>76.80</td></tr><tr><td></td><td>HM</td><td>78.85</td><td>74.01</td><td>96.35</td><td>96.24</td><td>76.08</td><td>85.86</td><td>90.58</td><td>37.51</td><td>79.52</td><td>68.40</td><td>81.52</td><td>81.23</td></tr><tr><td></td><td>Base</td><td>83.99</td><td>77.12</td><td>98.18</td><td>95.11</td><td>80.53</td><td>97.47</td><td>89.60</td><td>42.69</td><td>81.17</td><td>82.45</td><td>92.83</td><td>86.74</td></tr><tr><td>VPT + TA (APEX) Novel</td><td></td><td>76.76</td><td>71.10</td><td>95.06</td><td>97.27</td><td>75.08</td><td>77.58</td><td>92.06</td><td>35.21</td><td>78.98</td><td>63.80</td><td>79.89</td><td>78.37</td></tr><tr><td></td><td>HM</td><td>80.04</td><td>73.99</td><td>96.59</td><td>96.18</td><td>77.71</td><td>86.40</td><td>90.81</td><td>38.59</td><td>80.06</td><td>71.94</td><td>85.88</td><td>82.34</td></tr></table>

Table 12: Results for additional ablation study on scaling factor β. Our proposed methods shows robust performance on the selection of $\beta .$
<table><tr><td rowspan="2">β</td><td rowspan="2">|Average on 11 datasets</td><td rowspan="2">ImageNet Caltech101 OxfordPets</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">Stanford Cars</td><td rowspan="2">Flowers102 Food101</td><td rowspan="2"></td><td rowspan="2"> $\begin{array} { l } { \mathbf { F G V C } } \\ { \mathbf { A i r c r a f t } } \end{array}$ </td><td rowspan="2">SUN397 DTD EuroSAT UCF101</td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td></td></tr><tr><td>1.0</td><td>75.97</td><td>70.62</td><td>95.15</td><td>97.43</td><td>72.15</td><td>75.95</td><td>91.38</td><td>35.07</td><td>77.02</td><td>63.90</td><td>78.36</td><td>78.66</td></tr><tr><td>2.0</td><td>76.51</td><td>71.06</td><td>95.14</td><td>97.44</td><td>73.95</td><td>77.06</td><td>91.70</td><td>35.35</td><td>78.12</td><td>63.99</td><td>78.89</td><td>78.92</td></tr><tr><td>3.0</td><td>76.75</td><td>71.18</td><td>95.15</td><td>97.37</td><td>74.69</td><td>77.61</td><td>91.92</td><td>35.46</td><td>78.66</td><td>64.17</td><td>79.35</td><td>78.64</td></tr><tr><td>4.0 (APEX)</td><td>76.76</td><td>71.10</td><td>95.06</td><td>97.27</td><td>75.08</td><td>77.58</td><td>92.06</td><td>35.21</td><td>78.98</td><td>63.80</td><td>79.89</td><td>78.37</td></tr><tr><td>5.0</td><td>76.72</td><td>71.00</td><td>95.16</td><td>97.18</td><td>75.10</td><td>77.79</td><td>91.96</td><td>35.05</td><td>78.96</td><td>63.77</td><td>79.88</td><td>78.07</td></tr><tr><td>6.0</td><td>76.66</td><td>70.96</td><td>95.16</td><td>97.15</td><td>75.17</td><td>77.80</td><td>91.98</td><td>34.84</td><td>78.92</td><td>63.54</td><td>80.01</td><td>77.75</td></tr></table>

![](images/d96da47b5ae0a84668eda54116ad0e538bfcb7b0fa92cab72141bdce2d1b882e.jpg)  
Figure 12: Extended results for Figure 2. All results in different datasets show similar trends that indicate VPT yields a smaller discrepancy in performance between base and novel categories, suggesting a reduced risk of overfitting compared to TPT.

There is also a line of research focused on encoder-decoder or decoder-only architectures. BLIP (Li et al., 2022a) facilitates both encoding and decoding by training with three objective functions, utilizing synthetic data and data filtering. ALBEF (Li et al., 2021) employs a strategy of alignment before applying cross-attention, combined with a momentum update. Flamingo (Alayrac et al., 2022) enables few-shot inference in visionlanguage tasks through architectural innovations, using vision-language prompts.

Prompt Tuning Efficient tuning using soft prompts, originating in the domain of natural language processing, has gained a lot of attention (Lester et al., 2021). This approach has also been applied in the vision-language domain to adapt to downstream tasks. CoOp (Zhou et al., 2022b) was the first to apply learnable prompts for CLIP model, replacing manual prompts for each domain. ProDA (Lu et al., 2022) observes that these text prompts can be viewed as a distribution and proposes prompt distributional learning for higher quality results. CoCoOp (Zhou et al., 2022a) conditions text prompts on images to prevent overfitting to base classes. KgCoOp (Yao et al., 2023) regularizes by minimizing the discrepancy between learned and manual prompts. UPT (Zang et al., 2022) examines both VPT (Jia et al., 2022) and text prompts, proposing a unified approach to generate visual and textual prompts from the same architecture. MaPLe (khattak et al., 2023) employs the alignment of visual and text prompts for improvement with deep prompts, while DCP (Liu et al., 2023) uses an attention mechanism for this alignment. There is also a line of research aimed at preventing the forgetting of general knowledge. ProGrad (Zhu et al., 2023a) aligns gradient directions to preserve general knowledge, and Prompt-SRC (Khattak et al., 2023) utilizes multiple regularization losses with Gaussian aggregation of model weights to prevent forgetting.

Table 13: Extended results for ablation study on hyperparamter α related to Table 1.
<table><tr><td>α</td><td>Average on ImageNet Caltech101 OxfordPets 11 datasets</td><td></td><td></td><td></td><td>Stanford Cars</td><td>Flowers102 Food101</td><td></td><td>FGVC Aircraft</td><td></td><td></td><td></td><td>SUN397 DTD EuroSAT UCF101</td></tr><tr><td>0.0</td><td>75.38</td><td>70.80</td><td>95.13</td><td>97.03</td><td>75.19</td><td>77.87</td><td>91.94</td><td>33.57</td><td>78.32</td><td>61.68</td><td>70.90</td><td>76.80</td></tr><tr><td>0.1</td><td>75.86</td><td>71.06</td><td>95.19</td><td>97.19</td><td>75.17</td><td>77.67</td><td>92.10</td><td>34.34</td><td>78.82</td><td>62.68</td><td>72.74</td><td>77.52</td></tr><tr><td>0.2</td><td>76.10</td><td>71.20</td><td>95.14</td><td>97.29</td><td>75.04</td><td>77.52</td><td>91.96</td><td>34.75</td><td>78.80</td><td>63.18</td><td>74.10</td><td>78.08</td></tr><tr><td>0.3</td><td>76.27</td><td>71.20</td><td>95.09</td><td>97.39</td><td>74.67</td><td>77.33</td><td>91.92</td><td>35.16</td><td>78.90</td><td>63.74</td><td>75.08</td><td>78.54</td></tr><tr><td>0.4</td><td>76.34</td><td>71.18</td><td>95.14</td><td>97.47</td><td>74.22</td><td>76.96</td><td>91.88</td><td>35.34</td><td>78.68</td><td>64.16</td><td>75.87</td><td>78.84</td></tr><tr><td>0.5</td><td>76.29</td><td>71.04</td><td>95.15</td><td>97.50</td><td>73.59</td><td>76.56</td><td>91.78</td><td>35.45</td><td>78.40</td><td>64.32</td><td>76.41</td><td>79.01</td></tr><tr><td>0.6</td><td>76.13</td><td>70.82</td><td>95.14</td><td>97.47</td><td>72.74</td><td>76.13</td><td>91.64</td><td>35.33</td><td>78.00</td><td>64.30</td><td>76.95</td><td>78.96</td></tr><tr><td>0.7</td><td>75.88</td><td>70.46</td><td>95.17</td><td>97.39</td><td>71.82</td><td>75.66</td><td>91.44</td><td>35.25</td><td>77.38</td><td>64.23</td><td>77.10</td><td>78.79</td></tr><tr><td>0.8</td><td>75.54</td><td>70.06</td><td>95.07</td><td>97.36</td><td>70.85</td><td>75.09</td><td>91.22</td><td>34.93</td><td>76.56</td><td>64.04</td><td>77.33</td><td>78.39</td></tr><tr><td>0.9</td><td>75.10</td><td>69.62</td><td>95.01</td><td>97.31</td><td>69.63</td><td>74.49</td><td>90.98</td><td>34.53</td><td>75.68</td><td>63.53</td><td>77.44</td><td>77.92</td></tr><tr><td>1.0</td><td>74.61</td><td>69.08</td><td>94.91</td><td>97.24</td><td>68.40</td><td>73.71</td><td>90.70</td><td>33.97</td><td>74.52</td><td>63.05</td><td>77.73</td><td>77.39</td></tr></table>

Table 14: Comparison of the accuracy of the base, novel, and their harmonic means among the various prompt types on text encoder.
<table><tr><td>Prompt</td><td>Base Acc.</td><td>Novel Acc.</td><td>HM</td></tr><tr><td>Opt. manual prompt (Zhang et al., 2022)</td><td>84.15</td><td>75.24</td><td>79.17</td></tr><tr><td>Ens. (60 manual prompts (Radford et al., 2021))</td><td>84.02</td><td>76.17</td><td>79.70</td></tr><tr><td>Shallow prompt</td><td>83.99</td><td>76.76</td><td>80.04</td></tr></table>

Adapter-style Tuning Adapter-style tuning has been extensively explored as an alternative to prompt tuning. CLIP-Adapter (Gao et al., 2023) was the first proposed method in this area, utilizing a two-layer MLP structure with ReLU nonlinearity in between. Additionally, it incorporates a residual connection to preserve general knowledge. For improved efficiency, Tip-Adapter (Zhang et al., 2022) employs a cache-based model to save the features and labels of few-shot samples, using them to predict test outcomes without further training. This approach also facilitates better fine-tuning by using the cache as initial training points for further refinement. Differently, Task Residual (Yu et al., 2023) adopts a unique strategy by simply adding a residual or bias term vector for each class, reducing reliance on pre-trained features. Zhu et al. (2023b) enhances cache-based models through prior refinement, which involves selecting important features for the cache-based model.

Table 15: Accuracy on base-to-novel generalization of APEX on EVA-CLIP (Sun et al., 2023) and CoCa (Yu et al., 2022).
<table><tr><td>Model</td><td></td><td colspan="3">EVA-CLIP-B/16</td><td colspan="3">CoCa-B/32</td></tr><tr><td colspan="2">Dataset</td><td>ZS</td><td colspan="2">+TPT APEX</td><td>ZS</td><td colspan="2">+TVPT</td><td>APEX</td></tr><tr><td rowspan="3">Average on 11</td><td>Base</td><td>75.28</td><td>85.91</td><td>85.93</td><td>70.85</td><td>82.39</td><td>82.09</td><td></td></tr><tr><td>Novel</td><td>77.68</td><td>75.24</td><td>79.34</td><td>74.29</td><td></td><td>71.05</td><td>73.98</td></tr><tr><td>HM</td><td>76.46</td><td>80.22</td><td>82.50</td><td>72.53</td><td></td><td>76.30</td><td>77.87</td></tr><tr><td rowspan="3">ImageNet</td><td>Base</td><td>79.20</td><td>81.78</td><td>81.26</td><td>|67.10</td><td>69.50</td><td>69.46</td><td></td></tr><tr><td>Novel</td><td>75.60</td><td>72.28</td><td>75.83</td><td>66.60</td><td>62.33</td><td></td><td>66.46</td></tr><tr><td>HM</td><td>77.36</td><td>76.74</td><td>78.45</td><td>66.85</td><td></td><td>65.72</td><td>67.90</td></tr><tr><td rowspan="3">Caltech101</td><td>Base</td><td>98.60</td><td>98.87</td><td>98.82</td><td>96.70</td><td>97.86</td><td>98.04</td><td></td></tr><tr><td>Novel</td><td>97.30</td><td>95.05</td><td>97.22</td><td>96.30</td><td>94.12</td><td></td><td>95.98</td></tr><tr><td>HM</td><td>97.95</td><td>96.92</td><td>98.01</td><td>96.50</td><td>95.95</td><td></td><td>97.00</td></tr><tr><td rowspan="3"></td><td>Base</td><td>94.90</td><td>95.52</td><td>95.27</td><td>92.30</td><td>91.83</td><td></td><td>92.44</td></tr><tr><td>Novel</td><td>98.10</td><td>98.34</td><td>97.97</td><td>96.20</td><td></td><td>95.07</td><td>93.54</td></tr><tr><td>HM</td><td>96.47</td><td>96.91</td><td>96.60</td><td>94.21</td><td>93.42</td><td></td><td>92.99</td></tr><tr><td rowspan="3">Stanford Cars</td><td>Base</td><td>76.90</td><td>85.76</td><td>86.16</td><td>84.00</td><td>88.94</td><td></td><td>88.87</td></tr><tr><td>Novel</td><td>87.10</td><td>82.49</td><td>86.75</td><td>93.00</td><td></td><td>90.73</td><td>92.57</td></tr><tr><td>HM</td><td>81.68</td><td>84.09</td><td>86.45</td><td>88.27</td><td>89.83</td><td></td><td>90.68</td></tr><tr><td rowspan="3">Flowers102</td><td>Base</td><td>74.20</td><td>99.41</td><td>99.50</td><td>|69.10</td><td>96.33</td><td></td><td>96.83</td></tr><tr><td>Novel</td><td>81.10</td><td>77.32</td><td>79.94</td><td>74.70</td><td></td><td>65.61</td><td>70.09</td></tr><tr><td>HM</td><td>77.50</td><td>86.98</td><td>88.65</td><td>71.79</td><td>78.06</td><td></td><td>81.32</td></tr><tr><td rowspan="3">Food101</td><td>Base</td><td>90.30</td><td>90.34</td><td>90.24</td><td>|81.20</td><td>79.87</td><td></td><td>80.80</td></tr><tr><td>Novel</td><td>91.90</td><td>90.11</td><td>91.76</td><td>82.90</td><td>79.30</td><td></td><td>82.66</td></tr><tr><td>HM</td><td>91.09</td><td>90.22</td><td>90.99</td><td>82.04</td><td>79.58</td><td></td><td>81.72</td></tr><tr><td rowspan="3">FGVC Aircraft</td><td>Base</td><td>28.70</td><td>45.52</td><td>46.01</td><td>|21.40</td><td>40.71</td><td></td><td>39.81</td></tr><tr><td>Novel</td><td>32.50</td><td>26.75</td><td>32.12</td><td>25.50</td><td></td><td>22.04</td><td>25.22</td></tr><tr><td>HM</td><td>30.48</td><td>33.70</td><td>37.83</td><td>23.27</td><td>28.60</td><td></td><td>30.88</td></tr><tr><td rowspan="3">SUN397</td><td>Base</td><td>76.70</td><td>83.10</td><td>82.44</td><td>73.70</td><td>78.68</td><td></td><td>77.68</td></tr><tr><td>Novel</td><td>80.80</td><td>76.76</td><td>80.54</td><td>77.40</td><td></td><td>73.50</td><td>77.12</td></tr><tr><td>HM</td><td>78.70</td><td>79.80</td><td>81.48</td><td>75.50</td><td>76.00</td><td></td><td>77.40</td></tr><tr><td rowspan="3">DTD</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>83.25</td></tr><tr><td>Base Novel</td><td>62.80 63.90</td><td>83.78 61.32</td><td>84.15 64.39</td><td>|62.60 61.10</td><td></td><td>83.04 58.46</td><td>61.14</td></tr><tr><td>HM</td><td>63.35</td><td>70.81</td><td>72.96</td><td>61.84</td><td>68.62</td><td></td><td>70.50</td></tr><tr><td rowspan="3">EuroSAT</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>93.87</td></tr><tr><td>Base Novel</td><td>72.30 68.30</td><td>95.32 73.74</td><td>94.81 86.76</td><td>62.80 71.50</td><td>96.42</td><td>73.90</td><td>80.35</td></tr><tr><td>HM</td><td>70.24</td><td>83.15</td><td>90.61</td><td>66.87</td><td>83.67</td><td></td><td>86.59</td></tr><tr><td rowspan="3"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>73.50</td><td>85.58</td><td>86.58</td><td>|68.50</td><td></td><td>83.13</td><td>82.01</td></tr><tr><td>Novel HM</td><td>77.90 75.64</td><td>73.43 79.04</td><td>79.49 82.88</td><td>72.00 70.21</td><td>66.54 73.92</td><td>69.69 74.76</td><td></td></tr></table>

Table 16: The relative transfer difficulty values for all datasets by using Definition 1.
<table><tr><td>Dataset | ImageNet</td><td></td><td>Caltech</td><td>Pets</td><td>Cars</td></tr><tr><td>RTD</td><td> $1 . 4 \times 1 0 ^ { - 3 }$ </td><td> $1 . 0 8 \times 1 0 ^ { - 2 }$ </td><td> $3 . 0 1 \times 1 0 ^ { - 2 }$ </td><td> $7 . 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Dataset</td><td>Flowers</td><td>Food</td><td>Aircraft</td><td>SUN</td></tr><tr><td>RTD</td><td> $1 . 5 2 \times 1 0 ^ { - 2 }$ </td><td> $1 . 1 5 \times 1 0 ^ { - 2 }$ </td><td> $4 . 0 7 \times 1 0 ^ { - 2 }$ </td><td> $3 . 8 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Dataset</td><td>DTD</td><td>EuroSAT</td><td>UCF</td><td></td></tr><tr><td>RTD</td><td> $4 . 9 5 \times 1 0 ^ { - 3 }$ </td><td> $1 . 8 4 \times 1 0 ^ { - 1 }$ </td><td> $1 . 4 2 \times 1 0 ^ { - 2 }$ </td><td></td></tr></table>

Algorithm 2 Pseudo-Algorithm for Adaptive Infer   
ence of APEX   
Require: Pretrained visual encoder , Pretrained   
text encoder , Learned vision prompts $\hat { \mathbf { P } } ,$   
Learned shallow text prompts $\mathbf { P _ { 0 } }$ ,Learned   
adapter parameterized by matrix A and b, The   
C classes for base category $\{ 1 , \ldots , C \}$ , The   
$C _ { \mathrm { e v a l } }$ candidate classes for evaluation $\{ C +$   
$1 , \ldots , C + C _ { \mathrm { e v a l } } \}$   
Require: Initial Trained Text Embeddings   
$\{ \mathbf { W } _ { 0 , j } \} _ { j = 1 } ^ { C }$ , Initial Evaluation Text Embed   
ding $\begin{array} { r } { \{ \dot { \mathbf { W } } _ { \mathrm { 0 , e v a l } } \} _ { \mathrm { e v a l } = C + 1 } ^ { C + C _ { \mathrm { e v a l } } } , } \end{array}$ Evaluation Image   
I   
1: $\{ \mathbf { t } ^ { \prime } { } _ { j } \} _ { j = 1 } ^ { C } = \{ \mathcal { T } ( \mathbf { W } _ { 0 , j } ) \} _ { j = 1 } ^ { C }$   
2: for $\mathrm { e v a l } = C + 1 , \ldots , \bar { C } + C _ { \mathrm { e v a l } }$ do   
3: $\mathbf { t } _ { \mathrm { - e v a l } } ^ { \prime } = \mathcal { T } ( \mathbf { W } _ { \mathrm { 0 , e v a l } } )$   
4: $\tilde { \mathbf { t } } _ { \mathrm { e v a l } } = \mathcal { T } ( [ \mathbf { W } _ { \mathrm { 0 , e v a l } } , \mathbf { P } _ { \mathbf { 0 } } ] )$   
5: $\begin{array} { r } { d _ { \mathrm { e v a l } } ^ { \mathrm { a v g } } = 1 . 0 - \frac { 1 } { C } \sum _ { j = 1 } ^ { C } \mathrm { s i m } ( \mathbf { t } _ { \mathrm { e v a l } } ^ { \prime } , \mathbf { t } _ { j } ^ { \prime } ) } \end{array}$   
6: $\begin{array} { r } { d _ { \mathrm { e v a l } } ^ { \mathrm { n n } } = 1 . 0 - \operatorname* { m i n } _ { \forall j \in \{ 1 , \dots , C \} } \mathrm { s i m } ( \mathbf { t } _ { \mathrm { ~ e v a l } } ^ { \prime } , \mathbf { t } _ { j } ^ { \prime } ) } \end{array}$   
7: $\alpha _ { \mathrm { e v a l } } = \exp \left( - \beta \cdot ( d _ { \mathrm { e v a l } } ^ { \mathrm { a v g } } ) \cdot { \bf 1 } _ { ( d _ { \mathrm { e v a l } } ^ { \mathrm { n n } } > \epsilon ) } \right)$   
8: $\mathbf { t } _ { \mathrm { e v a l } } = \alpha _ { \mathrm { e v a l } } { \cdot } ( \mathbf { A } ^ { \top } \mathbf { \tilde { t } } _ { \mathrm { e v a l } } { + } \mathbf { b } ) + ( 1 - \alpha _ { \mathrm { e v a l } } ) { \cdot } \mathbf { \tilde { t } } _ { \mathrm { e v a l } }$   
9: end for   
10: $\mathbf { E } _ { 0 } = \mathtt { P a t h E m b e d d i n g } ( I )$   
11: $\mathbf { c } ^ { \prime } _ { L \nu } = \mathcal { V } ( [ \mathbf { c } _ { 0 } , \mathbf { E _ { 0 } } ] )$   
12: $\mathbf { z } ^ { \prime }  \mathtt { I m a g e P r o j } ( \mathbf { c } ^ { \prime } { } _ { L \nu } )$   
13: for $i = 1 , \ldots , J _ { \mathcal { V } } \mathbf { d o }$   
14: $[ \mathbf { c } _ { i } , \mathbf { E } _ { i } , \_ ] \longleftarrow \mathcal { V } _ { i } ( [ \mathbf { c } _ { i - 1 } , \mathbf { E } _ { i - 1 } , \hat { \mathbf { P } } _ { i - 1 } ] )$   
15: end for   
16: for $i = J _ { \mathcal { V } } + 1 , \dotsc , L _ { \mathcal { V } }$ do   
17: $[ \mathbf { c } _ { i } , \mathbf { E } _ { i } , \hat { \mathbf { P } } _ { i } ] \gets \mathcal { V } _ { i } ( [ \mathbf { c } _ { i - 1 } , \mathbf { E } _ { i - 1 } , \hat { \mathbf { P } } _ { i - 1 } ] )$   
18: end for   
19: $\mathbf { z } \gets \mathtt { I m a g e P r o j } ( \mathbf { c } _ { L _ { \nu } } )$   
20: $\begin{array} { r } { \bar { \alpha } = \frac { 1 } { C _ { \mathrm { e v a l } } } \sum _ { \mathrm { e v a l } = C + 1 } ^ { C + C _ { \mathrm { e v a l } } } { \alpha _ { \mathrm { e v a l } } } } \end{array}$   
21: $\mathbf { z } = { \bar { \alpha } } \cdot \mathbf { \bar { z } } ^ { \prime } + ( 1 - { \bar { \alpha } } ) \cdot \mathbf { z }$   
22: /\* Calculate the probability for class $i ^ { * } /$   
23: Calculate $\begin{array} { r l r l r l } { \mathrm { P r } ( y } & { { } } & { = } & { { } } & { i | { \bf z } , { \bf t } ) } & { { } } & { = } & { { } } \end{array}$   
exp(sim(z,t<sub>i</sub>)/τ)   
$\overline { { \sum _ { j = C + 1 } ^ { C + C _ { \mathrm { e v a l } } } \exp ( \sin ( \mathbf { z } , \mathbf { t } _ { j } ) / \tau ) } }$   
24: Predict as arg $\begin{array} { r } { \operatorname* { m a x } _ { i \in \{ C + 1 , \dots , C + C _ { \mathrm { e v a l } } \} } \operatorname* { P r } ( y \ = } \end{array}$   
$i | \mathbf { z } , \mathbf { t } )$

Table 17: The averaged cosine similarity value for interand intra-class and their relative ratio.
<table><tr><td>Dataset</td><td>ImageNet</td><td>Caltech</td><td>Pets</td><td>Cars</td><td>Flowers</td><td>Food</td></tr><tr><td>Inter</td><td>0.551</td><td>0.672</td><td>0.844</td><td>0.564</td><td>0.749</td><td>0.754</td></tr><tr><td>Intra</td><td>0.925</td><td>0.898</td><td>0.910</td><td>0.829</td><td>0.924</td><td>0.853</td></tr><tr><td>Ratio</td><td>1.680</td><td>1.336</td><td>1.078</td><td>1.470</td><td>1.234</td><td>1.131</td></tr><tr><td>Dataset</td><td>Aircraft</td><td>SUN</td><td>DTD</td><td>EuroSAT</td><td>UCF</td><td></td></tr><tr><td>Inter</td><td>0.746</td><td>0.487</td><td>0.803</td><td>0.896</td><td>0.673</td><td></td></tr><tr><td>Intra</td><td>0.858</td><td>0.780</td><td>0.823</td><td>0.934</td><td>0.866</td><td></td></tr><tr><td>Ratio</td><td>1.150</td><td>1.602</td><td>1.025</td><td>1.042</td><td>1.287</td><td></td></tr></table>