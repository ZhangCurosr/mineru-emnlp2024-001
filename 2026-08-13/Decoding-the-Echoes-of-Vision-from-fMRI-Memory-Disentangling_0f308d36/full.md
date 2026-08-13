# Decoding the Echoes of Vision from fMRI: Memory Disentangling for Past Semantic Information

Runze Xia, Congchi Yin, Piji Li∗

<sup>1</sup> College of Computer Science and Technology,

Nanjing University of Aeronautics and Astronautics, China

<sup>2</sup> MIIT Key Laboratory of Pattern Analysis and Machine Intelligence, Nanjing, China {xiarunze,congchiyin,pjli}@nuaa.edu.cn

## Abstract

The human visual system is capable of processing continuous streams of visual information, but how the brain encodes and retrieves recent visual memories during continuous visual processing remains unexplored. This study investigates the capacity of working memory to retain past information under continuous visual stimuli. And then we propose a new task Memory Disentangling, which aims to extract and decode past information from fMRI signals. To address the issue of interference from past memory information, we design a disentangled contrastive learning method inspired by the phenomenon of proactive interference. This method separates the information between adjacent fMRI signals into current and past components and decodes them into image descriptions. Experimental results demonstrate that this method effectively disentangles the information within fMRI signals. This research could advance brain-computer interfaces and mitigate the problem of low temporal resolution in fMRI. <sup>1</sup>

## 1 Introduction

The human visual system is highly intricate and plays a foundamental role in daily lives (Loomis et al., 2018). Exploring and comprehending this system is a key objective for researchers in the fields of neuroscience and artificial intelligence (Clark, 2013; Herreras, 2010). One particularly intriguing question pertains how the brain processes and retrieves recent visual memories, which holds significant implications for Brain-Computer Interfaces (BCIs) and cognitive neuroscience (Logothetis, 2008; Ranganath and D’Esposito, 2005; Marr, 2010).

In recent years, functional magnetic resonance imaging (fMRI), a revolutionary non-invasive neuroimaging technique (Ogawa et al., 1990), has become indispensable for studying brain function and cognitive processes by detecting blood flow changes associated with neural activity through blood-oxygen-level-dependent (BOLD) contrast (Bandettini et al., 1992). With its high spatial resolution, fMRI rapidly advances neuroscience researches (Glover, 2011). Therefore, fMRI provides our research with a unique perspective to explore the relationship between brain activity and memory functions.

![](images/3e7b47819bd19261a88888d134f9dd41d9f0d13922b477508de4e78f354ac01f.jpg)  
Figure 1: The schematic diagram of Memory Disentangling based on decoding image semantic information.

While significant progress (Xia et al., 2024; Ozcelik et al., 2022; Takagi and Nishimoto, 2023) has been made in fMRI studies involving static visual stimuli, research on continuous visual stimuli remains largely unexplored. In real-world scenarios, visual experiences are rarely isolated and static. Instead, our brain continuously processes streams of visual information, necessitating the tracking of scene changes and retention of critical visual details to support decision-making (Yin et al., 2020). Although studies under static visual stimuli have provided valuable insights into the visual system (Rossiter et al., 2001), they neglect the continuity and dynamics of visual information, limiting our understanding of how the brain encodes memory within a continuous visual flow. Hence, it is crucial to explore how the brain processes memory information and how the representation of memory changes within the brain under continuous visual stimuli. Therefore, we endeavor to analyze memory under continuous visual stimuli to advance research on how the brain processes continuous visual stimuli.

Memory is a core component of human cognitive architecture, allowing us to store and recall past experiences. In visual perception, memory involves not only encoding individual scenes but also integrating and updating continuous streams of visual information (Miller, 1956; Cowan, 2001). According to working memory theory, the human brain can temporarily store and manipulate information (Baddeley, 1992). However, the capacity of short-term memory especially working memory is limited, typically around a few items (Luck and Vogel, 1997).

One of the motivations of our study is to explore the ability of working memory to retain past information under continuous visual stimuli by analyzing fMRI signals, particularly the semantic information of images from the past few moments. We use two data analysis techniques, ridge regression analysis and trial-wise representational similarity analysis (RSA), to assess the correlation between fMRI signals and visual stimuli from different past time points. We find that the correlation between fMRI signals and past semantic information gradually decreases over time, retaining at most 3-4 items, which aligns with the characteristics of working memory (Luck and Vogel, 1997; Baddeley, 1992).

Based on the analysis of the correlation between fMRI signals and past visual stimuli, we propose the new Memory Disentangling task. This task aims to extract past visual stimuli information from brain activity and separate it from ongoing brain activity to mitigate the effects of proactive interference. To simplify this task, we focus on Memory Disentangling based on decoding the semantic information of these stimulus. Specifically, we decode the fMRI signals to get the semantic information of images both current and past moments. A schematic illustration of this process is shown in Figure 1. This task can contribute to advancing brain memory decoding research. Besides, by decoding past information, it also addresses the low temporal resolution issue inherent in fMRI signals.

To achieve this goal, we first propose a straightforward method for Memory Disentangling by employing multiple separate Multilayer Perceptron to map fMRI signals to semantic features of images at multiple time points. Additionally, inspired by proactive interference in working memory (Oberauer and Kliegl, 2001; Keppel and Underwood, 1962), we introduce contrastive learning for disentangling. This method leverages relationships between consecutive pairs of fMRI signals to enhance the accuracy of extracting past information. Subsequently, we discuss how to transform these semantic features into intuitive, textual representations of semantic content and evaluate their effectiveness.

Our contributions are as follows:

• We analyze the capacity of working memory using fMRI signals and proposed a new task“Memory Disentangling” based on these findings, aiming to decode past information from current brain signals and mitigate memory interference.

• We introduce a memory disentangled contrastive learning method to accomplish the Memory Disentangling task, leveraging the theory of proactive interference to disentangle past memory information from current fMRI signals.

• We conduct extensive experiments to validate the role of disentangled contrastive learning, demonstrating its effectiveness for mitigate memory interference and providing insights that guide future brain decoding tasks to consider the impact of past memories.

## 2 Analysis and Task Definition

As previously mentioned, working memory has the ability to temporarily store and manipulate information. We investigate the capacity of working memory to retain past information based on fMRI signals under continuous visual stimuli. Based on this investigation, we propose a new task in the field of brain decoding, termed “Memory Disentangling”, which will be detailed in Section 2.2.

## 2.1 Visual Memory Analysis

In this section, we explore whether fMRI signals can reflect past visual stimuli and the duration for which this information be retained in the fMRI signals. To achieve this goal, we employ ridge regression analysis and trial-wise RSA to examine the correlation between fMRI signals and visual stimuli from different past time points. The overview of this section is displayed in Figure 2.

![](images/54ce51601faeb0296d27dbb4d91999300bf47e0f0b18aa5bc8de06ea37b1011d.jpg)

![](images/4c568310a0ebd745c01e43f4ae2a5340d9eb8077a0f41d5567120c30a62a80af.jpg)

![](images/297c69e8fb8726bca9b97e906d63c4856f10aaa0c3414de9edb7ee507a06149f.jpg)  
Figure 2: Overview of visual memory analysis. (a) Acquisition of continuous visual stimuli data, including image embeddings and fMRI signals. (b) Ridge regression analysis for visual memory, where ${ \therefore } { \bf \nabla } ( { \bf \vec { k } } ^ { \prime } )$ represents the ridge regression model, and k is offset. The figure illustrates an example for k = 2. (c) Trail-wise RSA, with the meaning of k remaining consistent with the previous context. Note that, for explanatory purposes, the size of the RDMs in the figure is illustrative and not representative of the actual size.

![](images/258cb6de6d45733a8a47b9ad775e401d3ca3363efdffb2b6e72dec7e79794fd0.jpg)  
Figure 3: The results of ridge regression analysis for four participants.

Data Acquisition The Natural Scenes Dataset (NSD) (Allen et al., 2022), which is the largest fMRI image stimulation dataset, is applied in the experiment. During the dataset creation process, subjects in each session are guided to observe a sequence of images, and are asked whether the current image has shown before. The fMRI signals during the observation of each image are recorded. As illustrated in Figure 2(a), by using the templates provided by NSD, the 3D fMRI data collected from one specific subject can be converted into vectors, yielding sequences $F _ { 1 } , F _ { 2 } , \ldots , F _ { n }$ . These vectors are regraded as input to a pre-trained CLIP image encoder for obtaining meaningful embeddings of each image. The corresponding CLIP embedding sequence is written as $C _ { 1 } , C _ { 2 } , \ldots , C _ { n }$ , which will be used in the subsequent analyses.

![](images/fc6d6efe849d8efb5f6276c5c7b3a7457d62f62e5f0fc84ddeb205252b6dfbf7.jpg)  
Figure 4: The trail-wise RSA results for four participants.

Ridge Regression Analysis To explore the amount of past information retained in brain signals, we formulate it as analyzing the correlation between the fMRI vector $F _ { t }$ at the current time step t and the CLIP embeddings $C _ { t - k }$ of different past time points. The offset k measures the number of time steps. For example, $k = 0$ means the current fMRI vector is paired with the current CLIP embedding, and $k = 1$ means the current fMRI vector is paired with last time point’s CLIP embedding. As the time span increases, offset k changes from 0 to max<sub>k</sub>. Ridge regression, a method to handle multicollinearity by introducing a regularization term to stabilize the model, is employed to investigate the correlation between $F _ { t }$ and $C _ { t - k }$ . Ridge regression is performed for each $\langle F _ { t } , C _ { t - k } \rangle$ pair respectively as k varies. The process is illustrated

in Figure 2(b).

We set $m a x _ { k } = 9$ and explore different k values from 0 to $m a x _ { k } .$ , where Notably, since each image in the NSD is presented three times and appears randomly in the image sequence, we ensure that images in the test set are not included in the training set. Afterwards, we sequentially apply ridge regression analysis with k ranging from 0 to max<sub>k</sub> to the partitioned data. Additionally, we establish a lowerbound by randomly matching all brain signals and CLIP embeddings (adhering to the test set division principle) and also performing ridge regression analysis.

The results of test set for different k values are shown in Figure 3, with the x-axis representing different k values, and the ’rand’ representing our lowerbound result. From the results, it can be observed that as the value of k increases, the correlation between brain signals and corresponding CLIP representations gradually decreases. Particularly at $k \mathit { \Theta } = \mathit { 3 }$ , the correlation approaches the lower bound, indicating that the information contained in the brain signals related to more than three items is minimal or challenging to extract.

Trial-wise Representational Similarity Analysis Another method we used for memory retention analysis is trial-wise representational similarity analysis. The computation process for each session is illustrated in Figure 2(c). For each session, we construct two representational dissimilarity matrice (RDM), $R D M _ { f }$ and $R D M _ { c } ,$ , using the sequences $F _ { 1 } , F _ { 2 } , \ldots , F _ { n }$ and $C _ { 1 } , C _ { 2 } , \ldots , C _ { n } .$ . In an RDM, the rows and columns represent the vectors (fMRI vectors or CLIP embeddings) corresponding to the stimuli, and the cell values indicate the dissimilarity between vectors (1 - the Pearson correlation coefficient $\rho )$ . Thus, an RDM contains the dissimilarity levels between every pair of stimuli (both $F _ { i }$ and $C _ { i }$ sequences) and is a symmetric n $\times \ n$ matrix. The computation process for $R D F _ { f }$ is as follow, and the calculation for $R D M _ { c }$ follows the same procedure.

$$
R D M _ { f } = J - \left[ \begin{array} { c c c c } { \rho ( F _ { 1 } , F _ { 1 } ) } & { \cdot \cdot \cdot } & { \rho ( F _ { 1 } , F _ { n } ) } \\ { \rho ( F _ { 2 } , F _ { 1 } ) } & { \cdot \cdot \cdot } & { \rho ( F _ { 2 } , F _ { n } ) } \\ { \vdots } & { \cdot } & { \vdots } \\ { \rho ( F _ { n } , F _ { 1 } ) } & { \cdot \cdot \cdot } & { \rho ( F _ { n } , F _ { n } ) } \end{array} \right]
$$

where J is an all-ones matrix.

(1)

RSA (Kriegeskorte et al., 2008) is well-suited for researchers to compare data across different modalities and even to bridge data from different species. Unlike traditional RSA based on the entire RDM matrix, our method focuses on the similarity representation between individual data trials. We calculate the trial-wise similarity between $R D M _ { f }$ and $R D M _ { c }$ . Here, we also use the offset $k ,$ where the $t _ { t h }$ row of $R D M _ { c }$ corresponds to the $( t + k ) _ { t h }$ row of $R D M _ { f }$ . We compute their correlation coefficient $\rho _ { k , t }$ and average the results of all rows that meet this requirement to obtain the trial-wise representational similarity score $\rho _ { k , a v e }$ for each session with offset k. Finally, we compute the average of values across all sessions to obtain the final trailwise RSA score.

The results are shown in Figure 4. Similar to ridge regression analysis, trial-wise RSA also exhibits similar trends. Furthermore, we replace $R D M _ { c }$ with $R D M _ { f }$ to compute the similarity between current and past brain activity signals. We will further elaborate in Appendix B, with the results shown in Figure 10.

## 2.2 Task Description

Based on the analysis above, we propose a task called “Memory Disentangling.” This task involves extracting information from past visual stimuli encoded in brain activity and separating this information from ongoing brain activity to mitigate the effects of proactive interference.

As shown in Figure 1, given the fMRI signal at time $t , F _ { t }$ , the task is to decode the image descriptions viewed at the current and previous total $( m a x _ { k } + 1 )$ time points, denoted as $\mathrm { C a p } : =$ $\{ c a p _ { t } , c a p _ { t - 1 } , \ldots , c a p _ { t - m a x _ { k } } \}$ . Our analysis indicates that the brain signals at the current time primarily contain information about the last three moments, thus we set $m a x _ { k } = 2$

It is important to note that the Memory Disentangling task is not limited to decoding brain activity into descriptions of images, and it can also involve other forms of decoding such as image reconstruction. Unlike previous visual stimuli decoding tasks, the core challenge here is to capture information about multiple past moments from a single time point’s fMRI signal and to remove the interfering parts of past information, thereby enabling higher quality information decoding. Additionally, due to the low temporal resolution of fMRI signals, brain activity between two scan frames may be lost. Memory Disentangling, which focuses on extracting past information, might help to supplement the missing information between scan frames.

![](images/ce96488fbab0a82cd0a65a7b7c6c613d393e275a490c9c186a2b46c102904768.jpg)  
Figure 5: A schematic diagram of the straightforward approach using separate MLPs.

This could alleviate the issue of fMRI’s temporal resolution and enhance the development of brain decoding under continuous stimulation.

## 3 Method

In this section, we propose a method to address Memory Disentangling task, starting with a straightforward method. Initially, we employ separate Multilayer Perceptrons (MLPs) for predicting embeddings corresponding to each moment based on the current fMRI signal. This approach is easy to implement but has significant limitations as it does not leverage the memory relational memory dynamics between successive fMRI signals. To overcome these limitations, we design a disentangled contrastive learning method based on the theory of proactive interference.

## 3.1 Straightforward Method Using Separate MLPs

For this task, we reformulate it as predicting the CLIP embeddings $C _ { t - k }$ for the current and the preceding $k _ { t h }$ moment using the current fMRI signal $F _ { t }$ , followed by generating image captions using a pre-trained CLIP-to-caption model. The schematic diagram of the straightforward method is displayed in the Figure 5. Thus, the key point of the task lies in mapping the fMRI signal $F _ { t }$ to the CLIP embeddings $C _ { t } , C _ { t - 1 } , C _ { t - 2 }$ . This method is to assign an MLP, denoted as $M L P _ { k }$ , for the mapping of the $k _ { t h }$ past moment. By inputting $F _ { t }$ into each $M L P _ { k }$ we obtain k outputs, where each is subjected to an MSE loss with its corresponding $C _ { i }$ , and the losses are summed up for training. The formula for the loss is as follows:

$$
\mathcal { L } _ { m s e } = \sum _ { i = 0 } ^ { m a x _ { k } } M S E _ { i } ( P _ { t - i } , C _ { t - i } )\tag{2}
$$

After that, for each CLIP embedding, the Pretrained CLIP-to-caption model can generate corresponding image caption.

## 3.2 Disentangled Contrastive Learning

Motivated by the desire to disentangle the information from past memories embedded in the fMRI signal $F _ { t }$ , we propose a disentangled contrastive learning approach based on memory proactive interference theory, which posits that cognitive processes are subject to the influence of previously acquired knowledge. Its core idea is that the memory component of current brain signals closely resembles the stimuli seen in the previous moment, a relationship present in all adjacent fMRI signals, thereby exhibiting continuity. That is, the neural representation of past information at time t is hypothesized to bear a closer resemblance to the current information at time t-1.

Accordingly, we introduce a contrastive learning method to disentangle the brain signal into “before” and “now” components of semantic information, which we term “disentangled contrastive learning.” This enables the fMRI disentangle encoder to learn to disentangle past components. Subsequently, we use MLPs for mapping as in the first part, and this process is depicted in Figure 6.

For the disentangled contrastive learning, we input consecutive fMRI signals $F _ { t - 1 } , F _ { t }$ into the same fMRI disentangle encoder, yielding four components: $b e f o r e _ { t - 1 } , n o w _ { t - 1 } , b e f o r e _ { t } .$ , and now . We set $n o w _ { t - 1 }$ and before<sub>t</sub> as positive samples, with all other pairings as negative samples, and then employ an InfoNCE (Oord et al., 2018) loss for training. For simplicity, we denote before<sub>t 1</sub> as $b _ { t - 1 }$ , before<sub>t</sub> as $b _ { t }$ $\mathbf { \varepsilon } , \ \mathrm { n o w } _ { t - 1 } \ \mathrm { a s } \ n _ { t - 1 }$ , and now<sub>t</sub> as $n _ { t }$

To compute the similarity for positive pairs, we first calculate the cosine similarity between $b _ { t }$ and $n _ { t - 1 }$ , denoted as $s ( b _ { t } , n _ { t - 1 } )$

$$
s ( b _ { t } , n _ { t - 1 } ) = \frac { b _ { t } \cdot n _ { t - 1 } } { \| b _ { t } \| \| n _ { t - 1 } \| }\tag{3}
$$

where $b _ { t } \cdot n _ { t - 1 }$ represents the dot product of $b _ { t }$ and $n _ { t - 1 }$ , and $\left\| \boldsymbol b _ { t } \right\|$ and $\lVert n _ { t - 1 } \rVert$ denote the norms of $b _ { t }$ and $n _ { t - 1 }$ , respectively.

To clarify the negative samples formed by two consecutive moments, we refer to Table 1. We also calculate their similarities in the same manner.

Finally,the InfoNCE loss is defined as:

$$
\mathcal { L } _ { I n f o N C E } = - \log \frac { \exp { \left( s ( b _ { t } , n _ { t - 1 } ) / \tau \right) } } { \sum _ { ( x , y ) \in S } \exp { \left( s ( x , y ) / \tau \right) } }\tag{4}
$$

![](images/9ec9043ee7922bf129b04785d58490f5229c48070ac706f7f99eca623b948b9c.jpg)  
Figure 6: Overview of the disentangled contrastive learning method. The left half illustrates the example of disentangled contrastive learning, while the right half shows the mapping process to the target image CLIP representations. Further explanation of the division of positive and negative sample pairs in disentangled contrastive learning is provided in Appendix D.

<table><tr><td rowspan=1 colspan=1>Negative Pair 1</td><td rowspan=1 colspan=1> $( n _ { t } , b _ { t } )$ </td></tr><tr><td rowspan=1 colspan=1>Negative Pair 2</td><td rowspan=1 colspan=1> $( n _ { t - 1 } , b _ { t - 1 } )$ </td></tr><tr><td rowspan=1 colspan=1>Negative Pair 3</td><td rowspan=1 colspan=1> $( n _ { t } , n _ { t - 1 } )$ </td></tr><tr><td rowspan=1 colspan=1>Negative Pair 4</td><td rowspan=1 colspan=1> $\left( b _ { t } , b _ { t - 1 } \right)$ </td></tr><tr><td rowspan=1 colspan=1>Negative Pair 5</td><td rowspan=1 colspan=1> $( n _ { t } , b _ { t - 1 } )$ </td></tr></table>

Table 1: Negative pairs for disentangled contrastive learning

where τ is a temperature parameter that controls the concentration of the distribution. The final training loss is a combination of the MSE loss and the InfoNCE loss:

$$
\mathcal { L } = \mathcal { L } _ { m s e } + \alpha \mathcal { L } _ { I n f o N C E }\tag{5}
$$

where α is a weighting factor for $\mathcal { L } _ { I n f o N C E }$

## 3.3 Semantic Feature Decoding

The two methods described above convert fMRI signals into CLIP embeddings, representing the semantic information of the visual stimuli at various time points. Subsequently, we need to transform these embeddings into textual descriptions, which are easier to observe and evaluate. CLIPCap (Mokady et al., 2021) is an image captioning model that generates descriptions from the CLIP embeddings of images. Given its superior performance, we use a pre-trained CLIPCap model to generate descriptions from our predicted CLIP embeddings. Consequently, we can ultimately convert the current fMRI signals into textual descriptions of the visual stimuli from the past few moments.

## 4 Experimental Settings

## 4.1 Dataset and Processing

In our study, we utilize the Natural Scenes Dataset (NSD), a large-scale dataset of fMRI scans in response to visual stimuli from MS COCO dataset (Lin et al., 2014). This dataset includes fMRI scans from eight subjects, obtained using a 7-Tesla fMRI scanner. During the scans, subjects are asked to view images and judge whether they have seen the presented image before. Each subject observes 9,000-10,000 distinct images, with each image appearing three times, randomly distributed throughout the image sequence. In the experiment, each subject undergoes 30-40 scan sessions, with each session containing 12 scan runs. There is a rest period between each pair of runs, and each run contains a continuous sequence of 62-63 images. Therefore, each session contains a total of 750 images. During each scan, image is presented for 3 seconds, followed by a 1-second blank screen. For more detailed information about the dataset, please visit the official website<sup>2</sup>.

In our study, we use the data from subject 1, 2, 5, and 7, as they have complete image scanning sessions, totaling 27,750 trials. We utilize the preprocessed functional scans at a resolution of 1.8 mm provided by NSD, along with the predefined template nsdgeneral to obtain fMRI vectors.

For each fMRI signal in the session, we use a sliding window of size 3 to store the CLIP image embeddings of continuous visual stimuli. Additionally, to ensure the images in the window are temporally contiguous, any data where the images span two different runs (indicating a long interval between stimuli presentations) are removed. Since each image is presented three times, it is crucial to strictly control data splitting to prevent contamination. We first select a test dataset of size m by randomly choosing m data pairs, each pair containing the fMRI signal $F _ { t }$ and the CLIP representations of images at times t 2, t 1, and t, in the form of $\langle F _ { t } ; C _ { t } , C _ { t - 1 } , C _ { t - 2 } \rangle$ . Images appearing in the test set are marked. Subsequently, we evaluate the remaining data, discarding any data points where the images in the window are already marked, thereby obtaining the training set. This process ensures that the training data is free from contamination.

![](images/30b7c8265df83dcb24aaef6fa875b9b6aed6a8c4c0bdb6f1bca9359b57f8e994.jpg)  
Figure 7: Qualitative results of Memory Disentangling for Subject 1. Each example includes the decoded results at three different time points along with their corresponding visual stimuli.

![](images/8a43c8b9e6c7029c08d6f3f68cdb0c60545871c7aebe6813b282cc06060fa261.jpg)  
Figure 8: Quantitative results of Memory Disentangling for Subject 1.

## 4.2 Implementation

In our task analysis section, we employ the NeuroRA (Lu and Ku, 2020) toolkit to compute the RDM matrix. Regarding Memory Disentangling, we opt for a $L _ { i n f o n c e }$ weight α of 0.01, a selection derive from a ablation study in Section 5.3. The size of our testing data m is set to 500, and a validation set of the same size was randomly selected from the partitioned training set. During the training phase, we optimized the model using AdamW (Loshchilov and Hutter, 2019) with an initial learning rate of 1e-5. We employ 5 different seeds for partitioning and training to enhance the reliability of our results. The reported experimental outcomes represent the average of these results obtained from 5 random seeds.

## 4.3 Evaluation Metrics

Since we employ the pre-trained CLIPCap model to generate image captions from the predicted disentangled outputs, semantics-based evaluation becomes more appropriate. To evaluate the degree of matching between the generated captions and the images, we obtain the COCO captions for all images in test set, and use the CIDEr (Vedantam et al., 2015), METEOR (Denkowski and Lavie, 2014), and SPICE (Anderson et al., 2016) metrics for evaluation. For the calculation of evaluation metrics, we use the nlg-eval<sup>3</sup> library, which is specifically designed for NLG evaluation.

<table><tr><td>Metrics</td><td></td><td></td><td>CIDEr(%)</td><td></td><td></td><td>METEOR(%)</td><td></td><td></td><td>SPICE(%)</td><td></td></tr><tr><td></td><td> $\left| { \widehat { \alpha } } ^ { \setminus { \big | } k \big | } \right.$ </td><td>0</td><td>1</td><td>2</td><td>0</td><td>1</td><td>2</td><td>0</td><td>1</td><td>2</td></tr><tr><td>SF</td><td>1</td><td> $3 4 . 3 { \scriptstyle \pm 3 . 1 6 }$ </td><td> $1 2 . 9 { \scriptstyle \pm 2 . 2 5 }$ </td><td> $1 1 . 0 { \scriptstyle \pm 0 . 9 9 }$ </td><td> $1 1 . 3 _ { \pm 0 . 4 1 }$ </td><td> $\mathbf { 8 . 5 2 _ { \pm 0 . 2 8 } }$ </td><td> $8 . 2 4 _ { \pm 0 . 1 7 }$ </td><td> $8 . 6 8 _ { \pm 1 . 0 6 }$ </td><td> $2 . 8 5 { \scriptstyle \pm 0 . 9 2 }$ </td><td> $1 . 9 1 _ { \pm 0 . 4 5 }$ </td></tr><tr><td>Ours</td><td>0</td><td> $\mid 3 5 . 1 _ { \pm 4 . 4 6 }$ </td><td> $\mathbf { 1 3 . 4 _ { \pm 0 . 7 0 } }$ </td><td> $\mathbf { 1 1 . 7 _ { \pm 0 . 4 0 } }$ </td><td> $1 1 . 3 _ { \pm 0 . 5 5 }$ </td><td> $8 . 1 4 _ { \pm 0 . 4 2 }$ </td><td> $8 . 2 3 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $9 . 3 1 _ { \pm 1 . 0 9 }$ </td><td> $\mathbf { 3 . 2 1 { \scriptstyle \pm 0 . 4 0 } }$ </td><td> $\mathbf { 2 . 2 9 _ { \pm 0 . 3 6 } }$ </td></tr><tr><td>Ours</td><td>0.01</td><td> $\mathbf { \lvert 3 9 . 9 2 . 1 1 }$ </td><td> $1 2 . 6 _ { \pm 1 . 7 6 }$ </td><td> $1 1 . 5 { \scriptstyle \pm 1 . 0 }$ </td><td> $\mathbf { \lvert 1 1 . 7 _ { \pm 0 . 3 5 } }$ </td><td> $8 . 2 4 _ { \pm 0 . 3 3 }$ </td><td> $8 . 1 5 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $\mathbf { \lvert 9 . 9 5 _ { \pm 0 . 6 4 } }$ </td><td> $2 . 8 4 _ { \pm 0 . 5 2 }$ </td><td> $2 . 2 5 { \scriptstyle \pm 0 . 5 1 }$ </td></tr><tr><td>Ours</td><td>0.1</td><td> $3 8 . 5 { \scriptstyle \pm 6 . 9 1 }$ </td><td> $1 0 . 9 { \scriptstyle \pm 0 . 9 9 }$ </td><td> $1 1 . 7 _ { \pm 0 . 9 4 }$ </td><td> $1 1 . 4 _ { \pm 0 . 7 3 }$ </td><td> $8 . 1 8 { \scriptstyle \pm 0 . 2 }$ </td><td> $\mathbf { 8 . 2 7 _ { \pm 0 . 2 9 } }$ </td><td> $9 . 4 5 _ { \pm 1 . 1 1 }$ </td><td> $1 . 8 0 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $1 . 8 3 { \scriptstyle \pm 0 . 6 0 }$ </td></tr></table>

Table 2: Ablation study results, including the straightforward method and our proposed disentangled contrastive learning method. The symbol α represents the weight of the loss $\mathcal { L } _ { I n f o N C E }$ . ’SF’ stands for Straightforward method mentioned in Section 3.1, and the best result for each k is bolded.

## 5 Results and Analysis

## 5.1 Qualitative Results

To provide an intuitive understanding of the Memory Disentangling task under continuous visual stimuli, Figure 7 reports several decoding examples from Subject 1. Each example includes semantic decoding results at three time points, representing the caption results decoded at the current, and two previous moments.

The results indicate that decoding at the current moment $( k = 0 )$ yields partially accurate results. However, the accuracy for past moments is relatively poor, with a tendency to set the subject as “a person/man ...”. This may be due to the high frequency of human figures in the dataset. Additionally, the decoding results tend to be broad; for example, at time point 1 in the first sample and time point 0 in the second sample, giraffes and cows are both decoded as “animal”. Occasionally, descriptions matching the images are obtained, indicating that decoding past information remains challenging. Overall, the decoding results exhibit discrepancies with the visual stimuli, attributed to the low signal-to-noise ratio (SNR) of brain activity signals. This low SNR makes brain decoding highly challenging, and extracting past information from these signals even more difficult.

## 5.2 Quantitative Results

Figure 8 presents the evaluation of decoding results of Subject 1 at three time points across different metrics, including CIDEr, METEOR, and SPICE.

Consistent with intuition, all metrics show varying degrees of decline over time (represented in the figure as increasing k values), with the CIDEr metric showing the most pronounced drop. This trend suggests that the accuracy and richness of the decoded descriptions deteriorate as the temporal distance from the current moment increases. Additional quantitative results for other subjects are provided in Appendix C.

## 5.3 Ablation Study

We conducted multiple experiments on the NSD dataset to perform Memory Disentangling task. We evaluated both the straightforward method and our proposed disentangled contrastive learning method with varying loss weight schemes represented by α. The experimental outcomes are summarized in Table 2.

The disentangled contrastive learning method, with a loss weight of 0.01, consistently achieves optimal results at the current time point, demonstrating its positive role in removing past interference from fMRI brain signals. This effect may mitigate some of the effects of proactive interference on memory. However, this was not reflected in the decoding at the subsequent two time points, indicating a need for further improvement in extracting past information, and a relatively large weight of $\mathcal { L } _ { I n f o N C E }$ might impede the mapping from fMRI to CLIP space, resulting in reduced decoding performance and increased variance. We speculate that this might be due to some information loss in the representation of extracted past information.

## 6 Conclusion

This study proposes the task of Memory Disentangling by analyzing the past information contained in working memory under continuous visual stimuli. Based on the phenomenon of proactive interference, we introduce a disentangled contrastive learning method to complete the Memory Disentangling task, which involves decoding semantic content at multiple time points from fMRI signals and remove the interfering parts of past information. This approach may help alleviate the low temporal resolution of fMRI and contribute new insights to the field of brain decoding.

## Limitations

Although we explored the content of past information contained in fMRI signals, its interpretability remains limited. Additionally, while we focused on semantic decoding for Memory Disentangling tasks, we did not address other forms of memory disentanglement, such as image reconstruction. While our proposed disentangled contrastive learning method showed improvement in currenttime decoding, its effectiveness in extracting past memory information was suboptimal, necessitating further in-depth exploration in future research. Specifically, there is a need to investigate how to optimize models to better capture past memory accurately and to enhance the model’s ability to learn from brain signals. Furthermore, expanding to other Memory Disentangling tasks would help comprehensively assess the method’s generality and applicability, thus advancing the field of cognitive neuroscience.

## Ethical Statement

We are committed to maintaining the highest standards of ethical conduct in our research endeavors. The dataset used in this study adheres strictly to ethical guidelines, encompassing rigorous informed consent protocols, participant confidentiality safeguards, and meticulous data handling practices. Our commitment to transparency ensures that all research procedures are conducted with integrity, while prioritizing the security and privacy of participant data. By adhering to these ethical standards, we aim to responsibly advance the fields of neuroscience and artificial intelligence, contributing meaningfully to scientific knowledge and societal well-being.

## Acknowledgements

This research is supported by the National Natural Science Foundation of China (No.62476127, No.62106105), the Natural Science Foundation of Jiangsu Province (No.BK20242039), the CCF-Baidu Open Fund (No.CCF-Baidu202307), the

CCF-Zhipu AI Large Model Fund (No.CCF-Zhipu202315), the Fundamental Research Funds for the Central Universities (No.NJ2023032), the Scientific Research Starting Foundation of Nanjing University of Aeronautics and Astronautics (No.YQR21022), and the High Performance Computing Platform of Nanjing University of Aeronautics and Astronautics.

## References

Emily J Allen, Ghislain St-Yves, Yihan Wu, Jesse L Breedlove, Jacob S Prince, Logan T Dowdle, Matthias Nau, Brad Caron, Franco Pestilli, Ian Charest, et al. 2022. A massive 7t fmri dataset to bridge cognitive neuroscience and artificial intelligence. Nature neuroscience, 25(1):116–126.

Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. 2016. Spice: Semantic propositional image caption evaluation. In Computer Vision– ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part V 14, pages 382–398. Springer.

Alan Baddeley. 1992. Working memory. Science, 255(5044):556–559.

Peter A Bandettini, Eric C Wong, R Scott Hinks, Ronald S Tikofsky, and James S Hyde. 1992. Time course epi of human brain function during task activation. Magnetic resonance in medicine, 25(2):390– 397.

Jiaxuan Chen, Yu Qi, Yueming Wang, and Gang Pan. 2023. Mindgpt: Interpreting what you see with non-invasive brain recordings. arXiv preprint arXiv:2309.15729.

Zijiao Chen, Jiaxin Qing, and Juan Helen Zhou. 2024. Cinematic mindscapes: High-quality video reconstruction from brain activity. Advances in Neural Information Processing Systems, 36.

Andy Clark. 2013. Whatever next? predictive brains, situated agents, and the future of cognitive science. Behavioral and brain sciences, 36(3):181–204.

Nelson Cowan. 2001. The magical number 4 in shortterm memory: A reconsideration of mental storage capacity. Behavioral and brain sciences, 24(1):87– 114.

Simon W Davis, Benjamin R Geib, Erik A Wing, Wei-Chun Wang, Mariam Hovhannisyan, Zachary A Monge, and Roberto Cabeza. 2021. Visual and semantic representations predict subsequent memory in perceptual and conceptual memory tests. Cerebral Cortex, 31(2):974–992.

Michael Denkowski and Alon Lavie. 2014. Meteor universal: Language specific translation evaluation for any target language. In Proceedings ofthe ninth

workshop on statistical machine translation, pages 376–380.

Magdalena Fafrowicz, Anna Ceglarek, Justyna Olszewska, Anna Sobczak, Bartosz Bohaterewicz, Monika Ostrogorska, Patricia Reuter-Lorenz, Koryna Lewandowska, Barbara Sikora-Wachowicz, Halszka Oginska, et al. 2023. Dynamics of working memory process revealed by independent component analysis in an fmri study. Scientific Reports, 13(1):2900.

Gary H Glover. 2011. Overview of functional magnetic resonance imaging. Neurosurgery Clinics, 22(2):133– 139.

James V Haxby, M Ida Gobbini, Maura L Furey, Alumit Ishai, Jennifer L Schouten, and Pietro Pietrini. 2001. Distributed and overlapping representations of faces and objects in ventral temporal cortex. Science, 293(5539):2425–2430.

Esperanza Bausela Herreras. 2010. Cognitive neuroscience; the biology of the mind. Cuadernos de Neuropsicología/Panamerican Journal of Neuropsychology, 4(1):87–90.

Yukiyasu Kamitani and Frank Tong. 2005. Decoding the visual and subjective contents of the human brain. Nature neuroscience, 8(5):679–685.

Kendrick N Kay, Thomas Naselaris, Ryan J Prenger, and Jack L Gallant. 2008. Identifying natural images from human brain activity. Nature, 452(7185):352– 355.

Geoffrey Keppel and Benton J Underwood. 1962. Proactive inhibition in short-term retention of single items. Journal of verbal learning and verbal behavior, 1(3):153–161.

Nikolaus Kriegeskorte, Marieke Mur, and Peter A Bandettini. 2008. Representational similarity analysisconnecting the branches of systems neuroscience. Frontiers in systems neuroscience, 2:249.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In Computer Vision– ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer.

Nikos K Logothetis. 2008. What we can do and what we cannot do with fmri. Nature, 453(7197):869–878.

Jack M Loomis, Roberta L Klatzky, and Nicholas A Giudice. 2018. -sensory substitution of vision: Importance of perceptual and cognitive processing. In Assistive technology for blindness and low vision, pages 179–210. CRC press.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net.

Zitong Lu and Yixuan Ku. 2020. Neurora: A python toolbox of representational analysis from multimodal neural data. Frontiers in neuroinformatics, 14:563669.

Steven J Luck and Edward K Vogel. 1997. The capacity of visual working memory for features and conjunctions. Nature, 390(6657):279–281.

Junlian Luo and Thérèse Collins. 2023. The representational similarity between visual perception and recent perceptual history. Journal of Neuroscience, 43(20):3658–3665.

David Marr. 2010. Vision: A computational investigation into the human representation and processing of visual information. MIT press.

George A Miller. 1956. The magical number seven, plus or minus two: Some limits on our capacity for processing information. Psychological review, 63(2):81.

Ron Mokady, Amir Hertz, and Amit H Bermano. 2021. Clipcap: Clip prefix for image captioning. arXiv preprint arXiv:2111.09734.

Thomas Naselaris, Ryan J Prenger, Kendrick N Kay, Michael Oliver, and Jack L Gallant. 2009. Bayesian reconstruction of natural images from human brain activity. Neuron, 63(6):902–915.

Klaus Oberauer and Reinhold Kliegl. 2001. Beyond resources: Formal models of complexity effects and age differences in working memory. European Journal ofCognitive Psychology, 13(1-2):187–215.

Seiji Ogawa, Tso-Ming Lee, Alan R Kay, and David W Tank. 1990. Brain magnetic resonance imaging with contrast dependent on blood oxygenation. proceedings of the National Academy of Sciences, 87(24):9868–9872.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748.

Furkan Ozcelik, Bhavin Choksi, Milad Mozafari, Leila Reddy, and Rufin VanRullen. 2022. Reconstruction of perceived images from fmri patterns and semantic brain exploration using instance-conditioned gans. In 2022 International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE.

Furkan Ozcelik and Rufin VanRullen. 2023. Natural scene reconstruction from fmri signals using generative latent diffusion. Scientific Reports, 13(1):15666.

Charan Ranganath and Mark D’Esposito. 2005. Directing the mind’s eye: prefrontal, inferior and medial temporal mechanisms for visual working memory. Current opinion in neurobiology, 15(2):175–182.

John R Rossiter, Richard B Silberstein, Philip G Harris, and Geoff Nield. 2001. Brain-imaging detection of visual scene encoding in long-term memory for tv commercials. Journal of Advertising Research, 41(2):13–21.

Paul Scotti, Atmadeep Banerjee, Jimmie Goode, Stepan Shabalin, Alex Nguyen, Aidan Dempster, Nathalie Verlinde, Elad Yundler, David Weisberg, Kenneth Norman, et al. 2024. Reconstructing the mind’s eye: fmri-to-image with contrastive learning and diffusion priors. Advances in Neural Information Processing Systems, 36.

Jingyuan Sun, Mingxiao Li, Zijiao Chen, Yunhao Zhang, Shaonan Wang, and Marie-Francine Moens. 2024. Contrast, attend and diffuse to decode high-resolution images from brain activities. Advances in Neural Information Processing Systems, 36.

Yu Takagi and Shinji Nishimoto. 2023. High-resolution image reconstruction with latent diffusion models from human brain activity. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14453–14463.

Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. 2015. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4566–4575.

Weihao Xia, Raoul de Charette, Cengiz Oztireli, and Jing-Hao Xue. 2024. Dream: Visual decoding from reversing human visual system. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 8226–8235.

Elahe’ Yargholi and Gholam-Ali Hossein-Zadeh. 2016. Brain decoding-classification of hand written digits from fmri data employing bayesian networks. Frontiers in human neuroscience, 10:351.

Qin Yin, Elizabeth L Johnson, Lingfei Tang, Kurtis I Auguste, Robert T Knight, Eishi Asano, and Noa Ofen. 2020. Direct brain recordings reveal occipital cortex involvement in memory development. Neu ropsychologia, 148:107625.

## A Related Work

## A.1 Brain Decoding of Visual Stimuli

Brain decoding aims to interpret neural activity patterns and link them to perceptual, cognitive, or motor processes. Recent advancements in neuroimaging technologies, particularly functional magnetic resonance imaging (fMRI), significantly enhance our ability to decode visual stimuli. (Haxby et al., 2001) make groundbreaking progress in this field by revealing how the ventral temporal cortex encodes different categories of visual objects. (Kamitani and Tong, 2005) use multivoxel pattern analysis (MVPA) to classify the direction of visual stimuli based on brain activity. Building on this early work, researchers explore various brain decoding tasks, such as stimulus classification and reconstruction.

Stimulus classification involves categorizing different types of visual stimuli based on brain activity. (Yargholi and Hossein-Zadeh, 2016) utilize an enhanced Naive Bayes classifier to decode handwritten digits from fMRI data. The current focus in brain decoding shifts more towards image reconstruction. Early image reconstruction techniques use linear regression models to map fMRI signals to given image features (Naselaris et al., 2009; Kay et al., 2008). With the advancement of deep learning technologies, more image reconstruction methods now employ Latent Diffusion Models (LDM) with image generation capabilities, achieving highquality reconstruction results (Scotti et al., 2024; Ozcelik and VanRullen, 2023; Sun et al., 2024). Additionally, describing the content of images seen by subjects from brain signals can be viewed as a form of reconstruction—semantic reconstruction of images. (Chen et al., 2023) use cross-attention and GPT-2 to accomplish semantic reconstruction tasks.

Besides the reconstruction of static visual stimuli, some researchers also tackle the reconstruction of continuous visual stimuli. The low temporal resolution of fMRI makes this task particularly challenging. (Chen et al., 2024) use contrastive learning to map fMRI to the CLIP representation space, fine-tuning Stable Diffusion on a video-text dataset to successfully reconstruct coherent videos with clear semantic information.

## A.2 Tracking Visual Memory through Brain Activity Patterns

Research on visual memory trajectories focuses on decoding and tracking the storage and recall processes of visual stimuli in memory through brain activity patterns. This area of study not only enhances our understanding of how visual information is encoded and stored in the brain but also reveals the dynamic changes during memory retrieval. (Davis et al., 2021) employ fMRI and item-wise RSA to investigate how memory representations generated during the encoding of individual items influence subsequent contextual memory. (Luo and Collins, 2023) utilize electroencephalography (EEG) recordings and RSA analysis to explore the neural basis of sequential dependency in visual perception. Their findings indicate that EEG signals retain information about previously seen objects, which affects current perceptual responses. (Fafrowicz et al., 2023) use fMRI and Independent Component Analysis (ICA) to study the formation of working memory and false memory. Additionally, time-of-day effects are observed, influencing memory-related brain network activity and performance.

![](images/57e05d7558a9c62e5adba42ab8fe185c9d7b57988bb4e30c07a03b98133dd3b1.jpg)  
Figure 9: Further explanation of the positive and negative sample pairs division in Figure 6. The positive and negative sample pairs are represented by red solid lines and blue dashed lines, respectively, consistent with Figure 6.

![](images/44419c06ccffddd4a9d6bf6aa7b13b25568a5f0e35a948676b61b37556d67640.jpg)  
Figure 10: Trial-wise RSA Results of Current versus Past Brain Activity.

## B Similarity Analysis of Current and Past Brain Activity Signals

Since fMRI indirectly measures neural activity in the brain by detecting BOLD signals, and changes in blood oxygen levels and blood flow occur gradually and continuously, fMRI data also exhibit a certain level of continuity. We alse use trial-wise RSA to explore the correlation between brain activities at different times. Specifically, we replace $R D M _ { c }$ in Figure 2(c) with $R D M _ { f }$ , starting from $k = 1$ (since $k = 0$ represents the correlation between the current time and the current brain signal, which is always 1), keeping the rest of the operations unchanged. The RSA results between current and past brain activities are shown in Figure 10.

![](images/772f97f3d45e4992d94920617c78387d7d1f0d32a04dafd6ea4afe97b9ff30d7.jpg)  
Figure 11: Quantitative results of Memory Disentangling for subject 2.

From the figure, it can be observed that there is a gradual decrease in correlation between consecutive fMRI signals, and the RSA scores for each participant begin to stabilize around $k = 4$

## C Results from Other Subjects

In this section, we present the quantitative decoding results for the remaining participants, depicted in Figure 11, 12, and 13.

## D Additional Explanation of Disentangled Contrastive Loss

Due to the complexity of the positive and negative sample pairs division, in order to provide a clearer

![](images/628f680121c3427797293dd71595817ae331e07435fdc15a95d6cc997bc81a04.jpg)  
Figure 12: Quantitative results of Memory Disentangling for subject 5.

![](images/7b4fd7476f7d5e3ae7169b0c22f13ad9195d02c8710f6ceb9eb1e2783e97b711.jpg)  
Figure 13: Quantitative results of Memory Disentangling for subject 7.

explanation of this section, we separately illustrate the positive and negative samples in Figure 9 based on Figure 6.