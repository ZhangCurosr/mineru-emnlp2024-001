# On Training Data Influence of GPT Models

Yekun Chai Qingyi Liu\* Shuohuan Wang

Yu Sun\* Qiwei Peng Hua Wu\*

Baidu Inc. Sun Yat-sen University University of Copenhagen {chaiyekun,wangshuohuan}@baidu.com {liuqy95}@mail2.sysu.edu.cn

## Abstract

Amidst the rapid advancements in generative language models, the investigation of how training data shapes the performance of GPT models is still emerging. This paper presents GPTfluence, a novel approach that leverages a featurized simulation to assess the impact of training examples on the training dynamics of GPT models. Our approach not only traces the influence of individual training instances on performance trajectories, such as loss and other key metrics, on targeted test points but also enables a comprehensive comparison with existing methods across various training scenarios in GPT models, ranging from 14 million to 2.8 billion parameters, across a range of downstream tasks. Contrary to earlier methods that struggle with generalization to new data, GPTfluence introduces a parameterized simulation of training dynamics, demonstrating robust generalization capabilities to unseen training data. This adaptability is evident across both fine-tuning and instruction-tuning scenarios, spanning tasks in natural language understanding and generation. We make our code and data publicly available at https:// github.com/ernie-research/gptfluence.

## 1 Introduction

The advent of generative language models, particularly the GPT series (Radford et al., 2019; Brown et al., 2020; Zhang et al., 2022), has marked a paradigm shift in natural language processing (NLP) (Touvron et al., 2023; Jiang et al., 2023), code generation (Lozhkov et al., 2024; Chai et al., 2023), visual and language understanding (Achiam et al., 2023; Team et al., 2023). These models have redefined performance standards across an extensive range of tasks, igniting detailed investigations into the process of training dynamics and the intricate nature of learned representations. Despite these strides, the specific influence of individual training examples on the performance of GPT models remains a significantly underexplored area. This oversight presents a critical challenge in optimizing training processes, a challenge that grows in tandem with the increasing complexity and scale of these models.

Current research has yet to focus comprehensively on the influence of training data on autoregressive language models. Prior studies, such as those utilizing the BERT (Park et al., 2023) or T5 architecture (Guu et al., 2023), have predominantly concentrated on natural language understanding tasks, leaving a considerable void in the exploration of generative language models.

Furthermore, the majority of this research (Pruthi et al., 2020; Guu et al., 2023; K and Søgaard, 2021; Koh and Liang, 2017; Yeh et al., 2018) has focused on test loss as the primary metric of interest, neglecting other vital performance indicators. Metrics such as BLEU (Papineni et al., 2002) and ROUGE (Lin, 2004) scores are crucial for a thorough evaluation of a model's capabilities, particularly in the context of generative language models where downstream task performance is paramount. Additionally, the challenge of generalizability—extending methodologies to accommodate unseen data—persists as a significant barrier (Guu et al., 2023). This is particularly critical for models expected to adapt to the dynamic and evolving trajectory of NLP tasks.

In response to these gaps, we introduce GPTfluence, a novel framework designed to extend the analysis of training data influence beyond the limitations of existing methodologies and across a broader spectrum of tasks. Employing a featurized simulation approach, GPTfluence estimates the impact of individual training examples on the performance of GPT models, covering both natural language understanding and generation tasks. This expanded focus facilitates a comprehensive understanding of model training dynamics, providing insights into a wide array of evaluation metrics beyond mere test loss.

Extensive experiments on selected subsets from FLAN datasets (Wei et al., 2022), across a variety of tasks and GPT model variants (Biderman et al., 2023), ranging in size from 14 million to 2.8 billion parameters, validate the effectiveness and superiority of our approach. Notably, our method not only sheds light on the training dynamics of GPT models but also demonstrates remarkable generalization capabilities to unseen data.

Contribution To summarize, our contributions are as follows:

• We introduce GPTfluence, a featurized simulation approach that significantly advances the analysis of training data influence on GPT models. This approach not only enables a comprehensive comparison with existing methodologies but also marks the first extensive foray into the extensive investigation of training data's impact on the performance of GPT models across various scales.

• Our approach demonstrates effectiveness on GPT models across different scales, showing its generalization capability on unseen data.

• We release the GPTDynamics dataset, a collection encompassing over 350 runs of training dynamics data spanning six distinct model sizes and five NLP tasks, to facilitate further research advancement.

## 2 Preliminaries

In this section, we revisit the conceptual framework of training data attribution (TDA) methods, aiming to quantify the impact of individual training instances on the performance of models with respect to test data points.

## 2.1 Task Definition

Considering the data space $Z ,$ such as datasets utilized for instruction-tuning, we denote a training example by z and a test example by $z ^ { \prime } \mathrm { i n } Z$ . We employ a model, specifically a GPT variant in our experiments, parameterized by weights $\theta \in \mathbb { R } ^ { p }$ . Our objective is to forecast the model's performance on a target metric $\phi ( \theta , z ) : \mathbb { R } ^ { p } \times Z \to$ R, with a main focus in existing literature on predicting test set loss (Pruthi et al., 2020; Guu et al., 2023).

Practically, this involves working with a sequence of training batches $c = ( c _ { 1 } , c _ { 2 } , . . . , c _ { T } )$ delineating a training curriculum. Here, Ct symbolizes the batch of training examples utilized at step t.

The crux of our task is to ascertain the influence of training examples z on a test example of interest $z ^ { \prime } ,$ specifically in terms of a test metric score $\phi ( \theta , z ^ { \prime } )$ given the training curriculum c. This involves tracking changes in performance trajectory as a function of the curriculum $^ { c , }$ with prior research predominantly focused on test loss prediction, rather than a broader spectrum of performance metrics.

## 2.2 Training Data Attribution

TracIn Inspired by the fundamental theorem of calculus—which posits that the integral of a function's gradient over an interval equals the function's value difference across that interval—TracIn (Pruthi et al., 2020) employs the firstorder Taylor expansion to quantify the data influence on test example loss at each step as follows:

$$
\mathcal { L } _ { t + 1 } ( z ) \approx \mathcal { L } _ { t } ( z ) - \eta _ { t } \langle \nabla \mathcal { L } _ { t } ( z _ { i } ) , \nabla _ { \theta } \mathcal { L } _ { t } ( z ^ { \prime } ) \rangle\tag{1}
$$

where $\eta _ { t }$ represents the learning rate at step t, and $\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { t } ( \cdot )$ signifies the gradient of the loss function with respect to the model weights θ.

It adopts an influence measurement that utilizes checkpoint ensembling, dubbed TracInCP. This approach aggregates the influences calculated at predefined intervals throughout the training, providing a comprehensive view of the training data's impact over time.

$$
\mathcal { T } _ { \mathrm { T r a c I n } } ( z _ { i } , z ^ { \prime } ) = \sum _ { i = 1 } ^ { N } \eta _ { i } \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { t } ( z _ { i } ; \boldsymbol { \theta } _ { i } ) ^ { \top } \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { t } ( z ^ { \prime } ; \boldsymbol { \theta } _ { i } )\tag{2}
$$

where I denotes the loss change w.r.t. the training example $z ,$ and N indicates the total number of model checkpoints saved during training.

Simfluence (Guu et al., 2023) approaches the challenge by learning a linear function $f$ that correlates training samples z with the test loss $\mathcal { L } ( z ^ { \prime } ; \theta )$ , expressed as:

$$
\mathcal { L } _ { t } ( z ) = \alpha ( c _ { t } ) \mathcal { L } _ { t - 1 } ( z ) + \beta ( c _ { t } )\tag{3}
$$

Here, $\alpha ( c _ { t } )$ and $\beta ( c _ { t } )$ , the multiplicative and additive factors respectively, are determined using a linear model, with $c _ { t }$ indicating the batch of examples consumed at training step t. Although it offers a data-driven simulator derived from training dynamics trajectories, its mapping from training data indices to test data points constrains generalizability to new, unseen data.

While TracIn leverages the neural model's firstorder gradients and Simfluence employs a datadriven simulation approach, both primarily focus on predicting test loss. Our proposed method aligns with Simfluence's direction but seeks to overcome its limitations, extending our focus to encompass a wider array of performance metrics beyond mere test loss prediction.

## 3 GPTfluence: Featurized Simulation-based Approach

## 3.1 Overview

We present GPTfluence, a novel approach for tracking the impact of training examples on the training dynamics of GPT models using a featurized simulator. Figure 1 depicts the process of GPTfluence, encompassing the collection of training dynamics, the training of the simulator, and the execution of the final simulation. Similar to Guu et al. (2023), our initial step involves gathering a comprehensive dataset of training dynamics, which captures both the training curriculum and various target metrics for test examples, extending beyond traditional loss metrics to include performance measures like BLEU and ROUGE scores.

GPTfluence models these dynamics via an n-th order Markov process, incorporating both multiplicative and additive factors to reflect the influence of training examples. At its core, the simulator uses a pre-trained encoder to attain the general representation of training and test examples, ensuring adaptability to new, unseen data. This is achieved by modeling the intricate interplay between examples through the interactions within their condensed hidden vector representations. In its application, it can autoregressively forecast the complete performance trajectory of a test example, starting from its initial performance metrics and following the specified training curriculum.

The collection of training dynamics is pivotal for predicting a test sample's performance trajectory throughout the training process. As outlined in §2.1, a T time steps training run is characterized by a sequence of training batches c, each contributing to the model's evolving parameters, $\theta _ { t }$ , through gradient descent.

To monitor the performance evolution of a particular test example $z ^ { \prime }$ , we record its metric scores $y _ { t } = \phi ( \theta _ { t } , z ^ { \prime } )$ at every training step t, employing a variety of evaluation metrics beyond mere loss, such as BLEU and ROUGE. This comprehensive record, denoted as $y = \phi _ { 1 : T }$ , tracks the test example's performance across all $T$ steps of training.

From a broader dataset D, we sample K subsets $\mathcal { D } ^ { \prime } \subset \mathcal { D }$ for GPT model training, resulting in $K$ distinct training runs. These runs yield a rich dataset of training dynamics $\mathcal { D } _ { r u n } .$ encapsulating both the training curricula and the sequential target metric scores $\phi$ for each test point $z ^ { \prime } .$ This dataset is represented as $\mathcal { D } _ { r u n } = \{ c ^ { k } , y ^ { k } \} _ { k = 1 } ^ { K }$

## 3.2 Featurized Simulation Approach

In this work, we introduce a featurized simulation methodology designed to capture the effects of training examples on GPT model training dynamics. This method is predicated on conceptualizing the training process as a sequential, time-evolving Markov process, thereby enabling the simulation of metric trajectories across training iterations. Building upon the foundational insights of Guu et al. (2023), our model extends the conventional firstorder Markov assumption to an n-th order Markov process. This allows for the consideration of a test sample $z ^ { \prime } .$ , where its performance metric $\phi ( \cdot )$ at any given timestep t is influenced by its performance across the preceding n steps, encapsulated as $\{ \phi _ { t - 1 } , \phi _ { t - 2 } , \cdot \cdot \cdot , \phi _ { t - n } \}$

Our approach integrates both multiplicative and additive components within the simulation. The performance trajectory of a test sample $z ^ { \prime }$ is thus delineated by a combination of these factors, formulated as follows:

$$
\phi _ { t } ( z ^ { \prime } ) = \sum _ { j = 1 } ^ { n } \alpha _ { j } ( c _ { t } ) \phi _ { t - j } ( z ^ { \prime } ) + \beta ( c _ { t } )\tag{4}
$$

where $\alpha _ { 1 : n } ( \cdot )$ and $\beta ( \cdot )$ represent the learned functions attributed to the current training batch $c _ { t }$ Here, $\alpha _ { j } ( c _ { t } )$ and $\beta ( c _ { t } )$ are determined through the aggregation of influence factors $A _ { i , j }$ and $B _ { i }$ , respectively, across the training examples in $c _ { t } \mathrm { : }$

$$
\alpha _ { j } ( c _ { t } ) = \sum _ { i \in c _ { t } } A _ { i , j } , \quad \beta ( c _ { t } ) = \sum _ { i \in c _ { t } } B _ { i }\tag{5}
$$

We introduce a parameterized, featurized simulator that employs a pre-trained encoder $\Psi ( \cdot )$ such as BERT (Devlin et al., 2019) and GPT (Radford et al., 2019). This is adept at processing each training example $z _ { i }$ and test example $z ^ { \prime } .$ , generating predictive influence factors $A _ { i , j }$ and $B _ { i }$ through the encoded representations $h _ { z _ { i } }$ and $h _ { z ^ { \prime } }$

![](images/5f7636aebc0fda20281dbd65ba75b709aaa1e66dfc40a7bf28a81ae865c3394c.jpg)  
Figure 1: Overview of GPTfluence. Step 1: We sample training data to create curricula for training GPT models and compute the test metrics of test examples at each training step. All the training curricula and the ground-truth metrics are referred to as GPTDynamics. Step 2: We train our featurized simulator on GPTDynamics, taking into account training examples at current and previous steps with the test example as input and predicts the ground-truth metric. Step 3: Given a new curriculum with the test example of interest, start from the test metric at the first step, the simulator simulates the test metric in the future training steps in an autoregressive manner.

$$
h ^ { z _ { i } } = \Psi ( z _ { i } ) , \quad h ^ { z ^ { \prime } } = \Psi ( z ^ { \prime } ) ,\tag{6}
$$

where $h ^ { z _ { i } }$ and $h ^ { z ^ { \prime } }$ are the low-dimensional embeddings of the training and test examples, respectively. To preserve the encoder's semantic generalizability, we keep it frozen during the simulator's training.

The multiplicative and additive influence factors are then derived by passing the embeddings through the corresponding linear projections, which are subsequently integrated using a Frobenius product as follows:

$$
A _ { i , j } = \langle \mathbf { W } _ { ( j ) } ^ { \top } h _ { j } ^ { z _ { i } } , \mathbf { U } _ { ( j ) } ^ { \top } h ^ { z ^ { \prime } } \rangle _ { F }
$$

$$
B _ { i } = \langle \mathbf { W ^ { \prime } } ^ { \top } h _ { j } ^ { z _ { i } } , \mathbf { U ^ { \prime } } ^ { \top } h ^ { z ^ { \prime } } \rangle _ { F }\tag{7}
$$

(8)

where $\mathbf { W } _ { \mathrm { ( j ) } } , \mathbf { U } _ { \mathrm { ( j ) } } , \mathbf { W } ^ { \prime } , \mathbf { U } ^ { \prime }$ are learnable weights, $\langle \cdot , \cdot \rangle _ { F }$ represents the Frobenius inner product between the hidden representations of the training and test examples, yielding a refined estimation of the multiplicative influence exerted by each training example $z _ { i }$ on the test example's performance trajectory. Our approach offers a granular and comprehensive analysis of training dynamics through this intricate data-driven simulation.

To learn our featurized simulator Θ, we optimize

the following L2-regularized regression objective:

$$
\Theta ^ { \star } = \operatorname * { a r g m i n } _ { \Theta } \sum _ { t \in T } ( y _ { t } - \hat { \phi } _ { t } ( z ^ { \prime } ) ) ^ { 2 } + \lambda ( | | \Theta | | _ { 2 } ^ { 2 } )\tag{9}
$$

where λ is the discounting factor dictating the degree of L2-regularization, $\hat { \phi } _ { t } ( \cdot )$ is the test score prediction at step t using Eq.(4). Refer to Algorithm 1 for the pseudo-code.

## 3.3 Connection to Previous Approaches

Our approach offers a flexible framework that, under specific conditions, aligns with established models in the TDA literature. Specifically, when the focus narrows down to the overall influence of per-step dynamics, our approach converges to the datamodels (Ilyas et al., 2022; Engstrom et al. 2024). Moreover, in scenarios where the Markov order n is set to 1 and the input encoder is configured to process sample indices, our method reduces to Simfluence (Guu et al., 2023).

## 4 Experiments

## 4.1 Experimental Settings

## 4.1.1 GPTDynamics Data Collection

Datasets and GPT Training Scenarios In subsequent experiments, we refer to the comprehensive training process that employs the aggregated

<table><tr><td rowspan="3">Method</td><td rowspan="3">#Param</td><td colspan="3">RTE</td><td colspan="3">SST-2</td><td colspan="3">BoolQ</td></tr><tr><td rowspan="2">All-Steps</td><td>All-Steps</td><td>Final-Step Spear-</td><td>All-Steps</td><td>All-Steps</td><td>Final-Step Spear-</td><td>All-Steps</td><td>All-Steps</td><td>Final-Step Spear-</td></tr><tr><td>MSE (↓)</td><td>MAE(↓) man&#x27;s ρ (↑)</td><td>MSE (↓)</td><td>MAE (↓)</td><td>man&#x27;s ρ (↑)</td><td>MSE (↓)</td><td>MAE (↓)</td><td>man&#x27;s ρ (↑)</td></tr><tr><td>TracIn-CP (10-steps)</td><td rowspan="4"></td><td>1.156(0.838)</td><td>0.787(0.339)</td><td>0.460</td><td>0.551(0.560)</td><td>0.584(0.307)</td><td>-0.089</td><td>0.957(0.728)</td><td>0.735(0.332)</td><td>-0.066</td></tr><tr><td>TracIn-CP (all-steps)</td><td>0.757(0.591)</td><td>0.629(0.299)</td><td>0.460</td><td>0.446(0.555)</td><td>0.525(0.321)</td><td>-0.089</td><td>0.782(0.690)</td><td>0.680(0.339)</td><td>-0.066</td></tr><tr><td>Grad-Dot</td><td>410M 12.061(3.688)</td><td>2.906(0.410)</td><td>0.459</td><td>7.715(1.543)</td><td>1.918(0.205)</td><td>-0.084</td><td>12.527(3.617)</td><td>2.900(0.344)</td><td>-0.071</td></tr><tr><td>Simfluence</td><td>1.477(0.274)</td><td>0.634(0.111)</td><td>0.426(0.340)</td><td>1.133(0.287)</td><td>0.455(0.082)</td><td>0.696(0.156)</td><td>1.189(0.362)</td><td>0.485(0.082)</td><td>0.793(0.201)</td></tr><tr><td>Ours</td><td rowspan="5"></td><td>0.220(0.184)</td><td>0.334(0.140)</td><td>0.644(0.174)</td><td>0.111(0.045)</td><td>0.224(0.047)</td><td>0.834(0.129)</td><td>0.132(0.073)</td><td>0.251(0.075)</td><td>0.828(0.154)</td></tr><tr><td>TracIn-CP (10-steps)</td><td>1.225(0.744)</td><td>0.979(0.344)</td><td>-0.203</td><td>4.412(1.301)</td><td>1.697(0.170)</td><td>-0.058</td><td>0.999(1.034)</td><td>0.793(0.400)</td><td>0.649</td></tr><tr><td>TracIn-CP (all-steps)</td><td>1.137(0.740)</td><td>0.939(0.343)</td><td>-0.203</td><td>2.158(0.782)</td><td>1.218(0.187)</td><td>-0.058</td><td>0.858(1.043)</td><td>0.731(0.416)</td><td>0.649</td></tr><tr><td>Grad-Dot 1B</td><td>21.928(7.871)</td><td>4.332 (0.874)</td><td>-0.198</td><td>6.601(1.927)</td><td>2.077(0.193)</td><td>-0.057</td><td>18.270(5.630)</td><td>3.563(0.711)</td><td>0.650</td></tr><tr><td>Simfluence</td><td>0.889(0.551)</td><td>0.523(0.197)</td><td>0.360(0.207)</td><td>0.582(0.253)</td><td>0.410(0.084)</td><td>0.712(0.148)</td><td>0.876(0.470)</td><td>0.469(0.198)</td><td>0.862(0.050)</td></tr><tr><td>Ours</td><td rowspan="4"></td><td>0.099(0.078)</td><td>0.227(0.097)</td><td>0.757(0.123)</td><td>0.096(0.075)</td><td>0.221(0.084)</td><td>0.807(0.175)</td><td>0.068(0.058)</td><td>0.187(0.070)</td><td>0.953(0.034)</td></tr><tr><td>TracInCP (10-steps)</td><td>8.869(3.673)</td><td>2.700(0.650)</td><td>0.573</td><td>0.294(0.235)</td><td>0.447(0.176)</td><td>0.801</td><td>1.185(1.271)</td><td>0.804(0.436)</td><td>0.184</td></tr><tr><td>TracInCP (all-steps)</td><td>10.256(4.396)</td><td>2.967(0.652)</td><td>0.573</td><td>0.265(0.228)</td><td>0.419(0.178)</td><td>0.801</td><td>1.183(1.260)</td><td>0.800(0.434)</td><td>0.184</td></tr><tr><td>Grad-Dot</td><td>10.101(9.212)</td><td>2.580(1.327)</td><td>0.573</td><td>1.216(0.411)</td><td>0.935(0.175)</td><td>-0.801</td><td>1.990(1.082)</td><td>1.219(0.321)</td><td>0.184</td></tr><tr><td>Simfluence-linear</td><td>2.8B</td><td>2.032(1.214)</td><td>0.996(0.360)</td><td>0.845(0.061)</td><td>0.921(0.435)</td><td>0.634(0.194)</td><td>0.912(0.018)</td><td>1.545(1.293)</td><td>0.849(0.412)</td><td>0.681(0.087)</td></tr><tr><td>Ours</td><td rowspan="4">#Param</td><td>0.132(0.172)</td><td>0.273(0.129)</td><td>0.969(0.009)</td><td>0.023(0.015)</td><td>0.123(0.040)</td><td>0.979(0.006)</td><td>0.175(0.232)</td><td>0.305(0.165)</td><td>0.963(0.018)</td></tr><tr><td>Method</td><td></td><td>WebNLG</td><td></td><td></td><td>WMT-16 DE/EN</td><td></td><td></td><td>Average</td><td></td></tr><tr><td></td><td>All-Steps</td><td>All-Steps</td><td>Final-Step Spear-</td><td>All-Steps</td><td>All-Steps</td><td>Final-Step Spear-</td><td>All-Steps</td><td>All-Steps</td><td>Final-Step Spear-</td></tr><tr><td colspan="2"></td><td>MSE (↓)</td><td>MAE (↓)</td><td>man&#x27;s ρ (↑)</td><td>MSE (↓)</td><td>MAE (↓)</td><td>man&#x27;s ρ (↑)</td><td>MSE (↓)</td><td>MAE (↓)</td><td>man&#x27;s ρ (↑)</td></tr><tr><td>TracIn-CP (10-steps)</td><td rowspan="4">410M</td><td>0.048(0.072)</td><td>0.168(0.115)</td><td>0.836</td><td>0.030(0.071)</td><td>0.122(0.107)</td><td>0.963</td><td>0.548</td><td>0.479</td><td>0.421</td></tr><tr><td>TracIn-CP (all-steps)</td><td>0.050(0.073)</td><td>0.173(0.113)</td><td>0.836</td><td>0.030(0.071)</td><td>0.123(0.107)</td><td>0.963</td><td>0.413</td><td>0.426</td><td>0.421</td></tr><tr><td>Grad-Dot</td><td>0.062(0.080)</td><td>0.187(0.113)</td><td>0.837</td><td>0.033(0.073)</td><td>0.127(0.109)</td><td>0.963</td><td>6.479</td><td>1.608</td><td>0.421</td></tr><tr><td>Simfluence</td><td>0.036(0.029)</td><td>0.130(0.049)</td><td>0.986(0.002)</td><td>0.016(0.013)</td><td>0.101(0.034)</td><td>0.997(0.001)</td><td>0.770</td><td>0.361</td><td>0.779</td></tr><tr><td>Ours</td><td rowspan="4"></td><td>0.002(0.002)</td><td>0.033(0.017)</td><td>0.994(0.001)</td><td>0.002(0.004)</td><td>0.033(0.023)</td><td>0.998(0.000)</td><td>0.093</td><td>0.175</td><td>0.860</td></tr><tr><td>TracIn-CP (10-steps)</td><td>0.032(0.053)</td><td>0.132(0.095)</td><td>0.885</td><td>0.012(0.032)</td><td>0.075(0.069)</td><td>0.981</td><td>1.336</td><td>0.735</td><td>0.451</td></tr><tr><td>TracIn-CP (all-steps)</td><td>0.033(0.053)</td><td>0.135(0.094)</td><td>0.885</td><td>0.012(0.032)</td><td>0.076(0.069)</td><td>0.981</td><td>0.840</td><td>0.620</td><td>0.451</td></tr><tr><td>Grad-Dot</td><td>0.044(0.061)</td><td>0.154(0.097)</td><td>0.881</td><td>0.013(0.033)</td><td>0.075(0.071)</td><td>0.981</td><td>9.371</td><td>2.040</td><td>0.451</td></tr><tr><td>Simfluence</td><td rowspan="4">1B</td><td>0.167(0.127)</td><td>0.323(0.112)</td><td>0.823(0.030)</td><td>0.171(0.269)</td><td>0.309(0.168)</td><td>0.925(0.007)</td><td>0.537</td><td>0.407</td><td>0.737</td></tr><tr><td>Ours</td><td>0.007(0.005)</td><td>0.068(0.022)</td><td>0.984(0.005)</td><td>0.004(0.004)</td><td>0.049(0.020)</td><td>0.997(0.001)</td><td>0.055</td><td>0.150</td><td>0.900</td></tr><tr><td>TracInCP (10-steps)</td><td>0.005(0.008)</td><td>0.051(0.035)</td><td>0.978</td><td>0.001(0.002)</td><td>0.020(0.019)</td><td>0.997</td><td>2.071</td><td>0.804</td><td>0.707</td></tr><tr><td>TracInCP (all-steps)</td><td>0.005(0.008)</td><td>0.051(0.035)</td><td>0.978</td><td>0.001(0.002)</td><td>0.020(0.019)</td><td>0.997</td><td>2.342</td><td>0.851</td><td>0.707</td></tr><tr><td>Grad-Dot</td><td rowspan="4">2.8B</td><td>0.015(0.020)</td><td>0.089(0.061)</td><td>0.978</td><td>0.001(0.002)</td><td>0.021(0.019)</td><td>0.997</td><td>2.665</td><td>0.969</td><td>0.386</td></tr><tr><td>Simfluence-linear</td><td>0.102(0.065)</td><td>0.283(0.091)</td><td>0.971(0.004)</td><td>0.063(0.085)</td><td>0.203(0.119)</td><td>0.991(0.001)</td><td>0.933</td><td>0.593</td><td>0.880</td></tr><tr><td>Ours</td><td>0.001(0.001)</td><td>0.024(0.016)</td><td>0.997(0.000)</td><td>0.001(0.002)</td><td>0.020(0.016)</td><td>0.999(0.000)</td><td>0.066</td><td>0.149</td><td>0.981</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 1: Results of test loss estimation for instruction tuning. Results are averaged over 5 held-out test runs.

FLAN datasets along with task-specific instructions as instruction tuning. Conversely, the term ine-tuning is reserved to describe the process of individually optimizing models on separate tasks without the use of instructional prompts. Both instruction tuning and fine-tuning processes are encapsulated within our GPTDynamics dataset. We refer to Appendix §A.1 for detailed information.

GPT Backbone We employed Pythia (Biderman et al., 2023), a model suite recently made available to the public, as our foundational architecture. Within this suite, we selected five distinct models based on their sizes, encompassing 14M, 70M, 160M, 410M, 1B, and 2.8B, to ensure a broad range of computational capacities were represented.

## 4.1.2 Experiment Setup for Simulators

Baselines We select TracIn (Pruthi et al., 2020), Grad-Dot (Charpiat et al., 2019), and Simfluence (Guu et al., 2023) as our baselines. Refer to Appendix §A.2 for detailed information.

Evaluation Metrics We utilize a comprehensive set of metrics, including the Mean Squared Error (MSE) and Mean Absolute Error (MAE) calculated across all training steps, alongside the Spearman correlation coefficient (ρ) at the final step, to thoroughly assess performance.

<table><tr><td>Dataset</td><td>Method</td><td>All-Steps MSE (↓)</td><td>All-Steps MAE (↓)</td><td>Final-Step Spear- man&#x27;s ρ (↑)</td></tr><tr><td>RTE</td><td>Simfluence Ours</td><td>0.035(0.022) 0.036(0.029)</td><td>0.151(0.054) 0.151(0.060)</td><td>0.743(0.094) 0.746(0.095)</td></tr><tr><td>SST-2</td><td>Simfluence Ours</td><td>0.037(0.017) 0.014(0.006)</td><td>0.128(0.030) 0.081(0.018)</td><td>0.938(0.074) 0.943(0.073)</td></tr><tr><td>BoolQ</td><td>Simfluence Ours</td><td>0.032(0.019) 0.011(0.011)</td><td>0.140(0.038) 0.082(0.049)</td><td>0.992(0.002) 0.994(0.002)</td></tr><tr><td>WebNLG</td><td>Simfluence Ours</td><td>0.016(0.012) 0.011(0.014)</td><td>0.094(0.036) 0.078(0.043)</td><td>0.984(0.002) 0.985(0.002)</td></tr><tr><td>WMT-16 DE/EN</td><td>Simfluence Ours</td><td>0.010(0.008) 0.002(0.002)</td><td>0.067(0.029) 0.031(0.018)</td><td>0.998(0.003) 0.999(0.000)</td></tr><tr><td>Average</td><td>Simfluence Ours</td><td>0.026 0.015</td><td>0.116 0.084</td><td>0.931 0.933</td></tr></table>

Table 2: Results of test loss estimation for fine-tuning.

## 4.2 Test Loss Estimation

Instruction Tuning Table 1 presents a comparison between our approach and traditional TDA methods for instruction tuning. GPTfluence demonstrated a distinct edge over Simfluence and other gradient-based TDA techniques across a set of five natural language understanding (NLU) and natural language generation (NLG) tasks, as evidenced by the MSE and MAE metrics for the entire trajectory, alongside the Spearman correlation coefficients at the final time step across various test samples. Examples are shown in Fig. 2(a) and 2(b). Additionally, we observed that while the effectiveness of all evaluated TDA methods in predicting loss trajectories varied with changes in GPT sizes, GPTfluence maintained optimal performance, independent of the GPT scale.

<table><tr><td rowspan="2">Method</td><td rowspan="2">#Param</td><td colspan="5">WebNLG</td></tr><tr><td>All-steps</td><td>BLEU All-steps</td><td>Final-step Spear-</td><td>ROUGE-L All-steps All-steps</td><td>Final-step Spear-</td></tr><tr><td>Simfluence</td><td>410M</td><td>MSE (↓) 23.47(63.52)</td><td>MAE (↓) 2.34(3.26)</td><td>man&#x27;s ρ (↑) 0.81(0.02)</td><td>MSE (↓) 0.007(0.008)</td><td>MAE (↓) 0.055(0.038)</td><td>man&#x27;s ρ (↑) 0.708(0.067)</td></tr><tr><td>Ours</td><td></td><td>9.11(18.41) 20.58(60.80)</td><td>1.73(1.82)</td><td>0.90(0.03)</td><td>0.005(0.006)</td><td>0.045(0.034)</td><td>0.796(0.047)</td></tr><tr><td>Simfluence Ours</td><td>IB</td><td>9.72(23.70)</td><td>2.01(3.03)</td><td>0.87(0.03)</td><td>0.006(0.006)</td><td>0.052(0.031)</td><td>0.878(0.035)</td></tr><tr><td rowspan="3">Simfluence Ours</td><td rowspan="3">2.8B</td><td></td><td>1.63(2.02)</td><td>0.86(0.03)</td><td>0.004(0.005)</td><td>0.043(0.029)</td><td>0.903(0.020)</td></tr><tr><td>15.08(51.72)</td><td>1.52(2.90)</td><td>0.80(0.08)</td><td>0.005(0.006)</td><td>0.050(0.036)</td><td>0.817(0.063)</td></tr><tr><td>5.56(17.26)</td><td>1.15(1.42)</td><td>0.86(0.05)</td><td>0.003(0.003)</td><td>0.035(0.026)</td><td>0.911(0.050)</td></tr><tr><td>Method</td><td></td><td colspan="6">WMT-16 DE/EN</td></tr><tr><td rowspan="3"></td><td rowspan="3">#Param</td><td></td><td>BLEU</td><td></td><td></td><td>ROUGE-L</td><td></td></tr><tr><td>All-steps</td><td>All-steps</td><td>Final-Step Spear-</td><td>All-steps</td><td>All-steps</td><td>Final-Step Spear-</td></tr><tr><td>MSE (↓) 32.15(116.17)</td><td>MAE (↓) 2.25(4.08)</td><td>man&#x27;s ρ (↑) 0.83(0.03)</td><td>MSE (↓) 0.007(0.017)</td><td>MAE (↓) 0.039(0.055)</td><td>man&#x27;s ρ (↑) 0.931(0.014)</td></tr><tr><td>Simfluence Ours</td><td>410M</td><td>7.71(28.05)</td><td>1.14(1.92)</td><td>0.92(0.02)</td><td>0.004(0.009)</td><td>0.030(0.041)</td><td>0.964(0.012)</td></tr><tr><td>Simfluence Ours</td><td>IB</td><td>162.94(466.30) 46.33(122.50)</td><td>5.71(9.03) 3.34(4.68)</td><td>0.76(0.03)</td><td>0.025(0.038)</td><td>0.094(0.098)</td><td>0.833(0.031)</td></tr><tr><td>Simfluence</td><td>2.8B</td><td>64.07(319.93)</td><td>2.59(5.84)</td><td>0.93(0.01)</td><td>0.013(0.020)</td><td>0.066(0.069)</td><td>0.910(0.011)</td></tr><tr><td>Ours</td><td></td><td>24.27(93.41)</td><td>1.94(3.36)</td><td>0.90(0.05) 0.93(0.05)</td><td>0.008(0.022) 0.005(0.018)</td><td>0.040(0.059) 0.030(0.051)</td><td>0.912(0.045) 0.936(0.037)</td></tr><tr><td>Method</td><td></td><td colspan="5">Average</td><td></td></tr><tr><td rowspan="4"></td><td rowspan="2">#Param</td><td colspan="3">BLEU</td><td colspan="3">ROUGE-L</td></tr><tr><td>All-steps</td><td>All-steps</td><td>Final-step Spear-</td><td>All-steps</td><td>All-steps</td><td>Final-step Spear-</td></tr><tr><td></td><td>MSE (↓)</td><td>MAE (↓)</td><td>man&#x27;s ρ (↑)</td><td>MSE (↓)</td><td>MAE (↓)</td><td>man&#x27;s ρ (↑)</td></tr><tr><td>Simfluence</td><td>410M 27.81</td><td>2.29</td><td>0.82</td><td>0.007</td><td>0.047</td><td>0.820</td></tr><tr><td>Ours Simfluence</td><td rowspan="2">IB</td><td>8.41</td><td>1.43</td><td>0.91</td><td>0.004</td><td>0.037</td><td>0.880</td></tr><tr><td>91.76</td><td>3.86</td><td>0.81</td><td></td><td>0.015</td><td>0.073</td><td>0.855</td></tr><tr><td>Ours</td><td></td><td>28.02</td><td>2.51</td><td>0.90</td><td>0.008</td><td>0.055</td><td>0.907</td></tr><tr><td>Simfluence</td><td>2.8B</td><td>39.58</td><td>2.06</td><td>0.85</td><td>0.007</td><td>0.045</td><td>0.865</td></tr><tr><td>Ours</td><td></td><td>14.92</td><td>1.55</td><td>0.89</td><td>0.004</td><td>0.033</td><td>0.924</td></tr></table>

Fine-tuning In Table 2, it is evident that our approach consistently outperforms Simfluence when it comes to fine-tuning GPT models. On average, our method reduces the MSE and MAE across all training steps by 42% and 28%, respectively, when compared to Simfluence. This implies that our method is more robust and adaptable in simulating training dynamics.

## 4.3 Generalizing to Test Metric Estimation

We have expanded the evaluation of our model beyond the mere prediction of test loss, now including vital measures such as ROUGE and BLEU scores. We have not reported the performance of TracIn and Grad-Dot baselines due to its inability on such metric predictions.

Instruction Tuning As for instruction tuning, our findings, displayed in Table 3, demonstrate a superior performance of our method over Simfluence in predicting both BLEU and ROUGE-L scores and for GPTs of varying sizes. Intuitively, We draw some qualitative examples in the Fig. 2(c) and 2(d). Notably, for BLEU simulation on the WMT-16 DE/EN task, as the size of GPT increases, all steps MSE of Simfluence increases, whereas our method maintains a more stable performance, even exhibiting slight improvements from 0.92 to 0.93 in loss prediction accuracy at the final step. This suggests that our model is better equipped to manage more challenging tasks and larger model sizes, leveraging the pre-trained representations and instance interactions.

Fine-tuning Our method's superiority remains evident in the fine-tuning scenario, as depicted in Table 4, underscoring the robustness of our featurebased simulation approach. It's worth noting that the margin by which GPTf1uence outperforms Simfluence in BLEU metric simulation is not as pronounced in fine-tuning contexts as it is in instruction tuning settings. This discrepancy is likely due to the richer and more diverse data available in instruction tuning, which accentuates Simfluence's relative inefficiency, given its independent parameter learning for each training instance and a distinct simulator for each test instance.

## 4.4 Ablation Study

Practical Influence via Checkpoints Our featured simulator is adept at learning from past training dynamics. However, monitoring the training dynamics at every step can be expensive, especially when dealing with large-sized GPTs. Therefore, we conduct experiments to choose training checkpoints at specific intervals to approximate the reality of the neighboring points with the training state of that particular point. Then, we trained our simulator on the approximate training dynamics to find the balance between the cost of collecting training dynamics and the simulator performance.

Table 3: Results of test metric estimation on NLG datasets for instruction-tuning.
<table><tr><td>Dataset</td><td>Metric</td><td>Method</td><td>All-steps MSE (↓)</td><td>All-steps MAE (↓)</td><td>Final-Step Spear- man&#x27;s ρ (↑)</td></tr><tr><td rowspan="3">WebNLG</td><td>BLEU</td><td>Simfluence Ours</td><td>43.33 (77.34) 43.98 (81.40)</td><td>4.23 (3.52) 4.28 (3.57)</td><td>0.78 (0.02) 0.80 (0.01)</td></tr><tr><td rowspan="2">ROUGE-L</td><td>Simfluence</td><td>0.008 (0.007)</td><td>0.066 (0.031)</td><td>0.706 (0.038)</td></tr><tr><td>Ours</td><td>0.007 (0.006)</td><td>0.060 (0.029)</td><td>0.765 (0.040)</td></tr><tr><td rowspan="4">WMT-16 DE/EN</td><td rowspan="2">BLEU</td><td>Simfluence</td><td>32.11 (89.13)</td><td>2.76 (3.75)</td><td>0.82 (0.02)</td></tr><tr><td>Ours</td><td>30.26 (77.23)</td><td>2.91 (3.69)</td><td>0.81 (0.02)</td></tr><tr><td rowspan="2">ROUGE-L</td><td>Simfluence</td><td>0.018 (0.025)</td><td>0.091 (0.075)</td><td>0.796 (0.032)</td></tr><tr><td>Ours</td><td>0.012 (0.016)</td><td>0.075 (0.057)</td><td>0.843 (0.010)</td></tr><tr><td rowspan="4">Average</td><td rowspan="2">BLEU</td><td>Simfluence</td><td>37.72</td><td>3.49</td><td>0.80</td></tr><tr><td>Ours</td><td>37.12</td><td>3.59</td><td>0.81</td></tr><tr><td rowspan="2">ROUGE-L</td><td>Simfluence</td><td>0.013</td><td>0.079</td><td>0.751</td></tr><tr><td>Ours</td><td>0.009</td><td>0.068</td><td>0.805</td></tr></table>

Table 4: Results of test metric estimation on NLG datasets for fine-tuning.

Results are shown in Fig. 3. Unless otherwise specified, we instruction tuning the Pythia-410M for further analysis. In general, the performance of our simulator deteriorates as the number of checkpoint intervals increases. This is manifested by a rise in MSE and MAE at all steps and a drop in Spearman's ρ when the checkpoint interval is large. However, even when the number of checkpoint intervals is equal to 10, which means that we will use the training state of one point to approximate the training state of the previous ten points and the training dynamics collection time will be shortened by almost 90%, our method still has comparable prediction error at all steps and better Spearman coefficient than Simfluence.

![](images/1bb2c6e2610e3ceae555615fb95d42fd52650bc87155e28dec11dbc3790f0faa.jpg)  
(a) Loss simulation on RTE

![](images/6d91e1f115ce32e2b31643f6dc287d372f23c081364e7b98379c82ac54325d45.jpg)  
(b) Loss simulation on WMT16 DE/EN

![](images/754929ca3cc7f1d278fd64b0bd78cf8c44c6d5d16cc82eec6a79c4389e13c623.jpg)  
(c) BLEU simulation on WebNLG

![](images/9501cd5a3e18d0df27ea1300280dbcdb1ea3f8b62afd7267102b5daf058a004c.jpg)  
(d) ROUGE-L simulation on WMT16 DE/EN  
Figure 2: Illustration of loss and metric simulation on NLU and NLG tasks with different TDA methods for instruction tuning. See the §D for more examples.

Empirical Analysis of Markov Order Dependency Using the first-order Markov process to predict future states based on the prior step, potentially oversimplifies GPT training dynamics. Therefore, we consider the training dynamics as an n-th order Markov process (n = 2, 3, 5, 10) and experiment on both language understanding (RTE) and generative (WebNLG) tasks.

The result can be seen in Fig. 4. Overall, when considering more preceding training information, the simulation error initially increases and decreases for both datasets, as indicated by the allsteps MSE metric. It suggests that a high order n might introduce noise, leading to a degraded simulator's performance. Moreover, the final-step Spearman's ρ shows a significant increase from 0.746 to 0.785 for RTE with the increase of order n, but not the same for WebNLG. We guess considering more past training information could improve the prediction accuracy for NLU tasks.

Impact of Different Feature Representations To further explore the impact of various feature representations, we conducted experiments on two types of pre-trained encoders: BERT 1 and Pythia 2 with different sizes. Results are shown in Fig. 5. In general, BERT's feature representations produce better simulation results than the Pythia encoder. This could be due to its ability to encode context information in both directions. Interestingly, we also found that increasing the parameters of the Pythia encoder does not always lead to better performance of the performance simulator.

## 4.5 Analysis

Robustness across Varying Model Sizes We conducted experiments to validate how our simulator handles the complexity of GPTs of different sizes, ranging from 14M to 2.8B, specifically focusing on instruction tuning scenarios. Results are presented in Fig. 6. Our loss simulation experiments revealed that despite the inconsistent simulation performance trend with increasing GPT size, our featurized model consistently surpassed Simfluence. These findings demonstrate the superiority of our model in effectively capturing and managing model complexity.

![](images/0c6e6069d2ba5322c626fd5aa59ae60e6ccd901654ae20f165a48d52b50f198e.jpg)  
Figure 3: Variation curves of the average performance of GPTfluence for loss simulation in five datasets when different checkpoint intervals are selected.

![](images/6cde061bfbb0c13d09e6885b5b324ed143661e44d0f142db40ab8f610326b89d.jpg)

![](images/f3c40ef28147c7d45f98e0a7fa4887c9d4b8b64497396d92a80b3817f087d8b9.jpg)  
(a) RTE

![](images/fb36d68d1e4d353d0a2c5b1b3e21906ad6b483a02776765b5a25ad9d728e6ed1.jpg)

![](images/2285d33ab45dde6e4a03cb15d04d5edaafc0e4aef61408ceaea41963ce60d28b.jpg)

![](images/cb7510f7551d96da41db8f579d9c5a11de7078c1c9ab9dbb59e74303b58850bf.jpg)  
(b) WebNLG

![](images/28d909a8b636db7afb09cf5a7f29da4a215021d45372ae4acfa2cc407c4fed42.jpg)  
Figure 4: Analysis on the impact of n-th order Markov process on language understanding (RTE) and generation (WebNLG) tasks, varying n from 1 to 10.

Unseen Data Generalization Unlike Simfluence, which restricts the parameters only indexed by seen samples of past training runs, our GPTfluence can handle unseen samples via sample parameterization. We conducted experiments on RTE and WebNLG tasks in fine-tuning scenarios to further verify the unseen data generalization. For a future training run, we experiment in three different unseen data scenarios: 1) Examples in the training curriculum are unseen; 2) Test examples are unseen; 3) Both examples in training the curriculum and test examples are unseen.

We defer the results in Table 11 in Appendix.

![](images/2ec6a1d17e50e646b6743d9bc02f01f0113c2e5e54cda236f47d846f10bd13a3.jpg)

![](images/726ef8fa0e2366e316b3b9d95083634335d68d0c7800080849c8de7f36fab032.jpg)  
Figure 5: Impact of feature representation of different pre-trained encoders on loss simulation.

![](images/c0609a80c0bb209f81448f4d84df9b23ef6fca9e8517f67150e989f77b9de9fb.jpg)

![](images/4a3c42ef503b9f228b74b844d63c6278bda4e28805a5bab67df7ee8a150cc2a5.jpg)

![](images/34d98fef8a684fd12b88be019ce3848527455c928ecb0b7f2625eb8dc6580f1c.jpg)  
Figure 6: Comparison of the loss simulation between GPTfluence and Simfluence on instruction tuning Pythia model series, ranging from 14M to 2.8B.

Overall, GPTfluence can generalize to unseen data, which includes simulating loss and performance metrics. What's more, we find that GPTfluence is better at generalizing to unseen training data to simulate the impact of test samples that have been seen in the past. To illustrate this more visually, we show the effect of GPTfluence's simulation of the unseen training data setting with loss and performance metrics, respectively. As shown in Fig. 8, the generalization performance of GPTfluence is mostly satisfactory.

## 4.6 Use Case: Mislabelled Data Identification

Following previous studies (Yeh et al., 2018; Pruthi et al., 2020), we present a mislabeled data identification use case to evaluate our TDA-based method.

Experimental Setup We employ the Pythia-410M model as our classifier and utilize a subset of the SST-2 dataset. The methods compared include the following: Random, where we bypass influence calculation and apply random shuffling³. TracIn-CP, which uses self-influence as the metric by computing the gradient dot-product between a sample and itself. Similarly, GPTfluence calculates the influence by simulating the multiplicative factor α on the sample itself.

Results The results are depicted in Fig. 7. When examining the fraction of mislabelled data identified, GPTfluence demonstrates comparable performance to random selection, albeit slightly underperforming compared to TracIn-CP. However, the marginal difference in mislabel detection is offset by the notable improvement in test accuracy achieved with GPTfluence. Our method outperforms both TracIn-CP and random selection, particularly excelling in the early stages of mislabel detection, which is crucial when reviewing a small fraction of data. In scenarios where precision is key, especially with limited data available for review, GPTfluence proves its efficacy.

![](images/f8b5113289a768b52eab528ad359afd73b9cd59a3a1a23af3bf6250476e4beca.jpg)

![](images/3a2f2d5c5bb7845d489267aa5488c4cd2a43be655a2081b1e2a17d53b274ba5e.jpg)  
Figure 7: SST-2 Mislabelled Data Identification with GPTfluence, TracIn-CP and Random Selection.

To simulate mislabeled data, we corrupted 40% of the training set by flipping the labels, resulting in an initial classification accuracy of 0.53. We then sequentially corrected mislabelled samples by inspecting fractions of the dataset ranked by our influence metric, computed via the TDA method. After correcting the mislabels, we retrained the classifier and reported the test accuracy on the cleaned dataset.

## 5 Related Work

Our methodology extends the frontier of TDA techniques, which are instrumental in understanding the influence of individual training instances on model predictions. This body of work bifurcates into two main strands: gradient-based approximation methods and simulation-based approaches.

Gradient-Based Approximation Methods This strand of research capitalizes on gradient information to infer the influence of training instances on model predictions, providing a quantifiable measure of individual data points’ contributions (Koh and Liang, 2017; Yeh et al., 2018; K and Søgaard, 2021). Influence Functions, a pioneering method in this domain, leverages the mathematical framework of influence functions for estimating the impact of dataset perturbations on model predictions. Complementing this, TracIn (Pruthi et al., 2020) employs gradient-based approximations to trace the influence of training data on test predictions. Similarly, Grad-Dot (Charpiat et al., 2019) uses gradient dot products to approximate the influence of training examples. A contemporary work (Xia et al., 2024) that adapts the TracIn framework for models optimized with Adam. LESS incorporates LoRA (Hu et al., 2021) and random projection (Park et al., 2023) techniques to enhance data selection processes. These methods primarily rely on gradients to quantify data influence, offering tractable solutions with varying degrees of approximation accuracy.

Simulation-Based Approaches An alternative research vein adopts model-based simulations to represent training dynamics (Ilyas et al., 2022; Guu et al., 2023). Simfluence (Guu et al., 2023) pioneers the simulation-based category by learning a linear model that predicts the influence of training examples through multiplicative and additive factors, as detailed in §2. Recent efforts (Engstrom et al., 2024) have focused on simulating the overall influence of training examples, aiming at predicting the cumulative influence of training data for refined data selection.

Our contribution distinctly advances the simulation-based direction by forecasting the end-point influence and modeling the entire trajectory of training dynamics using featurized representations. This approach provides a more in-depth understanding of training data influence, facilitating dynamic adjustments and insights into the model training curricula.

## 6 Conclusion and Future Work

In this paper, we explore the data attribution analysis for GPT models through GPTfluence, a novel featurized simulator approach. This methodology not only surpasses the predictive capabilities of traditional test loss metrics but forecasts essential task performance metrics across a broad spectrum of GPT model sizes, ranging from 14M to 2.8B parameters. Our comprehensive evaluations across diverse downstream tasks and fine-tuning scenarios substantiate the superior efficacy of our approach. In the future, extending this approach to other tasks and training regime presents a promising avenue for future research.

## Acknowledgements

We would like to thank all anonymous reviewers for their insightful and constructive feedback. Qiwei Peng is supported by DisAI - Improving scientific excellence and creativity in combating disinformation with artificial intelligence and language technologies, a project funded by European Union under the Horizon Europe, GA No. 101079164.

## Ethical Consideration

While our study focuses on predicting the influence of training data on GPT models, we recognize the broader ethical implications that our research may entail, especially as it contributes to the advancement of large language models (LLMs) that are increasingly integrated into societal functions.

Data Use and Privacy Our research utilizes publicly available datasets and respects privacy concerns by anonymizing any potentially identifiable information. We ensure that our data handling practices comply with all relevant data protection regulations and ethical guidelines, safeguarding against misuse.

Potential Misuse We are cognizant of the potential misuse of predictive models in manipulating or unfairly influencing AI systems. Our research aims to contribute to the understanding and mitigation of such risks by providing tools to analyze and adjust the influence of training data. We encourage the application of our findings in ethical ways that promote fairness and transparency in AI.

Broader Impact This study advances understanding of data influence on LLMs, offering a methodological approach for detailed impact analysis. This work not only enhances the interpretability and transparency of LLMs but also lays the groundwork for more informed and ethical decisions in data curation and model training.

## Limitations

This work introduces a novel feature-based approach within the simulation-based framework for predicting the influence of training data on GPT models. While our methodology represents a significant advancement in the field, it is not without its limitations, which we discuss below:

Dependence on Extensive Training Dynamics A fundamental constraint of our approach is its reliance on a comprehensive set of training dynamics to train the simulator effectively. This requirement, while crucial for the accuracy of our predictions, necessitates considerable computational resources and time. The efficiency of data influence simulators remains an area ripe for further exploration, with the aim of reducing the computational overhead without compromising on performance.

Limited Dataset Scope Our experimental validation is confined to a subset of the FLAN datasets, constrained by the logistical and computational costs associated with collecting a large-scale training dynamics dataset. Despite this limitation, we have conducted over 352 training experiments across six different GPT model sizes (ranging from 14M to 2.8B parameters) to amass the GPTDynamics dataset. This dataset, which we are making publicly available, is a step towards mitigating the data scarcity in this research area, yet the need for more expansive datasets encompassing a broader range of tasks and languages remains.

Model Size Constraints The high computational costs involved in executing multiple runs on larger language models, such as those with 13B or even 72B parameters, have limited the scale of the models we could feasibly include in our study. While our findings are robust across the examined model sizes, extending our analysis to larger models with hundreds of billions of parameters would likely yield additional insights into the scalability and generalizability of our approach.

Generalization to Other Domains While our study focuses on GPT models and a specific subset of datasets, the generalizability of our approach to other model architectures and domains is not fully explored. Future work could extend our methodology to different types of language models and beyond, including vision and multimodal systems, to assess the applicability and adaptability of our featurized simulation-based approach.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar van der Wal. 2023. Pythia: A suite for analyzing large language models across training and scaling.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Yekun Chai, Shuohuan Wang, Chao Pang, Yu Sun, Hao Tian, and Hua Wu. 2023. Ernie-code: Beyond english-centric cross-lingual pretraining for programming languages. In Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, pages 10628–10650. Association for Computational Linguistics.

Guillaume Charpiat, Nicolas Girard, Loris Felardos, and Yuliya Tarabalka. 2019. Input similarity from the neural network perspective. Advances in Neural Information Processing Systems, 32.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Logan Engstrom, Axel Feldmann, and Aleksander Madry. 2024. Dsdm: Model-aware dataset selection with datamodels. arXiv preprint arXiv:2401.12926.

Kelvin Guu, Albert Webson, Ellie Pavlick, Lucas Dixon, Ian Tenney, and Tolga Bolukbasi. 2023. Simfluence: Modeling the influence of individual training examples by simulating training runs. arXiv preprint arXiv:2303.08114.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, and Aleksander Madry. 2022. Datamodels: Predicting predictions from training data. arXiv preprint arXiv:2202.00622.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Karthikeyan K and Anders Søgaard. 2021. Revisiting methods for finding influential examples.

Pang Wei Koh and Percy Liang. 2017. Understanding black-box predictions via influence functions. In International conference on machine learning, pages 1885–1894. PMLR.

Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81.

Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. 2024. Starcoder 2 and the stack v2: The next generation. arXiv preprint arXiv:2402.19173.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318.

Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, and Aleksander Madry. 2023. Trak: Attributing model behavior at scale. arXiv preprint arXiv:2303.14186.

Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. 2020. Estimating training data influence by tracing gradient descent. Advances in Neural Information Processing Systems, 33:19920–19930.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAl blog, 1(8):9.

Gemini Team, Rohan Anil, Sebastian Borgeaud Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. 2022. Finetuned language models are zero-shot learners.

Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. 2024. Less: Selecting influential data for targeted instruction tuning. arXiv preprint arXiv:2402.04333.

Chih-Kuan Yeh, Joon Kim, Ian En-Hsu Yen, and Pradeep K Ravikumar. 2018. Representer point selection for explaining deep neural networks. Advances in neural information processing systems, 31.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. 2022. Opt: Open pretrained transformer language models.

## A Implementation Details

## A.1 Tasks and Datasets for GPTDynamics

We conduct experiments on a subset of FLAN (Wei et al., 2022), a diverse array of datasets for instruction tuning, to conduct a thorough evaluation of TDA methods. Our dataset selection spans both NLU and NLG tasks, thereby offering a broad spectrum of challenges for TDA methods to tackle.

The NLU tasks selected include RTE (Natural Language Inference), SST-2 (Sentiment Classification), and BoolQ (Reading Comprehension). For NLG, we delve into WebNLG (Struct-to-Text) and WMT-16 DE/EN (Machine Translation) tasks.

To exploit the superior generalization benefits that instruction tuning brings to language models, we have assembled a specialized subset for instruction fine-tuning. This subset amalgamates the previously mentioned five tasks with CNN-DM (Summarization), crafting an extensive testing environment of FLAN data. We sourced task-specific instructions directly from the original FLAN paper.

## A.2 Comparison Baselines

TracIn (Pruthi et al., 2020) is a gradient-based used to calculate the influence through a first-order gradient approximation. It considers the influence of the training example z on the test example $z ^ { \prime }$ as a loss change in $z ^ { \prime } ,$ which is provided by each gradient step of the training example z. In practice, TracInCP was proposed as an alternative approximation that considers specific checkpoints during training. TracInCP calculates the gradient dot product of z and $z ^ { \prime }$ at these checkpoints. In our experiments, we used TracInCP with 10 checkpoints and all steps’ checkpoints to estimate the influence.

Grad-Dot (Charpiat et al., 2019) is a heuristic gradient-based TDA method. They also compute the effect of a training sample on a test sample by the dot product of the gradients but computed on top of the final trained model.

Simfluence (Guu et al., 2023) is a novel framework for TDA. It characterizes the loss variation of test samples during training by modeling it as a Markov process. Then, it learns a unique multiplicative and additive influence parameter for each training example. It is worth noting that in the original paper, the framework that considers both multiplicative and additive influences is referred to as Simfluence-linear. However, for simplicity in this paper, we use the term Simfluence to refer to the same model.

## A.3 Implementation Details of Instruction Tuning

GPTDynamics Collection for Instruction Tuning We instruction tuned Pythia from 14M to 2.8B (i.e., 14M, 70M, 160M, 410M, 1B, and 2.8B) on the instruction tuning dataset referenced in Appendix A.1. We collect a total of randomly sampled 768 instances from aforementioned five tasks, with each samples 128 of 200 data points in one training run for instruction tuning. The data division followed the same protocol as in the finetuning scenarios. All Pythia models underwent comprehensive fine-tuning, with the exception of the Pythia-2.8B model, which was fine-tuned using the parameter-efficient LoRA technique (Hu et al., 2021). The LoRA module was implemented within the query, key, and value projection matrices of the self-attention module, with a LoRA rank of 8, alpha set to 4, and a dropout probability of 0.05. We evaluated the Pythia models using the identical datasets as those in the fine-tuning experiments. For the WebNLG and WMT16 DE/EN datasets, we evaluated BLEU and ROUGE-L scores in addition to test loss, employing a top-p sampling strategy for generation with a temperature of 0.2 and topp probability of 0.95. Detailed instruction-tuning hyperparameters are reported in Table 5.

GPTfluence Training Setup The architecture of our simulator is a pre-trained sentence encoder followed by parallel weight-sharing fully-connected layers for predicting influence factors. The trainable model size of the simulator is 11.4M excluding pre-trained embeddings (frozen). Unless specified, we use the sentence transformer⁴ as our pre-trained encoder. For the simulator training, we combine all five FLAN datasets and train our simulator in a multi-task manner, each dataset has 27 training runs. All reported results are averaged over 5 heldout runs. We set the order n of Markov process assumptions equal to 1 for instruction tuning. Detailed training hyperparameters of GPTfluence are shown in Table 6.

## A.4 Implementation Details of Fine-Tuning

GPTDynamics Collection for Fine-Tuning All the experiments are conducted on the NVIDIA Tesla V100 GPUs unless specified. We fine-tune Pythia-410M on five datasets: SST-2, BoolQ, RTE, WebNLG, and WMT16 DE/EN. For each dataset, we perform a total of 32 training runs, with each sample 128 of 200 data points from the original training set for GPT training. The split of training runs is divided into 25 for training, 2 for validation, and 5 for test. All reported results are averaged over 5 held-out runs. For NLG datasets, we measure BLEU, ROUGE-L scores besides the test loss, using a top-p sampling strategy for generation with a temperature setting of 0.2 and a top-p probability of 0.95. Note that we collect ROUGE-L scores on a scale from 0 to 1. The fine-tuning hyperparameters are shown in Table 7.

<table><tr><td>Instruction-Tuning Hyperparameters</td><td>Pythia-14M</td><td>Pythia-70M</td><td>Pythia-160M</td><td>Pythia-410M</td><td>Pythia-1B</td><td>Pythia-2.8B</td></tr><tr><td>Optimizer Adam&#x27;sβ</td><td></td><td></td><td></td><td>AdamW (0.9, 0.999)</td><td></td><td></td></tr><tr><td>Adam&#x27;s €</td><td></td><td></td><td></td><td>1e-6</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Weight decay</td><td>5e-7</td><td>5e-7</td><td></td><td>0.001</td><td></td><td></td></tr><tr><td>Learning rate</td><td></td><td></td><td>5e-7</td><td>2e-7</td><td>2e-7</td><td>1e-5</td></tr><tr><td>Learning rate schedule</td><td></td><td></td><td></td><td>Linear decay</td><td></td><td></td></tr><tr><td>Warmup steps</td><td></td><td></td><td></td><td>0</td><td></td><td></td></tr><tr><td>Batch size</td><td></td><td></td><td></td><td>8</td><td></td><td></td></tr><tr><td>Max sequence length</td><td>2048</td><td>2048</td><td>2048</td><td>2048</td><td>2048</td><td>1024</td></tr><tr><td>Training epochs</td><td>3</td><td>3</td><td>3</td><td>3</td><td>2</td><td>2</td></tr><tr><td>Training steps</td><td>288</td><td>288</td><td>288</td><td>288</td><td>192</td><td>192</td></tr><tr><td>Precision</td><td></td><td></td><td></td><td>fp32</td><td></td><td></td></tr></table>

Table 5: Hyper-parameter settings for instruction tuning GPTDynamics data across Pythia models, ranging in size from 14M to 2.8B.
<table><tr><td>Hyperparameters</td><td>Pythia-14M</td><td>Pythia-70M</td><td>Pythia-160M</td><td>Pythia-410M</td><td>Pythia-1B</td><td>Pythia-2.8B</td></tr><tr><td>L2 regularizaiton λ Optimizer Adam&#x27;sβ Adam&#x27;s € Learning rate</td><td></td><td></td><td>1e-6</td><td>1e-5 AdamW (0.9, 0.999) 1e-8 1e-5</td><td></td><td>1e-5</td></tr><tr><td>Learning rate schedule Warmup steps Batch size Max training epochs</td><td>50</td><td>50</td><td>50</td><td>Linear decay 200 128 50</td><td>50</td><td>50</td></tr><tr><td>Pre-trained encoder Max sequence length Early stopping</td><td>512</td><td>512</td><td>512</td><td>MiniLM-L6-v2 512</td><td>512</td><td>512</td></tr><tr><td>Precision Seed</td><td></td><td></td><td></td><td>√ fp32</td><td></td><td></td></tr></table>

Table 6: Hyperparameters of training our featurized simulator for instruction tuning on Pythia models of size from 14M to 2.8B. We use the same training hyperparameters as in the loss simulation for the BLEU and ROUGE-L score simulation on WebNLG and WMT16 DE/EN datasets.

GPTfluence Training Setup We train a single featurized simulator on training runs for each dataset with the L2-regularized regression objective as defined in section 3.2. We freeze the parameters of the pre-trained encoder during training for better generalization. We set the order n of Markov process assumptions equal to 1 for fine-tuning. Detailed training hyperparameters are shown in Table 8.

## A.5 Implementing GPTfluence

GPTfluence Training To elucidate the intricate process of collecting training data dynamics and the training of the featurized simulator with GPTfluence, we present the pseudo-code in Algorithm 1. The execution of this algorithm yields a GPTfluence simulator, which is adept at simulating the target performance trajectory and assessing the impact of training examples on a given test point.

GPTfluence Evaluation For evaluation, The simulator autoregressively forecasts upcoming testset metrics, based on the previous n observations. Specifically, it commences with the initial test metric recorded at the starting step, thereafter predicting the subsequent performance metrics across the training curriculum.

## B Experiment Results

In this section, we provide additional experimental results and detailed descriptions to complement the

<table><tr><td>Fine-Tuning Hyperparameters</td><td>SST-2</td><td>RTE</td><td>BoolQ</td><td>WebNLG</td><td>WMT16 DE/EN</td></tr><tr><td>Optimizer Adamβ Adam € Weight decay</td><td rowspan="6"></td><td rowspan="6"></td><td rowspan="6">AdamW (0.9, 0.999)</td><td rowspan="6"></td><td rowspan="6"></td><td rowspan="6"></td></tr><tr><td>1e-6 0.001</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>5e-7</td></tr><tr><td>Learning rate Learning rate schedule</td><td>5e-7</td><td></td><td>5e-7 1e-6</td></tr><tr><td></td><td></td><td></td><td>Linear decay</td><td>5e-7</td></tr><tr><td>Warmup steps</td><td></td><td></td><td>0</td><td></td></tr><tr><td>Batch size</td><td></td><td></td><td>4</td><td></td></tr><tr><td>Max sequence length</td><td></td><td></td><td>2048</td><td></td></tr><tr><td>Training epochs</td><td></td><td></td><td>3</td><td></td><td></td></tr><tr><td>Training steps</td><td></td><td></td><td>96</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>fp32</td><td></td><td></td></tr><tr><td>Precision</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 7: Fine-tuning hyper-parameter settings of GPTDynamcis for various tasks.
<table><tr><td>Hyperparameters</td><td>SST-2</td><td>RTE</td><td>BoolQ</td><td>WebNLG</td><td>WMT16 DE/EN</td></tr><tr><td>L2-regularizaiton&#x27;s λ Optimizer Adam&#x27;s β Adam&#x27;s €</td><td></td><td></td><td>1e-5 AdamW (0.9, 0.999)</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>200 128</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Max training epochs</td><td>300</td><td>300</td><td>300</td><td>300</td><td>300</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Pre-trained encoder</td><td>MiniLM-L6-v2</td><td>MiniLM-L6-v2</td><td>MiniLM-L6-v2</td><td>MiniLM-L6-v2</td><td>MiniLM-L6-v2</td></tr><tr><td>Max sequence length</td><td>512</td><td>512</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>512</td><td>512</td><td>512</td></tr><tr><td>Early stopping</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>√</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Precision</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Seed</td><td></td><td></td><td>fp32</td><td></td><td></td></tr></table>

Table 8: Hyperparameters of training our featurized simulator for each dataset for fine-tuning. We use the same training hyperparameters as in the loss simulation for the BLEU and ROUGE-L score simulation on WebNLG and WMT16 DE/EN datasets.

<table><tr><td colspan="2">Hyperparameters</td></tr><tr><td>L2 regularizaiton λ</td><td>1e-5</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Adam&#x27;s β</td><td>(0.9, 0.999)</td></tr><tr><td>Adam&#x27;s €</td><td>1e-8</td></tr><tr><td>Learning rate</td><td>1e-3</td></tr><tr><td>Learning rate schedule</td><td>Linear decay</td></tr><tr><td>Warmup steps</td><td>200</td></tr><tr><td>Batch size</td><td>128</td></tr><tr><td>Max training epochs</td><td>300</td></tr><tr><td>Early stopping</td><td>√</td></tr><tr><td>Precision</td><td>fp32</td></tr><tr><td>Seed</td><td>42</td></tr></table>

Table 9: Training hyperparameters of Simfluence for fine-tuning. It is noted that we use the same hyperparameters for both loss and metric simulation, as we see that different hyperparameters has little effect on Simfluence's performance.

main findings.

## B.1 Empirical Analysis on Markov Property

Table 10 presents a comprehensive results of how the order of the Markov process influences test loss, BLEU, and ROUGE-L metrics during instruction tuning simulations.

## B.2 Unseen Data Generalization

We offer in-depth simulations of loss and performance metrics across scenarios involving unseen training data, unseen test data, and both unseen training and test data. Simulation results for finetuning are detailed in Table 11, while those for instruction-tuning can be found in Table 12. Illustration examples are shown in the Fig. 8.

## C Computational Complexity

We conducted a comparison of inference latency and floating point operations (FLOPs) among various TDA methods. Results are presented in Table 13. TracIn-CP, a representative of gradientbased methods, exhibited the highest inference latency and FLOPs. This is attributable to the need to do forward and backward operations directly on the GPTs. Conversely, GPTfluence solely depends on a considerably smaller simulator during inference.

Algorithm 1 GPTfluence Training Procedure   
Input: Language modeling task P, pre-trained   
GPT θ, Target sample $z ^ { \prime } ,$ Dataset D, Subset size   
I, Target metric $\phi ,$ Training dynamic $D _ { r u n } .$ Multi  
plicative factor function $\alpha ( \cdot )$ , Additive factor func  
tion $\beta ( \cdot )$ , L2 regularization weight, Featurized sim  
ulator Θ, Markov order n-th   
Output: Simulator $\hat { \Theta }$   
1: Initialize $\mathcal { D } _ { r u n }$ with an empty set   
2: for $k = 1$ to $K$ do   
3: Sample a subset $D ^ { \prime } \subset D$ of size I   
4: for Sample batch $c _ { t } \in D ^ { \prime }$ do   
5: Update $\theta _ { t }$ using P based on $c _ { t }$   
6: Calculate target metric $y _ { t } ^ { k } =$   
$\phi ( \theta _ { t } , z ^ { \prime } )$   
7: Add $c _ { t }$ and $y _ { t } ^ { k }$ into $D _ { r u n }$   
8: end for   
9: end for   
10: Initialize $g _ { \theta }$ with pre-trained encoder   
11: while not converged do   
12: Sample a mini-batch $B _ { t r a i n } , B _ { t e s t }$ from   
$D _ { r u n }$   
13: for each $z _ { i } \in B _ { t r a i n }$ do   
14: Compute multiplicative and additive in  
fluences $A _ { i , 1 : n } , B _ { i }$   
15: end for   
16: $\alpha = \{ \alpha _ { j } ( B _ { t r a i n } ) | j = 1 , 2 , . . . , n \}$   
17: $\beta = \beta ( B _ { t r a i n } )$   
18: Update Θ with $\alpha , \beta , \gamma$   
19: end while   
20: return $\hat { \Theta }$

Furthermore, we analyzed the convergence and validation performance of our GPTfluence in comparison with Simfluence. As shown in Fig. 9, GPTfluence exhibits a better convergence efficiency and also has lower validation all-steps MSE. This underscores the better training efficiency and model capacity of our featurized simulator.

## D Qualitative Examples

In this section, we provide additional quantitative examples, including loss and metric simulations, for a comparison. This includes experimental re-

```csv
Task Order All-steps All-steps Final-step Spear-
MSE (↓) MAE (↓) man's ρ(↑)
RTE 1 0.036(0.029) 0.151(0.060) 0.746(0.095)
2 0.036(0.029) 0.151(0.060) 0.747(0.094)
3 0.036(0.030) 0.149(0.062) 0.750(0.094)
5 0.037(0.032) 0.147(0.067) 0.757(0.093)
10 0.036(0.032) 0.150(0.071) 0.785(0.088)
Task Order All-steps All-steps Final-step Spear-
MSE (↓) MAE (↓) man's ρ(↑)
WEBNLG 1 0.011(0.014) 0.078(0.043) 0.985(0.002)
2 0.010(0.011) 0.072(0.039) 0.986(0.002)
3 0.011(0.012) 0.073(0.040) 0.986(0.002)
5 0.012(0.014) 0.082(0.044) 0.983(0.003)
10 0.012(0.014) 0.082(0.044) 0.983(0.002)
BLEU
Task Order All-steps All-steps Final-step Spear-
MSE (↓) MAE (↓) man's ρ (↑)
WEBNLG 1 43.98(81.40) 4.28(3.57) 0.80(0.01)
2 43.31(80.70) 4.24(3.54) 0.80(0.03)
3 43.67(81.77) 4.24(3.57) 0.80(0.02)
5 44.79(76.57) 4.39(3.49) 0.78(0.02)
10 47.83(99.06) 4.35(3.72) 0.74(0.03)
Task Order ROUGE-L
All-steps All-steps Final-step Spear-
MSE (↓) MAE (↓) man's ρ (↑)
WEBNLG 1 0.01(0.01) 0.06(0.03) 0.77(0.04)
2 0.01(0.01) 0.06(0.03) 0.76(0.04)
3 0.01(0.01) 0.06(0.03) 0.76(0.03)
5 0.01(0.01) 0.06(0.03) 0.77(0.04)
10 0.01(0.01) 0.06(0.03) 0.78(0.03)
```  
Table 10: Impact of the Markov process order on test loss, BLEU, and ROUGE-L metrics in instruction tuning simulations.

sults across various training scenarios and the use of unseen data, among others.

## D.1 Simulation For Instruction-Tuning

We provide additional qualitative examples for instruction-tuning simulations, highlighting test loss and performance metrics:

• Simulation of test loss for Pythia-410M is shown in Fig. 10.

• Simulation of test loss for Pythia-1B is depicted in Fig. 11.

• BLEU metric simulation for Pythia-410M can be found in Fig. 12.

• BLEU metric simulation for Pythia-1B is illustrated in Fig. 13.

• ROUGE-L metric simulation with Pythia-410M is presented in Fig. 14.

• ROUGE-L metric simulation with Pythia-1B is detailed in Fig. 15.

## D.2 Simulation For Fine-Tuning

We provide additional qualitative examples showcasing simulations of test loss and performance metrics for fine-tuning, as follows:

• For test loss simulation, see Fig. 16.

• For BLEU metric simulation, refer to Fig. 17.

• For ROUGE-L metric simulation, see Fig. 18.

![](images/1d0b2a6c6b8bbc8a98a07488c7a38276bcdd0e0f6c5b93f2a57caf2fc78eac01.jpg)  
(a) RTE

![](images/90a4cc4934f612db6c5c09e3d055d7ac8c2a798369f49df17829800f4c0ee4df.jpg)  
(b) WebNLG

Figure 8: Illustration of simulation results on unseen training data. The top shows the loss simulation on RTE, while the bottom shows the BLEU metric simulation for WebNLG. Additional qualitative examples for different settings and metrics are provided in § D.3.
<table><tr><td>Method</td><td>Latency (sec/sample)</td><td>FLOPs</td></tr><tr><td>TracIn-CP</td><td>153.0</td><td> $1 . 1 \times 1 0 ^ { 1 3 }$ </td></tr><tr><td>Simfluence</td><td>0.1</td><td> $1 . 6 \times 1 0 ^ { 1 }$ </td></tr><tr><td>Ours</td><td>0.2</td><td> $5 . 3 \times 1 0 ^ { 6 }$ </td></tr></table>

Table 13: Inference latency and FLOPs of GPTfluence, Simfluence, and TracIn-CP.

![](images/a93b5fab83d82efd8e1c5fc3352e4a72804f2795f44386103adeb68f5f317db4.jpg)

![](images/86a2bdec0737536fc4ebd28053ff7a73d48feb9522a43da85ceb5122d62b6db2.jpg)  
Figure 9: Comparison of our method and Simfluence with respect to training loss (Left) and validation allsteps MSE (Right).

## D.3 Simuation with Unseen Data

We provide detailed simulations of test loss and performance metrics across different tasks and scenarios, as detailed below:

• For the RTE task, test loss simulations under various conditions are presented in Fig. 19 (unseen test data), Fig. 20 (unseen training data), and Fig. 21 (unseen training and test data).

• For the WebNLG task, test loss simulations are shown in Fig. 22 (unseen test data), Fig. 23 (unseen training data), and Fig. 24 (unseen training and test data).

<table><tr><td>Task</td><td>Metrics</td><td>Training Data Unseen</td><td>Test Data Unseen</td><td>All-steps MSE</td><td>All-steps MAE</td><td>Final-Step Spearman&#x27;s ρ</td></tr><tr><td rowspan="3">RTE</td><td rowspan="3">Loss</td><td>√</td><td>x</td><td>0.346(0.281)</td><td>0.513(0.211)</td><td>0.913(0.052)</td></tr><tr><td>x</td><td>√</td><td>0.351(0.489)</td><td>0.444(0.325)</td><td>-0.024(0.050)</td></tr><tr><td>√</td><td>√</td><td>0.984(4.569)</td><td>0.568(0.728)</td><td>-0.048(0.045)</td></tr><tr><td rowspan="9">WEBNLG</td><td rowspan="3">Loss</td><td>√</td><td>x</td><td>1.251(0.962)</td><td>1.003(0.413)</td><td>0.892(0.011)</td></tr><tr><td>x</td><td>√</td><td>0.403(0.575)</td><td>0.476(0.476)</td><td>0.123(0.019)</td></tr><tr><td>√</td><td>√</td><td>0.886(2.112)</td><td>0.699(0.549)</td><td>0.190(0.013)</td></tr><tr><td rowspan="3">BLEU</td><td>√</td><td>x</td><td>94.99(273.96)</td><td>6.13(5.72)</td><td>0.51(0.02)</td></tr><tr><td></td><td></td><td>106.19(150.51)</td><td>7.14(5.02)</td><td>0.18(0.08)</td></tr><tr><td>x √</td><td>√</td><td>153.63(219.29)</td><td>8.66(6.39)</td><td>0.15(0.01)</td></tr><tr><td rowspan="3">ROUGE-L</td><td>√</td><td>x</td><td>0.008(0.009)</td><td>0.069(0.036)</td><td>0.578(0.062)</td></tr><tr><td></td><td>√</td><td>0.009(0.008)</td><td>0.073(0.034)</td><td>0.288(0.049)</td></tr><tr><td>x √</td><td>√</td><td>0.010(0.010)</td><td>0.075(0.039)</td><td>0.168(0.091)</td></tr></table>

Table 11: Results of loss and metric simulation on unseen data for RTE and WebNLG Datasets for finetuning.
<table><tr><td>Task</td><td>Metrics</td><td>Training Set OOD</td><td>Test Set OOD</td><td>All-steps MSE (↓)</td><td>All-steps MAE (↓)</td><td>Final-step Spear- man&#x27;s ρ (↑)</td></tr><tr><td rowspan="3">RTE</td><td rowspan="3">Loss</td><td>√</td><td>x</td><td>0.781(0.793)</td><td>0.730(0.419)</td><td>-0.082(0.214)</td></tr><tr><td>x</td><td>√</td><td>1.137(2.927)</td><td>0.725(0.619)</td><td>-0.011(0.033)</td></tr><tr><td>√</td><td>√</td><td>1.110(1.057)</td><td>0.888(0.482)</td><td>-0.047(0.062)</td></tr><tr><td rowspan="9">WEBNLG</td><td rowspan="3">Loss</td><td>√</td><td>x</td><td>2.398(1.722)</td><td>1.435(0.508)</td><td>0.358(0.006)</td></tr><tr><td>x</td><td>√</td><td></td><td>22.627(203.637)1.530(3.062)</td><td>0.247(0.008)</td></tr><tr><td>√</td><td>√</td><td>2.708(1.415)</td><td>1.580(0.432)</td><td>0.072(0.003)</td></tr><tr><td rowspan="3">BLEU</td><td>√</td><td>X</td><td></td><td>200.50(270.92) 10.82(7.08)</td><td>0.34(0.06)</td></tr><tr><td>x</td><td>√</td><td>115.19(188.30) 7.33(5.69)</td><td></td><td>-0.03(0.03)</td></tr><tr><td>√</td><td>√</td><td></td><td>329.61(369.14) 14.20(8.11)</td><td>0.10(0.04)</td></tr><tr><td rowspan="3">ROUGE-L</td><td>√</td><td>X</td><td>0.12(0.06)</td><td>0.33(0.08)</td><td>0.35(0.02)</td></tr><tr><td>x</td><td>√</td><td>0.01(0.02)</td><td>0.09(0.05)</td><td>0.06(0.06)</td></tr><tr><td>√</td><td>√</td><td>0.13(0.05)</td><td>0.35(0.06)</td><td>0.10(0.01)</td></tr></table>

Table 12: Results of loss and metric simulation on unseen data for RTE and WebNLG Datasets for instruction tuning.

• BLEU metric simulations for the WebNLG task are illustrated in Fig. 25 (unseen test data), Fig. 26 (unseen training data), and Fig. 27 (unseen training and test data).

• ROUGE-L metric simulations for the WebNLG task are depicted in Fig. 28 (unseen test data), Fig. 29 (unseen training data) and Fig. 30 (unseen training and test data).

![](images/5a100b3b340a9c9ff5133ba2ceeabcd6e5c2a4214f1c0789377fe7efdb6b8045.jpg)

![](images/63029d76494bc3712e3e7dbb52f7274ec4803f6805b10ff5bd16306d55bc1549.jpg)  
(a) BoolQ

![](images/0b9a305fc2c9d81a5a86ba572de5f41f290117f605af8cd996ec570aadcfdb4b.jpg)

![](images/e10b7ce2dcad43fa68c5ad28c63894ebd8dbd5cec2283ddd1328c102feb17cbe.jpg)

![](images/b72bc7177faa465500fe0b60272650f4c70981960cd469e625e76f2d4ef9135e.jpg)  
(b) RTE

![](images/dd9460528a1eac108e900dc21e2f165771ab711b391d57c1747b3dda60d5f232.jpg)

![](images/e9b83f6eaaeac434c8eb760578443aa7ee805d4a5a0146e552e6345c0187c8be.jpg)

![](images/18d735c88c1ce3221d55520fc3dffb083de162c6a1ec834bfb7f120340394ce4.jpg)  
(c) SST-2

![](images/f46dd0355da878ddafebcb0bcc53531d9c333d4b6375ba5aec3e45be89599737.jpg)

![](images/bcdaffd77431f0cf9a7d885e9b920fac46760e3e4c4b9865df969a331384160b.jpg)

![](images/a1a9a855d8ae6aaf047123fa330523f9658f6fb2587f21708170fcb01c209873.jpg)

![](images/3a19c3c5b57b213d7d3d8c5ea3af45ff8f7fa791b94696473b2c90231ab9c5de.jpg)  
(d) WebNLG

![](images/d2609837b4bef5f7147bbe1a9d8e611a8710b65c5c4ba2c51c5fcb35da5f1a03.jpg)

![](images/de0a06456bace7215ca63af8c611cfd490ba85e5d34af6e62a4882b80cd2e644.jpg)  
(e) WMT16 DE/EN

![](images/3c34a7918215916854820923e722ac7989fe9e837ba8d1a190c40562b99261cd.jpg)  
Figure 10: Comparative loss simulations for instruction tuning using GPTf1uence versus other TDA methods on Pythia-410M across the BoolQ, RTE, SST-2, WebNLG, and WMT16 DE/EN datasets.

![](images/dde0b25d8343a5d7cfac56d351678005ee9c07a822dd4ff8bdf58bb2213345c9.jpg)

![](images/35b998a68040f76a293d96ebae70c1bc510e54aee72c1069963fb0d6565d96b0.jpg)  
(a) BoolQ

![](images/5c60d38c53d0568803fe93829d88ba10f05204edbbff6f4fa9e5baf863facc4d.jpg)

![](images/1b50c817d3222d90366c0d6ee6c9c5a2920a050b596900dacdde3e9d6250994b.jpg)

![](images/4a10ffb1453607d99642427a36ab2a9866d0c21886442fe790ca39e17fac9fa9.jpg)  
(b) RTE

![](images/016a1411769dbd181bd44323d5660c32f1b8cfc428ca6c72d68ead71de1c5d3e.jpg)

![](images/5e40fb35c2c2cb7c919dc422fae53793bf59698823c1e40b29eb8a113943c882.jpg)

![](images/f2a6d33f3f49d3ce119da12114b68467b7072fe791d6e1f50444606a1f7a8a77.jpg)  
(c) SST-2

![](images/aae8cbf5fe3ee67b275427f593742455b81d629a4137367e189cf678ee3fb380.jpg)

![](images/e917e8b3a0ca7c6422d320ed01476548a3dfca1b8ad95a9de0dd121b24b08b80.jpg)

![](images/f797733cff86366a045f0becab2f66628bc323936678d79d47c53c98c8449998.jpg)

![](images/93abbc06ea89ca44aadd85653c045c0b025fed4e23f18a00591384278aaf3a78.jpg)  
(d) WebNLG

![](images/c3d7b63f6aa90ea2bdd753b36581a25c61c0588ca25aa1328a3305ee6a3acf17.jpg)

![](images/a80b93e4a1ac13c2f9e4d26951ed4e7bae2cc7727d15e6d58e28555d6741e6de.jpg)  
(e) WMT16 DE/EN

![](images/aef5cfb2146bb1171d55df93b0a99a8e0c0083f04f88083f765fbada1fbaaab4.jpg)  
Figure 11: Loss simulation comparisons between GPTfluence and alternative TDA methods for instruction tuning on Pythia-1B, across the BoolQ, RTE, SST-2, WebNLG, and WMT16 DE/EN datasets.

![](images/386b846236f0e53f6491d1ce10355c20ccf2151351aeb3b8f71a3ad7a7247df0.jpg)

![](images/950c419bab98d09e4d7070209a5f74510137ab86a1ed61da31657b73acb1b5db.jpg)  
(a) WebNLG

![](images/e59ce761455ceda2b7ad2dc652efc7cfaa6f8daf92eb46567d8ab7c3eef080ea.jpg)

![](images/8732f428cdc2f4fe62a792cb7ea7001c17790844be682176cd4879c5f4a55636.jpg)

![](images/093ac3e08a3059091e20e91f0e25fecafb56c6bd84fa48ff6558038fa97f9071.jpg)  
(b) WMT16 DE/EN

![](images/589aeb959d634fb4aa8ccab950b09501eb4c45748e0d7cf251e008b5a65eefd6.jpg)  
Figure 12: BLEU metric simulation comparisons for instruction tuning using GPTfluence versus Simfluence on Pythia-410M, across the WebNLG and WMT16 DE/EN datasets.

![](images/6979488868bccc9c18657ccc62dbc72bb160881d5dcdabf61940ba44a5b353df.jpg)

![](images/2416d03522ebe8978b878caf94f3c9b970918c12962599f5110b9bbc98dece67.jpg)  
(a) WebNLG

![](images/6b5c8543ef737dc9e46cac36c93ce1a32d2ab67c0d8e50a7d5bbdb28e3206103.jpg)

![](images/9969fad4c6052e476d64584c99794955359fb2ccd09349df2ee09341373a3ccf.jpg)

![](images/7f60f9189084a1775c296426a319b7036452ab4aee9d4037ccf9e36daea12708.jpg)  
(b) WMT16 DE/EN

![](images/6a3543f2b124f0f89c8104e461987cfdfa2a2ed6802b92357d040c466881439a.jpg)  
Figure 13: Test examples of the BLEU simulation of GPTfluence and Simfluence for instruction tuning with Pythia-1B on WebNLG and WMT16 DE/EN datasets.

![](images/2867551b595ee0f53a86eab3be0e36da591d8a9f3cfefd2bca9b14ab767555ad.jpg)

![](images/d61003af155a65a2b1d5253ed53944b026f75de91cee2c08a3f63be29756b1f5.jpg)  
(a) WebNLG

![](images/a018e850833b8ced5e68c1025500a8ef5b06c2f28f06727a7fd62286bb900731.jpg)

![](images/09fb9dbc3481843bdd8d731911674156a20379e263d3a86c370da2612b23ee16.jpg)

![](images/3fb858006e7b6247f237e7726701dce3d6605810359407df4c7da3d35a68e309.jpg)  
(b) WMT16 DE/EN

![](images/3c3e9677b1b091774bd77977c66a0c161bb2753ff5d2c9a6d6e61922deec3c2e.jpg)  
Figure 14: Test examples of the ROUGE-L simulation of GPTfluence and Simfluence for instruction tuning with Pythia-410M on WebNLG and WMT16 DE/EN datasets

![](images/bbbf330ea474fc12a83eaba9f10148cfa1021588eb5a6a7e2660fe316b216acd.jpg)

![](images/0afd9d6179d390377d3df25916b421142910290a0bffbe37fa637f6ee3e7604c.jpg)  
(a) WebNLG

![](images/4869ad69e3be8da85bd766e40a422743d1bfe3aeef6f1a5bc3ed0f6fa1432d69.jpg)

![](images/af8e807c2b0712b500c60c3c5f743146e9d555c2cc8501a9fb4a8153f307b2d1.jpg)

![](images/7932a44a693ff9424ea41133fac6880a122bc0a509cef5a38f998d0a578f4ddf.jpg)  
(b) WMT16 DE/EN

![](images/994415d86d3cad5e45b20b4607a44d1e4e8877bc849e1ff02fc3b56bcc1be213.jpg)  
Figure 15: Test examples of the ROUGE-L simulation of GPTfluence and Simfluence for instruction tuning with Pythia-1B on WebNLG and WMT16 DE/EN datasets.

![](images/d2310aeb9623046b38c84178ddee8361b3a6030517571c1fae18a026845a37e8.jpg)

![](images/c1bd8a76ebce8a1aca6abe667b5b63f9fe25ada0d53d1ef18ad4211fe1cddf6c.jpg)  
(a) BoolQ

![](images/5bac606c071d99e5d1b51d45356f6d641a533ef2c62de85518516dbb0590df95.jpg)

![](images/486ff5092989457e3e90a459fd4cc04fcae74a28b5ae45c091f2aa0b400bdb45.jpg)

![](images/04488bcb1eaf8ae92c9e59540e73d00f453d97628ad9d76fde7de2e8d767acdd.jpg)  
(b) RTE

![](images/290ca4207f7d3b9159e5f33868c837d9f1e07a3289ab1d75b86380d3de26c660.jpg)

![](images/1d098f9f3bec66610483cf2b69e7cc4eaa04a366b3017b585c263026cfce414a.jpg)

![](images/c0e3e463165fe5d661b5fb2de2fcda7354867db234996c8d138e3cf1c9293926.jpg)  
(c) SST-2

![](images/6bdd991320ef88000aed0732c2c8788fe3e3da268a49f1d76824c5461663c481.jpg)

![](images/9aabe16eb010088370635a8a71f2df6fd02d46812f4405c0f2ba0afe5c0c4ba5.jpg)

![](images/97de3a937d82c25d5b90545422cd76552c6bb4c8522e81c0b6e09cff6008f6f7.jpg)

![](images/d4958c43978d5ab7ae4b9939d69cfdd5777fa03fc93ab8a631bca9e7c7162f71.jpg)  
(d) WebNLG

![](images/d68cfd22041867201612fff82432542b8e2ca011bf616a356dbd806a499146dc.jpg)

![](images/eab479d98ba83e0be60f24a062a06aedf5ede5fb63ae53e2ff00e22063f2948f.jpg)  
(e) WMT16 DE/EN

![](images/e0de4dade2084a6c355ef9cf7383a43f9e6dac5df86773d6021bf17e9377e9b0.jpg)  
Figure 16: Loss simulation comparisons of GPTf1uence versus Simfluence for fine-tuning on Pythia-410M across BoolQ, RTE, SST-2, WebNLG, and WMT16 DE/EN datasets.

![](images/5ee5933dd92919a9953ec30170c83e0b2762eeee226982c7c9bd5ca278be395d.jpg)

![](images/b30261898e5680173388a037b74f22db8e969f61658fd60a2a70c7ef98ab33bb.jpg)  
(a) WebNLG

![](images/75da89878e2b34a025d9bf23343db2024fc5553bb33285cc2b48cf7b69641057.jpg)

![](images/73147fc59bb84f59ec7af39522d62b6b15637017a186a4508e9371190beac839.jpg)

![](images/b7e36e3102ca9d8d868592b409908934f0bd61bf45a0c8f0dc77827967b26eaf.jpg)  
(b) WMT16 DE/EN

![](images/7f467f9a6b2b220fc663a5ad4fd864b2798467a59963a9e2a21043e8afcdb385.jpg)  
Figure 17: BLEU metric simulation comparison for fine-tuning Pythia-410M using GPTfluence and Simfluence on the WebNLG and WMT16 DE/EN tasks.

![](images/763b03f5a17208601d71c96a827b1a42f93bcb39e597a78237a31aea23c74fec.jpg)

![](images/14af24c15441d5d84a273b9b2d80c7c3ff8c2f2368fe4b9b1559cd979829a0ec.jpg)  
(a) WebNLG

![](images/d474701227da41fff58b05e06ba9e590cec28eed9202fd91feeeede593c6ac84.jpg)

![](images/12a5bf728fe3bc37eaa427e844d115d1bc26989196ab0ec7ee8f9875addf40a3.jpg)

![](images/e274bdd904b53952938b3dc39bbc1e50e891822d8dedf5e999cfcb5d9dd90fcf.jpg)  
(b) WMT16 DE/EN

![](images/af0c1974a8c1ec37929262bf0ec11b9e02b87b51350420480ab0c9a0a023b934.jpg)  
Figure 18: ROUGE-L metric simulation comparison for fine-tuning Pythia-410M with GPTf1uence and Simfluence on the WebNLG and WMT16 DE/EN tasks

![](images/dcf0918031e16be4ac56fcccbf7e8fd667a6d20482f28b4701101b953ee5fad7.jpg)  
(a) Test Example1

![](images/2e0bacf251524104cebf24666308826b408599d21b1a91209cf18c404327e95b.jpg)  
(b) Test Example2

![](images/0dca4f62b85dfe8a27d19cde1186ac253a821f58b258b43f90c6ae92a1f0313d.jpg)  
(c) Test Example3  
Figure 19: Examples of loss simulation of GPTf1uence for the RTE task on unseen test data.

![](images/a62fd512ea1bd19a79c498a9693a9e3c27ca0cb3592d06619a28ace027d2343a.jpg)  
(a) Test Example1

![](images/d1bbf621587e685a4d97dbdd15c604ee32852c6244fec89db67d15f6ff4a73e0.jpg)  
(b) Test Example2

![](images/8c20b984b9723b6773e6f09f27afe6d1ecf87c2f9327217ff6ab794f3005359c.jpg)  
(c) Test Example3  
Figure 20: Examples of loss simulation of GPTf1uence for the RTE task on unseen training data.

![](images/c84f5bb71ba089f71b69e304c473d6fd716e94000aa396ea008f362c72d2b0ad.jpg)  
(a) Test Example1

![](images/4c7c7d1e0ab79159246918b5d6086174754f6850bd642198c77d7ce1f972d062.jpg)  
(b) Test Example2

![](images/11b005b08526f1c1e0ac972360cab4c127ad71ba52075f2665d4c3072be1258e.jpg)  
(c) Test Example3  
Figure 21: Examples of loss simulation of GPTfluence for the RTE task on unseen training and test data.

![](images/8cd5ed178eb85dbc01d611790896c49977bc6ffbefb3c581fb66c2fffc870179.jpg)  
(a) Test Example1

![](images/aadbd3a6d576f5219743966253fb1fb8c534b25369b96f4080511c2b80f2a931.jpg)  
(b) Test Example2

![](images/0bd7ae1631949e1e55b0774b5fe7943243f300f7ed0db0ef6d5c201bbfbc6a60.jpg)  
(c) Test Example3  
Figure 22: Examples of loss simulation of GPTf1uence for the WebNLG task on unseen test data

![](images/485aa3a67d5f76eab4da85ea5d36f1c274a436c39be09e85323494eb6a622aa4.jpg)  
(a) Test Example1

![](images/e4dbfc51f8a94130e23516e02e19f4c2e8dde0f98387c1397cd63f3d0bbbfae3.jpg)  
(b) Test Example2

![](images/c5be06693ac10867ac48ce5799834a4c5014955b6f994cf5298ee01454758e53.jpg)  
(c) Test Example3  
Figure 23: Examples of loss simulation of GPTfluence for the WebNLG task on unseen training data

![](images/a411dc16019b4e5c8cdcbec80a1a9f20a54f059abdc86884b31f84ab35577683.jpg)  
(a) Test Example1

![](images/bce01e66407c20a07e8c5baa997907b33f3e9936c9a5a83ee82d31c094ef147e.jpg)  
(b) Test Example2

![](images/57d55d975329060ce3c9d3fb1b262013f317101914908c5845607fc0e45f51f4.jpg)  
(c) Test Example3  
Figure 24: Examples of loss simulation of GPTfluence for the WebNLG task on unseen training and test data.

![](images/2b4acffe8bb2c1bf6d74c6e5219d03872057fb6efb4bec5d767e8f14e8ca785d.jpg)  
(a) Test Example1

![](images/e3dc9c76891e0ccf2cc791d93954d9b8d69695190df0c49f76bc824a607aa789.jpg)  
(b) Test Example2

![](images/b6045937bbee1eb755d443bc78c49eaae5d08a6636ab873f43e60c74cb78eb3e.jpg)  
(c) Test Example3  
Figure 25: Examples of BLEU simulation of GPTf1uence for the WebNLG task on unseen test data

![](images/aefae8af1dd5cc44643fc23ff3286a976f559cca6825835acb3f5e3abacb9649.jpg)  
(a) Test Example1

![](images/d7615aaffab1d95a2c6a14225568f4c62fb4a4e223bbea6e2a4335541145cdfe.jpg)  
(b) Test Example2

![](images/2fb2115e156c0ffbc6d30bee9a2855a2b4af8299707f090e2339e98211910895.jpg)  
(c) Test Example3  
Figure 26: Examples of BLEU simulation of GPTfluence for the WebNLG task on unseen training data.

![](images/d81d5d32be60156d71e5cb646693992bd589bfba578b8f5c34f5abcea1dc028c.jpg)  
(a) Test Example1

![](images/9c8b6ab71a83b4885bb7068fd91fbd823dbf3a6f878ceb72359cdac7fdff0651.jpg)  
(b) Test Example2

![](images/421e6b167e901bd381cff2e8285919d0298ddbe60f1d89e7f7c130bdb9e6fb8d.jpg)  
(c) Test Example3  
Figure 27: Examples of BLEU simulation of GPTfluence for the WebNLG task on unseen training and test data.

![](images/b98c13ae88e00a4efc1520ba412b021c97bb22e12a4825985bf67f3cfbfeec12.jpg)  
(a) Test Example1

![](images/62f0d4d0ac3d95ec7772363091126e502834b396f5de77aec724d1ca3a6d0e51.jpg)  
(b) Test Example2

![](images/062245a7bad549e80bde1861c6e8793feac994a7eb315d4f81dbadd303ddb1d3.jpg)  
(c) Test Example3  
Figure 28: Examples of the ROUGE-L simulation of GPTf1uence for the WebNLG task on unseen test data.

![](images/2eca92032aeeb76d80c5d2cb7bbd8aad42471a4ff890754aa3b258fd3a3ccf0b.jpg)  
(a) Test Example1

![](images/450fdf8ce4f64942313880c3299ccd95adc52a24d13e4bc655a0d4736fc33dfc.jpg)  
(b) Test Example2

![](images/c8f24993500a5f09d2ed64f0965907f78704949fd2bbfaf5071c5dc9c068149c.jpg)  
(c) Test Example3  
Figure 29: Examples of the ROUGE-L simulation of GPTf1uence for the WebNLG task on unseen training data.

![](images/33cb28214951b5674cbfb9d8661187a7f2214a9a4441b1f713b3387cf431f585.jpg)  
(a) Test Example1

![](images/3a6c24efcfd4761eb1f8697a0c687a388618b4a5189a029ee494db694898a27f.jpg)  
(b) Test Example2

![](images/ed3082b4ab25a00af82dca9f296baeabede8eaf807fbcf354f3371252fc72090.jpg)  
(c) Test Example3  
Figure 30: Examples of the ROUGE-L simulation of GPTf1uence for the WebNLG task on unseen training and test data.