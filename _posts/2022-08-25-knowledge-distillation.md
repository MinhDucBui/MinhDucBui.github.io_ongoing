---
title: "[Paper] Survey: Knowledge Distillation in NLP"
date: 2022-06-10 11:30:47 +01:00
modified: 2022-08-22 11:30:47 +01:00
tags: [paper, nlp]
description: Master Thesis
image: "/assets/img/knowledge-distillation/introduction.jpg"
---
*Part of the related work section in my Master Thesis. Download the full version
<a href="/assets/img/master-thesis/Master_Thesis.pdf" download>here</a>.*


# Table of Contents
1. [Motivation & Preliminaries](#1-motivation---preliminaries)
2. [Brief History: Knowledge Distillation for Transformers](#2-brief-history--knowledge-distillation-for-transformers)
3. [Transformer Components Distillation](#3-transformer-components-distillation)
4. [Distillation Setup Strategies](#4-distillation-setup-strategies)
5. [Challenges](#5-challenges)

Knowledge Distillation is a technique to transfer knowledge from a
well-trained teacher model to a student model, typically smaller than
the teacher. During training, Knowledge Distillation allows the student
model to access the learned knowledge from the teacher model, providing
more information to the student. Thus, the student can achieve similar
or even better performance than the teacher.

As our thesis is based on multilingual transformers, we restrict our
review to Knowledge Distillation for transformers.

# 1. Motivation & Preliminaries

**Motivation.** {% cite Bucila2006ModelC %} first introduced the concept of
Knowledge Distillation as a way to compress models by transferring the
learned knowledge to a smaller model. In their work, the primary goal
was to compress a large complex ensemble into smaller, faster models
without significant performance loss. First, they train the ensemble
model (also called *teacher*) and then transfer the acquired knowledge
to a smaller model (also called *student*) by mimicking the teacher's
behavior. Thus the idea of Knowledge Distillation[^1] was born.

Knowledge Distillation is especially beneficial in the case of complex,
overparameterized teachers. From an optimization perspective,
{% cite du2018power, soltanolkotabi2018theoretical %} show that high capacity
models (i.e., the teacher) can find a good local minimum due to
over-parameterization. Therefore, using these over-parameterized models
to guide a lower capacity model during training can facilitate
optimization. The state-of-the-art multilingual models are massive
pre-trained transformers consisting of millions of parameters which some
argue are over-parameterized but need the capacity during pre-training
{% cite Hao_2019, conneau2020unsupervised, dufter2021identifying %}. Our
approach uses Knowledge Distillation to precisely distill these
transformer models to induce cross-lingual knowledge into students.

**Vanilla Knowledge Distillation.** The most straightforward approach to
Knowledge Distillation is to distill from the output layer by matching
the output of the teacher and student
{% cite Bucila2006ModelC, ba_deep_2013, hinton2015distilling %}. Initially,
{% cite Bucila2006ModelC %} used the teacher to make predictions on an unlabeled
dataset producing labels for the student to mimic. The created labels
are referred to as *hard labels*, and the dataset used to train the
student is called *transfer dataset*. As hard labels only transfer
information about the highest class probability, {% cite ba_deep_2013 %} propose
to use the logits of the teacher, called *soft labels*. The idea is that
the soft labels of a well-trained teacher provide additional supervisory
signals of the inter-class similarities to the student. They directly
used the logits $z$ by minimizing the squared difference between the
logits produced by the teacher and the logits produced by the student:

$$\begin{equation} \tag{1}\label{eq:logits_mse} 
     L_{\text{logits}} = \frac{1}{2N} \sum_{x \in X} \Vert z_{x}^{TE} - z_{x}^{ST}\Vert_2^2 ,
\end{equation}$$ 

where $N$ is the number of examples, $z_{x}^{TE}$ the
logit output of the teacher and $z_{x}^{ST}$ of the logit output of the
student.

One problem that may arise in well-trained teachers is that they are
overconfident and thus almost always predict classes (tokens) with very
high confidence. The learned similarities between similar classes
(tokens) reside in the ratios of very small probabilities in the soft
targets - These, however, have very little influence on the cost
function $\eqref{eq:logits_mse}$ during distillation. To leverage the
information residing in these small probabilities {% cite hinton2015distilling %}
propose to *soften* the distribution by modifying the predicted
probability distribution of the teacher. {% cite hinton2015distilling %} introduce
a variable called temperature into the final softmax: 

$$\begin{equation} \tag{2}\label{eq:temperature}
    \sigma (z_{x}; T)\_i := p_i = \frac{\exp{(z_{x, i}/T)}}{\sum_j \exp{(z_{x, j}/T)}},
\end{equation}$$

where $T$ denotes the temperature, which is normally set
to $1$ for the softmax function. A higher temperature $T$ causes a
softer probability distribution over tokens, resulting in a higher
entropy in the probability distribution. {% cite hinton2015distilling %} then use
the same temperature $T$ when training the student to match these soft
targets. Furthermore, {% cite hinton2015distilling %} showed that matching logits
is a special case of the modified softmax in $\eqref{eq:temperature}$. Using the Kullback-Leibler divergence
between teacher and student, we get the *(unscaled) Hinton loss
function*: 

$$\begin{aligned}
     \sum_{x \in X} D_{KL} \left(\sigma(z_{x}^{TE}; T) \Vert  \sigma(z^{ST}_x; T)\right).
\end{aligned}$$ 

To compensate for the changed magnitude of the gradients
caused by the soft targets scale, {% cite hinton2015distilling %} multiply the
loss function with $T^2$ to get the *Hinton loss function*:

$$\begin{equation} \tag{3}\label{eq:hinton_loss}
    L_{H} = T^2 \cdot \sum_{x \in X} D_{KL} \left(\sigma (z_{x}^{TE}; T) \Vert \sigma (z_{x}^{ST};
T) \right).
\end{equation}$$ 

This allows us to change the temperature while
experimenting without changing the relative contributions of the soft
targets. The Hinton loss $\eqref{eq:hinton_loss}$ is the vanilla setup that is mainly used and
referred to as (vanilla) Knowledge Distillation.

The main difference between different Knowledge Distillation strategies
is how one mimics the teacher's behavior. Formally, we can define a
transformation $f^{TE}$ and $f^{ST}$ of the teacher and student network
inputs, respectively, to some informative representation for transfer.
{% cite jiao2020tinybert %} calls these transformations *behavior functions*.
Knowledge Distillation can then be defined in general as minimizing the
objective function 

$$\begin{aligned}
    L_{\text{KD}} = \sum_{x \in X} L \left(f^{TE}(x), f^{ST}(x) \right),
\end{aligned}$$ 

where $L(\cdot)$ is a loss function evaluating
the difference between a teacher and student networks, $x$ the (text)
input, and $X$ denotes the training set. Behavior functions
can be defined for any type of representation of the input, e.g., the
logits of the final output or some intermediate representations in the
network, which the student then mimics.

# 2. Brief History: Knowledge Distillation for Transformers

Knowledge Distillation methods first found their use in the area of
Computer Vision. As AlexNet {% cite Krizhevsky_2012 %}, one of the largest
neural networks at that time, completely outperformed other methods on
ImageNet {% cite deng2009imagenet %}, the trend towards bigger models started in
that area. To keep the models to a relatively acceptable size while
still having the same performance as the big models, a variety of model
compression techniques were developed, such as Knowledge Distillation,
see the survey of {% cite Gou_2021 %} for a more in-depth look into Knowledge
Distillation in Computer Vision.

**Task-Specific Distillation.** {% cite sun2019patient %} was one of the first
known successes at using Knowledge Distillation for BERT at the
fine-tuning stage. They introduce *Patient Knowledge Distillation (PKD)*
which uses the intermediate representations of BERT for a more effective
distillation (additional to the Hinton Loss). This is motivated by
previous findings of {% cite romero2015fitnets %} showing that distilling
intermediate representations can serve as hints for the student during
training, improving the final performance. Similar to PKD, XtremeDistil
{% cite mukherjee2020xtremedistil %} also distills from intermediate
representations but additionally utilizes parameter projection that is
agnostic of teacher architecture. Subsequently, inspired by the findings
that attention weights of BERT capture linguistic knowledge and that
BERT becomes more complex in higher layers {% cite clark2019does %}, *Stacked
Internal Distillation (SID)* {% cite aguilar2020knowledge %} additionally
distills the attention probabilities of the teacher and from lower
layers first. However, these works used distillation for fine-tuned
models on specific downstream tasks (*task-specific distillation*). In
our thesis, we want to produce a general-purpose student that can be
fine-tuned on any downstream task. While we are not focusing on
task-specific distillation
{% cite liu2019improving, turc2019wellread, tang2019distilling, kaliamoorthi2021distilling %}
or multi-task distillation
{% cite tan2019multilingual, clark2019bam, liu2020mkd %}, we still want to
mention important work in this field.

**General-Purpose Students.** To produce general-purpose students,
distillation happens during the pre-training tasks, mainly the masked
language modeling task. In this task, the transformer model is trained
to predict the original token for each masked token by maximizing the
estimated probability for this token. The standard loss function is to
minimize the cross-entropy between the transformer's predicted
distribution and the one-hot distribution of all tokens. We denote the
loss as the masked language modeling loss $L_{\text{MLM}}$
{% cite devlin2019bert %} as 

$$\begin{equation} \tag{4}\label{eq:ce}
    L_{\text{MLM}}= - \sum_{x \in X} \left( \sum_i l_{x, i} \log(p_{x, i}) \right),
\end{equation}$$ 

where $l_{x, i}$ is the ground-truth and $p_{x, i}$ the
predicted probability for $i$-th token in the text input $x$. Typically
a transformer applies a softmax function to obtain $p_{x, i}$, denoted
by $\sigma (z_{x})=p_{x} = (p_{x, 1}, ..., p_{x, C})$, where $\sigma $ is
the softmax function and $z_{x} = (z_{x, 1}, ..., z_{x, C})$ the logit
output of the text input $x$.

*DistilBert* {% cite sanh2020distilbert %} extends the work of {% cite sun2019patient %}
by distilling during pre-training on a large-scale corpus with a
soft-label distillation loss and a cosine embedding loss to construct a
general-purpose student. Furthermore, they initialize the student from
the teacher by taking one layer out of two. Similar to SID
 {% cite aguilar2020knowledge %}, TinyBert {% cite jiao2020tinybert %}  introduce
additional attention distillation to produce general-purpose students.
Specifically, they distill self-attention distributions
$\text{Attention}(Q, K, V)$. TinyBert further uses task
distillation with data augmentation to strengthen performance on
downstream tasks. They show that their $6$ layer model performs on par
with its teacher BERT on the GLUE benchmark {% cite wang2019glue %} . Instead of
using shallower students, MobileBERT {% cite sun2020mobilebert %}  uses thinner
students by introducing bottleneck structures which have been shown to
be more effective {% cite turc2019wellread %} . Compared to TinyBert, they
outperform it on GLUE and SQuAD with a similar-sized MobileBERT while
not utilizing any task-distillation or data augmentations during
fine-tuning. However, MobileBERT constructs another teacher network and
many architectural modifications to help facilitate training. MiniLM
 {% cite wang2020minilm %} only distills the self-attention module with an added
loss function: They align the relation between values in the
self-attention module, which is calculated via the multi-head scaled
dot-product between values $V$. Furthermore, they only distill from the
last transformer layer of the teacher. Compared to TinyBert and
MobileBert, MiniLM alleviates the difficulties in layer mapping between
the teacher and student models, and the layer number of our student
model can be more flexible. {% cite khanuja2021mergedistill %}introduce
MergeDistil, which uses multiple mono- or multilingual teachers to
distill into one general-purpose student to leverage language-specific
LM.

To further dive deeper into previous works and build a foundation for
further exploration for our thesis, we distinguish previous works into
which parts of the transformer they distill from.

# 3. Transformer Components Distillation

This section will explore different behavior functions and their
corresponding loss function to transfer knowledge from a transformer
teacher into a transformer student. We categorize the possible parts of
the transformer that we can distill from into: *Final output layer*,
*hidden layer representations*, *attention* and *embedding
distillation*.

We summarize our findings in the next Table:

<figure>
<img src="/assets/img/master-thesis/table_1.png" alt="interview-img">
<em>Categorizing of previous works into different transformer parts
distillation during pre-training. Notice that TinyBert does not utilize
soft or hard targets during pre-training, only in their second
distillation stage (task-distillation). Furthermore, we specify the loss
function (KL: Kullback Leibler Loss; MSE: Mean-squared error;
COS: Cosine Embedding Loss; CE: Cross-entropy loss; SVD:
Singular Value Decomposition.</em>
</figure>

**Final Output Layer.** One simple idea to induce knowledge from a
teacher during pre-training is to use the predicted distribution over
all tokens of the teacher model as *soft targets* for training the
student. The hope is that soft targets are much more informative than
hard targets as they can reveal similarities between tokens, e.g., the
masked token is \"dog\" and our well-trained teacher model would give a
high probability to \"dog\", but also to \"cat\" as both could make
sense in the masked sentence. This information about similarities
between tokens can then be transferred to our students. One problem that
may arise in well-trained teachers is that they almost always predict
the correct token with very high confidence.

Almost all works use the hard and soft targets to distill knowledge from
the teacher. PKD, SID, and DistilBERT use the cross-entropy loss to
distill from the soft and hard targets, while XtremeDistil uses the
mean-squared error to align soft targets. In some implementations, the
Kullback Leibler loss is used to align soft targets. During
pre-training, TinyBert does not use any output layer distillation as
{% cite jiao2020tinybert %} argue that their goal is to primarily learn the
intermediate structures of BERT. Furthermore, they conduct experiments
showing that output layer distillation does not bring additional
improvements on downstream tasks.

**Hidden Layer Representation Distillation.** Using only logits to
transfer cross-lingual knowledge from the teacher can be problematic as
they only serve as one \"task-specific\" knowledge, in this case, masked
language modeling. {% cite romero2015fitnets  %} show that additional hints from
the *intermediate layers* can improve the training process and the final
performance of the student.

As one has to decide which layer of the teacher to distill from
(typically, the student is smaller than the teacher), we have to define
a mapping function. For this purpose, we assume that the teacher model
has $N$ Transformer layers and the student $M$ Transformer layers, where
$M \leq N$. The index $0$ corresponds to the embedding layer, $M+1$ the
index of the student's prediction layer, and $N+1$ the index of the
teacher's prediction layer. The mapping function is then defined as

$$\begin{equation} \tag{5}\label{eq:mapping-function}
    n = g(m), \qquad 0 < m \leq M, 0 < g(m) \leq N
\end{equation}$$ 

between indices from student layers to teacher layers,
where the teacher's $g(m)$-th layer is distilled into the $m$-th layer
of the student model. Typically, there are four different mapping
strategies:

- *Uniform*: We distill uniformly over the teacher layers. E.g. assume 12 layer teacher and 6 layer student, then 
  the mapping function is $g(m) = m * 2$.

<figure>
<img src="/assets/img/master-thesis/uniform.PNG" alt="interview-img" width="400">
</figure>

- *Top*: We distill from the top layers of the teacher: $g(m) = m + N - M$.

<figure>
<img src="/assets/img/master-thesis/top.PNG" alt="interview-img" width="400">
</figure>

- *Bottom*: We distill from the bottom layers of the teacher: $g(m) = m$.

<figure>
<img src="/assets/img/master-thesis/bottom.PNG" alt="interview-img" width="400">
</figure>

- *Last*: We distill only from the last layer of the teacher into the  last layer of the student: $g(M) = N$.


PKD {% cite sun2019patient %}  focuses on extracting intermediate representations
but only for the `[CLS]` token, reasoning that BERT's original
implementation {% cite devlin2019bert %}  mostly uses the output from the last
layer's `[CLS]` token to perform predictions for downstream tasks. The
hope is that if the student can imitate the representation of `[CLS]` in
the teacher's intermediate layers for any given input, the
generalization ability can be similar to the teacher. The additional
loss is then defined as the mean squared error between the normalized
hidden states of the `[CLS]` token in all layers: 

$$\begin{equation} \tag{6}\label{eq:msecls}
    L_{\text{MSECLS}} = \sum_{x \in X} \sum_{m=1}^{M} \text{MSE} \left(\text{CLS}_m^{ST}(x), \text{CLS}_{g(m)}^{TE}(x) \right)^2,
\end{equation}$$ 

where $\text{CLS}^{ST}\_m(x) \in \mathbb{R}^{d}$ and
$\text{CLS}^{TE}\_{g(m)}(x) \in \mathbb{R}^{d}$ extracts the `[CLS]`
token for the input $x$ in the layer $m$ for student and $g(m)$ teacher
respectively. PKD was tested with the mapping function *top* and
*uniform*, where the strategy *uniform* showed better performance.
Similar to PKD, SID {% cite aguilar2020knowledge  %} also uses hidden vector
representations for the `[CLS]` token to additionally align internal
representations, but opted to use cosine similarity 

$$\begin{aligned}
    L_{\text{cos}CLS}(x) = 1 - cos\left(\text{CLS}_m^{ST}(x), \text{CLS}_{g(m)}^{TE}(x) \right)
\end{aligned}$$ 

with the *uniform* mapping strategy. The obvious problem
with focusing on the `[CLS]` is that it only works for tasks that use
the `[CLS]` for prediction such as tasks in the GLUE dataset
{% cite wang2019glue %}.

{% cite jiao2020tinybert %} extended the idea of distilling hidden layer
representations to student representations with any arbitrary number of
hidden dimensions with their proposed model *TinyBert*. Let
$H^{ST}\_m \in \mathbb{R}^{L \times d'}$ and
$H^{TE}\_{g(m)} \in \mathbb{R}^{L \times d}$ refer to the hidden states
in layer $m$ of student and teacher networks respectively, $L$ the
sequence length and the scalars $d', d$ denote the hidden sizes of the
student and teacher models respectively. Instead of assuming that the
hidden dimension is the same for the teacher and student $d' = d$, they
allow for $d \neq d'$ in general. This allows for creating thinner
models by using a smaller hidden dimension $d' < d$ than the teacher.
They achieve this by introducing a learnable linear transformation
matrix $W \in \mathbb{R}^{d' \times d}$ to transform the hidden states
of the student network into the same space as the teacher network's
states, i.e., $H^{ST}\_m W \in \mathbb{R}^{L \times d}$. TinyBert
transfers the knowledge which resides in the hidden layers output
representation from *each token* in the sequence by minimizing the MSE
across all tokens: 

$$\begin{equation} \tag{7}\label{eq:msehidn}
    L_{\text{Hid}\_MSE}(\ \cdot \ ; m) = \text{MSE}(H^{ST}_m W_m, H^{TE}_{g(m)}),
\end{equation}$$ 

where the matrices $H^{ST}\_m \in \mathbb{R}^{L \times d'}$ and
$H^{TE}\_{g(m)} \in \mathbb{R}^{L \times d}$ can have different hidden
sizes $d \neq d'$. Furthermore, they experimented with all three mapping
functions and concluded that *uniform* works best in their case. Notice
that we can also align hidden representations with the cosine loss

$$\begin{equation} \tag{8}\label{eq:coshidn}
    L_{\text{COS}hidn} = 1 - \text{cos}(H^{ST}_m W_m, H^{TE}_{g(m)}), 
\end{equation}$$ 

with the matrices
$H^{ST}\_m \in \mathbb{R}^{L \times d'}$ and
$H^{TE}\_{g(m)} \in \mathbb{R}^{L \times d}$.

*XtremeDistil* also uses a projection to make all output spaces
compatible. The projection however is non-linear and they use the
KL-divergence: 

$$\begin{aligned}
    L_{\text{KL}hidn}(\ \cdot \ ; m) = D_{KL}\left(\textit{Gelu}\left(H^{ST}_m W_m + b_m\right) \Vert H^{TE}_{g(m)} \right),
\end{aligned}$$ 

where $W_m$ is the learnable projection matrix, $b_m$ the learnable bias term and *Gelu* (Gaussian Error Linear Unit) is the
non-linear projection function.

As we have seen, different mapping functions were used across previous
work. It might not be useful to distill from intermediate
representations as in {% cite jiao2020tinybert, mukherjee2020xtremedistil, aguilar2020knowledge %}, especially when the
student is a network of lower representational capacity.
{% cite yang2021knowledge %} showed that using only the last hidden layer (*last*
strategy) for matching representation results in the best performance,
albeit experiments were conducted with Convolutional Neural Networks. To
further align the last hidden representation, {% cite yang2021knowledge %} use the
teacher's pre-trained Softmax Regression (SR) classifier to first pass
the input $x$ to the teacher network resulting in the output

$$\begin{aligned}
    \sigma (z_x^{TE}) = \sigma (W^{TE}_{N+1} H_N^{TE}),
\end{aligned}$$ 

where $W^{TE}\_{N+1}$ is the weight matrix of the softmax
classifier head. Moreover, they feed the same input to the student to
get the last hidden representation $H_M^{ST}$ and input the
representation to the teacher's SR classifier to obtain

$$\begin{aligned}
    \sigma (z_x^{ST}) = \sigma (W^{TE}_{N+1} H_M^{ST}).
\end{aligned}$$ 

They then optimize the loss 

$$\begin{aligned}
    L_{SR} = - \sigma  \left(W^{TE}_{N+1} H_M^{TE} \right) \log \left( \sigma (W^{TE}_{N+1} H_M^{ST}) \right),
\end{aligned}$$ 

while keeping the classifier's parameter $W^{TE}_{N+1}$
frozen. This loss is aligning the last hidden representations since if
$\sigma (z_x^{TE}) = \sigma (z_x^{ST})$ (which the loss is optimizing to)
then this implies that $H_N^{TE} = H_M^{ST}$.

**Attention Distillation.** {% cite clark2019does  %} found that the attention
weights of the transformer model BERT can capture linguistic knowledge,
which can be used to transfer linguistic knowledge into our student.



Motivated by this, {% cite jiao2020tinybert  %} additionally uses attention
distillation for TinyBert, which uses the mean squared error to align
the teacher and students matrices of the multi-head attention. Let the
matrices $Q, K, V \in \mathbb{R}^{L \times d_k}$ denote the queries,
keys and values, where $d_k$ is the dimension of keys. As already
discussed, the attention $\text{Attention}(Q, K, V)$ is calculated with

$$\begin{equation} \tag{8}\label{eq:attention}
    A = \frac{QK^{T}}{\sqrt{d_k}} \in \mathbb{R}^{L \times L}, \quad \text{Attention}(Q, K, V) = \sigma (A)V \in \mathbb{R}^{L \times d_k},
\end{equation}$$ 

where $d_k$ acts as a scaling factor. We then calculate
the attention loss with 

$$\begin{equation} \tag{9}\label{eq:lossatt}
    L_{\text{MSEAtt}} = \frac{1}{h} \sum_{i=1}^h \text{MSE}(A_i^{ST}, A_i^{TE}),
\end{equation}$$ 

where $h$ is the number of attention heads,
$A_i \in \mathbb{R}^{L \times L}$ refers to the attention matrix before
applying the softmax, the index $i$ corresponds to the $i$-th head of
teacher or student, $L$ is the input text length, and MSE$(\cdot)$ means
the mean squared error loss function. They argue that using $A$ instead
of $\text{Attention}(Q, K, V)$  show faster convergence rate and
better performances. We visualize the TinyBert attention and
intermediate hidden representation distillation in the next Figure:

<figure>
<img src="/assets/img/master-thesis/layer_distill.PNG" alt="interview-img" width="500">
<em>Visualization of a attention and intermediate hidden representation
distillation. Taken from {% cite jiao2020tinybert %}.</em>
</figure>

Nonetheless {% cite aguilar2020knowledge  %} make use of the attention matrix
$\text{Attention}(Q, K, V)$ and chose the KL-divergence loss.
For a given head in a transformer, the loss can be formulated as

$$\begin{aligned}
    L_{\text{KLAtt}} = \frac{1}{L} \sum_i^L {\tt Att}(Q^{TE}, K^{TE}, V^{TE})_i \log \left(\frac{Att(Q^{TE}, 
K^{TE}, V^{TE})_i}{Att(Q^{ST}, K^{ST}, V^{ST})_i} \right),
\end{aligned}$$ 

where $Att(Q, K, V)_i = \text{Attention}(Q, K, V)_i \in  \mathbb{R}^{d_k}$
describe the $i$-th row of the attention probability matrix.

MiniLM {% cite wang2020minilm %}  additionally aligns the relation between values
in the self-attention module, which is calculated via the multi-head
scaled dot-product between values $V^{ST}$ and $V^{TE}$.

**Embedding Distillation.** Multiple lines of work indicate that the
embedding layer is important for a successful cross-lingual
representation {% cite pires-etal-2019-multilingual, wu-dredze-2019-beto, dufter2021identifying %}.
Therefore, it is also important to distill from our multilingual
teacher's embedding layer to our students.

DistilBert uses the cosine embedding loss to align the embeddings of
teacher and student: 

$$\begin{aligned}
    L_{\text{COS}embed} = 1 - \text{cos}(E^{ST}_m, E^{TE}).
\end{aligned}$$ 

where the matrices $E^{ST}$ and $E^{TE}$ refer to the
embeddings of student and teacher networks, respectively. Instead of
using the cosine loss, TinyBert uses the MSE: 

$$\begin{equation}
    L_{\text{Emb}\_MSE} = \text{MSE}(E^{ST} W_e, E^{TE}).
\end{equation}$$ 

Again, TinyBert allows for a mismatch between dimensions
of $E^{ST}$ and $E^{TE}$ by using a learnable linear transformation
$W_e$.

XtremeDistil {% cite mukherjee2020xtremedistil %}  uses Singular Value
Decomposition (SVD) to project the word embeddings of the teacher to a
lower-dimensional space for the student since they use the same
WordPiece vocabulary.

# 4. Distillation Setup Strategies

This thesis introduces a novel distillation setup to induce aligned
monolingual students, which is why we review different distillation
setup strategies for transformers in this section. We restrict our
literature review to a \"fixed\" teacher at the distillation time and
not an, e.g., jointly trained teacher-student setup, such as in
{% cite jin2019knowledge %}. Furthermore, we focus on the general distillation
stage, where we only perform distillation on the Masked Language
Modeling objective. Finally, the task is to distill knowledge from the
multilingual teacher(s) to our student while MLM pre-training on
multiple monolingual text corpora, i.e., multilingual corpus, obtaining
a general-purpose student.

**One Teacher, One student.** In this setup, only one teacher exists and
is distilled into one student. {% cite reimer_sbert  %} use parallel data to
facilitate strong cross-lingual sentence representations by training the
student model such that (1) identical sentences in different languages
are close and (2) the original source language from the teacher model
SBERT {% cite sbert_reimers_2019 %}  are adopted and transferred to other
languages.


They do so by minimizing the mean squared error of (1) the source
sentence embedding of the teacher with the target sentence embedding
using parallel data and (2) the source sentence embedding of the teacher
and the student, see the next figure for a visualization:

<figure>
<img src="/assets/img/master-thesis/sbert_kd.PNG" alt="interview-img">
<em>Visualization of the distillation strategy of {% cite reimer_sbert.  %} Given
parallel data (here: English and German), the student model is trained
such that the sentence embeddings for the English and German sentences
are close to the teacher English sentence vector. Adopted from
{% cite reimer_sbert %}.</em>
</figure>

Furthermore, all discussed monolingual distillation approaches can be
extended to create a multilingual student straightforwardly: Distill a
multilingual teacher into a student during the MLM task on a
multilingual corpus. Since the teacher is already multilingual and the
representations aligned {% cite pires-etal-2019-multilingual, wu-dredze-2019-beto %}, the cross-lingual
knowledge can then be distilled into one student. Consequently, every
monolingual distillation approach can be directly applied, e.g. PKD {% cite sun2019patient %}, DistilBert {% cite sanh2020distilbert %}  or TinyBert
 {% cite jiao2020tinybert %}. We can generalize the distillation loss as a
*convex combination* or a *linear combination* of all chosen loss
functions. E.g. DistilBert uses a convex combination of the masked
language modeling loss $L\_{\text{MLM}}$, the
Hinton Loss $L_H$ \eqref{eq:hinton_loss} and the cosine embedding loss
$L_{\text{COS}embed}$ \eqref{eq:coshidn}: 

$$\begin{aligned}
    L_{\text{DB}} = \alpha \cdot L_H + \beta \cdot L_{MLM} + \gamma \cdot L_{\text{COS}embed}(\ \cdot \ ; M),
\end{aligned}$$ 

where $M$ is the index of the last hidden layer, $\alpha, \beta, \gamma \in [0, 1]$ with $\alpha + \beta + \gamma = 1$.


**Multiple Teacher, One Student.** Language-specific language models may
perform better given a sizable pre-training data volume than
multilingual teachers in their respective language but are, in turn,
monolingual and do not use any positive language transfer. Distilling
multiple task-agnostic language-specific teachers into one task-agnostic
student can help the student be competitive or outperform the individual
language-specific LMs while still being multilingual.
{% cite khanuja2021mergedistill %} merges multiple monolingual or multilingual
pre-trained LMs into a single task-agnostic multilingual student using
task-agnostic Knowledge Distillation, which they call *MergeDistill*.
The difficulty is that each LM can have its own vocabulary.
{% cite khanuja2021mergedistill %} use the union of all teacher LM vocabularies
for the student vocabulary. They use a vocab mapping step *teacher
$\rightarrow$ student*, converting each teacher token index to its
corresponding student token index. They first tokenize and predict for
each language using their respective teacher LM and get the top-$k$
logits for each masked word. For distillation, they then use the Hinton
Loss $L_H$ $\eqref{eq:hinton_loss}$ and MLM loss $L_\text{MLM}$.
Interestingly, their experiments show that due to the shared
multilingual representations, the student is able to perform in a
zero-shot manner on related languages that the teacher does not cover.

**One Teacher, Multiple Students.** In this setup, we have one
multilingual teacher and want to distill into multiple mono-, bi- or
multilingual students, for which the representations for each language
are aligned across students. In this work, we will focus on distilling
into multiple monolingual students. To the best of our knowledge, this
is the first work that investigates distilling from a multilingual
teacher into monolingual students sharing a representation space.

# 5. Challenges

The trend towards bigger and bigger models in NLP is fueled by the high
generalization power of the learned representations. Smaller models lack
the inductive biases to learn these representations from the training
data alone but may have the capacity to represent these solutions
 {% cite ba_deep_2013, stanton2021does %}. We discussed several methods such as
DistilBert {% cite sanh2020distilbert %}  and TinyBert {% cite jiao2020tinybert %}  that
achieve similar performances on some downstream tasks as the teacher
with just a fraction of the total parameter number. In the previous
sections, we discussed different Knowledge Distillation strategies to
induce knowledge into the student, e.g., the effects of distilling
different transformer components into the student. However, we highlight
two open challenges in regards to Knowledge Distillation in general
(Fidelity vs. Generalization) and our thesis (KD for Monolingual
Students).

**Fidelity vs. Generalization.** Recently, {% cite stanton2021does  %} show that
while Knowledge Distillation can improve the generalization abilities of
students, there often remains a low fidelity, i.e., the ability of a
student to match a teacher's predictions. Previous works
 {% cite furlanello_2018, mobahi_2020 %} already show that in self-distillation,
the student can improve generalization. This can only happen by virtue
of failing at the distillation procedure: The student fails to match the
teacher:

<figure>
<img src="/assets/img/master-thesis/fidelity.PNG" alt="interview-img">
<em>The effect of enlarging the CIFAR-100 distillation dataset with
GAN-generated samples. The shaded region corresponds to
$\mu \pm \sigma $, estimated over three trials. (a) The teacher and
student have the same model capacity. Student fidelity increases as the
dataset grows, but the test accuracy decreases. (b) The teacher has a
larger model capacity than the student. Student fidelity again increases
when the dataset grows, but the test accuracy now also slightly
increases. Figure taken from
{% cite stanton2021does %}.</em>
</figure>


These experiments, however, hold true for students that have the *same
model capacity* as the teacher. Often there is a significant disparity
in generalization between large teacher models and smaller students.
Importantly, {% cite stanton2021does  %} then show that for these larger teacher
models, improvements in fidelity translate into improvements in
generalization (Figure (b)).

**KD for Monolingual Students.** As this is the first work that explores
distilling multilingual encoders into monolingual components (to the
best of our knowledge), the question remains which parts of the teacher
are important to distill from to improve (1) alignment and (2)
cross-lingual downstream task performance. Another open question is
whether sharing between students and weight initialization from the
teacher improves (1) and (2). Finally, as we have multiple students, we
can not utilize the default approach to solve cross-lingual downstream
tasks: Fine-tuning one multilingual model in the source language and
evaluating/train with the same model in the target language. We must
explore fine-tuning strategies for multiple (monolingual) students for
cross-lingual downstream tasks.

# References

{% bibliography --cited %}

[^1]: Knowledge Distillation can also be referred to as teacher-student
    Knowledge Distillation.



