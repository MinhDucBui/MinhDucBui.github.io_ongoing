---
title: "[Paper] Survey: Parameter-Efficient Fine-Tuning in NLP"
date: 2022-06-28 11:30:47 +01:00
modified: 2022-08-22 11:30:47 +01:00
tags: [paper, nlp]
description: Master Thesis
image: "/assets/img/efficient-tuning/introduction.jpg"
---
*Part of the related work section in my Master Thesis. Download the full version
<a href="/assets/img/master-thesis/Master_Thesis.pdf" download>here</a>.*




# Table of Contents
1. [Motivation](#1-motivation)
2. [Adapters](#2-adapters)
3. [Sparse Fine-Tuning](#3-sparse-fine-tuning)



# 1. Motivation

The standard approach to solving a new task with a
pre-trained transformer is by adding a task-head (e.g., a linear
classification layer) on top of the pre-trained transformer (encoder)
and minimizing the task loss end-to-end. However, this approach results
in a completely new unique, large model making it harder to track what
significantly changed during fine-tuning and therefore making it also
hard to transfer acquired task-specific knowledge (*modularity*). The
latter is important in our monolingual setup as we want to transfer the
acquired task-specific knowledge by our source student into our target
student. Ideally, transferring the
acquired task-specific knowledge still matches the results of fully
fine-tuning one model.

The first work that we explore is Adapters
{% cite rebuffi2017learning, houlsby2019parameterefficient %} which inserts a
small subset of trainable task-specific parameters between layers of a
model and only changes these during fine-tuning, keeping the original
parameters frozen. We then discuss sparse fine-tuning, which only
changes a subset of the pre-trained model parameters. Specifically, we
consider Diff-Pruning {% cite guo_2020 %}, which adds a sparse, task-specific
difference-vector to the original parameters, and BitFit {% cite bitfit_2021 %},
which enforces sparseness by only fine-tuning the bias terms and the
classification layer.

# 2. Adapters

Adapters were initially proposed for computer vision to adapt to
multiple domains {% cite rebuffi2017learning %} but were then used as an
alternative lightweight training strategy for pre-trained transformers
in NLP {% cite houlsby2019parameterefficient %}. Adapters introduce additional
parameters to a pre-trained transformer, usually small, bottleneck
feed-forward networks inserted at each transformer layer. Adapters
enable us to keep the pre-trained parameters of the model fixed and only
fine-tune the newly introduced parameters on either a new task
{% cite houlsby2019parameterefficient, stickland2019bert, pfeiffer2021adapterfusion %}
or new domains {% cite bapna2019simple %}. Adapters perform either on par or
slightly below full fine-tuning
{% cite houlsby2019parameterefficient, stickland2019bert, pfeiffer2021adapterfusion %}.
Importantly, adapters learn task-specific representations which are
compatible with subsequent transformer layers {% cite adapterhub_2020 %}.


**Placement & Architecture.** Most work insert adapters at each layer of
the transformer model, the architecture and placement of adapters are,
however, non-trivial: {% cite houlsby2019parameterefficient %} experiment with
different adapter architectures and empirically validated that using a
two-layer feed-forward neural network with a bottleneck worked well:

<figure>
<img src="/assets/img/master-thesis/placement.PNG" alt="interview-img">
</figure>

This simple down- and
up-projection with a non-linearity has become the common adapter
architecture. The placement and number of adapters within each
transformer block are still debated. {% cite houlsby2019parameterefficient %}
place the adapter at two positions: One after the multi-head attention
and one after the feed-forward layers. {% cite stickland2019bert %} just use one
adapter after the feed-forward layers, which {% cite bapna2019simple %} adopted
and extends by including a layer norm {% cite ba2016layer %} after the adapter.
{% cite pfeiffer2021adapterfusion %} test out different adapter positions and
adapter architectures jointly and came to the conclusion to use the same
adapter architecture as {% cite houlsby2019parameterefficient %} but only places
the adapter after the feed-forward neural network:

<figure>
<img src="/assets/img/master-thesis/architecture_adapter.JPG" alt="interview-img">
</figure>

**Modularity of Representations.** One important property of Adapters is
that they learn task-specific knowledge within each adapter component.
The reason is that they are placed within a frozen transformer block
layer, forcing the adapters to learn an output representation compatible
with the subsequent layer of the transformer model. This results in
modular adapters, meaning that they can be either stacked on top of each
other or replaced dynamically {% cite mad_x %}. This modularity of adapters can
be used to fine-tune multiple monolingual aligned students on
cross-lingual downstream tasks: Instead of fine-tuning the whole student
model on the source language, we insert and fine-tune the adapters,
which then can be inserted into the monolingual student corresponding to
the target language. 

# 3. Sparse Fine-Tuning

Sparse fine-tuning (SFT) only fine-tunes a small subset of the original
pre-trained model parameters at each step, effectively fine-tuning in a
parameter efficient way. The fine-tuning procedure can be describes as

$$\begin{aligned}
    \Theta_{\text{task}} =  \Theta_{\text{pretrained}} + \delta_{\text{task}}
\end{aligned}$$ 

where $\Theta_{\text{task}}$ is the task-specific
parameterization of the model after fine-tuning,
$\Theta_{\text{pretrained}}$ is the set of pretrained parameters which
is fixed and $\delta_{\text{task}}$ is called the task-specific diff
vector. We call the procedure sparse if $\delta_{\text{task}}$ is
sparse. As we only have to store the nonzero positions and weights of
the diff vector, the method is parameter efficient. Method generally
differ in the calculation of $\delta_{\text{task}}$ and its induced
sparseness.

**Methods.** {% cite guo_2020 %} introduce Diff Pruning, which determines
$\delta_{\text{task}}$ by adaptively pruning the diff vector during
training. To induce this sparseness, they utilize a differentiable
approximation of the $L_0$-norm penalty {% cite louizos_2017 %}. {% cite bitfit_2021 %}
induce sparseness by only allowing non-zero differences in the bias
parameters (and the classification layer) of the transformer model. The
method, called BitFit, and Diff-Pruning, both can match the performance
of fine-tuned baselines on the GLUE benchmark {% cite guo_2020, bitfit_2021 %}.
{% cite ansell_2021 %} learn sparse, real-valued masks based on a simple variant
of the Lottery Ticket Hypothesis {% cite frankle_2018 %}: First, they fine-tune
a pre-trained model for a specific task or language, then select the
subset of parameters that change the most which correspond to the
non-zero values of the diff vector $\delta_{\text{task}}$. Then, the
authors set the model to its original pre-trained initialization and
re-tune the model again by only fine-tuning the selected subset of
parameters. The diff vector $\delta_{\text{task}}$ is therefore sparse.

**Comparison to Adapters.** In contrast to Adapters, Sparse fine-tuning
(SFT) does not modify the architecture of the model but restricts its
fine-tuning to a subset of model parameters {% cite guo_2020, bitfit_2021 %}.
As a result, SFT is much more expressive, as they are not constricted to
just modifying the output of Transformer layers with shallow MLP but can
directly modify the pre-trained model's embedding and attention layers
{% cite ansell_2021 %}. Similar to Adapters, {% cite ansell_2021 %} show that their sparse
fine-tuning technique has the same concept of modality found in Adapters
{% cite mad_x %}. Again this modularity can be used in our monolingual setup to
fine-tune on cross-lingual downstream tasks.


# References

{% bibliography --cited %}




