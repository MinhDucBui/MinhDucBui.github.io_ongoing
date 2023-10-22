---
title: "[Project]  Distilling Multilingual Encoders into Monolingual Components"
date: 2022-08-04 12:30:47 +01:00
modified: 2022-08-22 11:30:47 +01:00
tags: [project, nlp]
description: Master Thesis
image: "/assets/img/distilling-multilingual/introduction.jpg"
---
*In this blog post, we motivate and lay out my main contributions of my Master Thesis. To read the full thesis, 
download my thesis <a href="/assets/img/master-thesis/Master_Thesis.pdf" download>here</a>. To 
read the related work sections as blog posts, 
see [Cross-Lingual Representation](https://minhducbui.github.io/cross-lingual-representation/), [Parameter-Efficient Fine-Tuning in NLP](https://minhducbui.github.io/efficient-tuning/)
and [Knowledge Distillation](https://minhducbui.github.io/knowledge-distillation/).*



# Table of Contents
1. [Motivation](#1-motivation)
2. [Research Objective & Contribution](#2-research-objective---contribution)


# 1. Motivation

Natural language processing (NLP) has made significant progress in
recent years, achieving impressive performances across diverse tasks.
However, these advances are focused on just a tiny fraction of the 7000
languages in the world, e.g., English, where sufficient amounts of text
in the respective language are available (high-resource languages).
Nevertheless, when the situation arises where text data in a language is
scarce, language technologies fail. These languages are called
low-resource languages, e.g., Swahili, Basque, or Urdu; see the next Figure for a more apparent distinction between
high-, mid-, and low-resource languages:

<figure>
<img src="/assets/img/master-thesis/nlp_resource_hierarchy.png" alt="interview-img">
<em>A conceptual view of the NLP resource hierarchy categorised in
availability of task-specific labels and availability of unlabeled
language-specific text. Taken from {% cite ruder2019unsupervised %}.</em>
</figure>

This leaves low-resource languages and, therefore, most languages
understudied, which further increases the digital language divide[^1] on
a technological level. Being able to develop technologies for
low-resource languages is vital for scientific, social, and economic
reasons, e.g., Africa and India are the hosts of around 2000
low-resource languages and are home to more than 2.5 billion inhabitants
{% cite magueresse2020lowresource %}. \"Opening\" the newest NLP technologies
for low-resource languages can help bridge the gap, e.g., digital
assistants, or help reduce the discrimination against speakers of
non-English languages
{% cite tatman-2017-gender, rabinovich-etal-2018-native, Zhiltsova2019MitigationOU %}.

To improve language technologies for low-resource languages, the field
of Cross-Lingual Representation Learning is focused on creating
high-quality representations for these languages by gaining benefit from
abundant data in another language via a shared representation space. As
static word representations gained in popularity, many multilingual
embedding methods have been presented
{% cite mikolov2013exploiting, hermann-blunsom-2014-multilingual, hu2020xtreme %}.
The idea behind these methods is to induce embeddings[^2] such that the
embeddings for two languages are aligned, i.e., word translations, e.g.,
*cat* and *Katze*, have similar representations. Recently, however,
large pre-trained language models, the so-called transformer models,
took static word embedding methods over in virtually every aspect,
partly because these models induce context-dependent word
representations, capturing the rich meaning of a word better
{% cite peters2018deep, howard2018universal, Radford2018ImprovingLU, devlin2019bert %}.
E.g., mBERT and XLM-R, transformer-based multilingual masked language
models pre-trained on text in (approximately) 100 languages, can obtain
impressive performances for a variety of cross-lingual transfer tasks
{% cite pires-etal-2019-multilingual, conneau2020unsupervised %}. Even though
these models were not trained with any cross-lingual objectives, they
still produce representations that can generalize well across languages
for a wide range of downstream tasks
{% cite wu-dredze-2019-beto, conneau2020unsupervised %}. To analyze the
cross-lingual transfer ability of multilingual models, the model is
first fine-tuned on annotated data of a downstream task and then
evaluated in the zero or few-shot scenario, i.e., evaluated with the
fine-tuned models in the target language {% cite hu2020xtreme %}  with no or few
additional labeled target language data.

As impressive as these multilingual transformers might seem,
low-resource languages still perform sub-par to high-resource languages
{% cite wu-dredze-2019-beto, conneau2020unsupervised %}, partly due to the fact
of a smaller pre-training corpus {% cite conneau2020unsupervised %} , the curse
of multilinguality {% cite conneau2020unsupervised %}  and the importance of
vocabulary curation and size
{% cite chung-etal-2020-improving, artetxe-etal-2020-call %}. E.g., the curse
of multilinguality argues assuming that the model capacity stays
constant that adding more languages leads to better cross-lingual
performance on low-resource languages up until a point where the overall
performance on monolingual and cross-lingual benchmarks degrades.
Intuitively explained, adding more languages to the model has two
effects: (1) Positive cross-lingual transfer, especially for
low-resource languages, and (2) lower per-language capacity, which then,
in turn, can degrade the overall model performance. These two effects of
*capacity dilution* and positive transfer need to be carefully traded
against each other. The model either needs to have a large model
capacity[^3] or is specialized (constrained) towards a subset of
languages beforehand. For these reasons, it is hard to create a single
model that can effectively represent a diverse set of languages. One
solution is to create language-specific models (monolingual models) with
language-specific vocabulary and model parameters
{% cite virtanen2019multilingual, antoun-etal-2020-arabert %}, but in return,
monolingual models need enough text to pre-train the model on the
language modeling task, which is typically not available for
low-resource languages. Additionally, we can not benefit from any
cross-lingual transfer from related languages, making it harder to
create an adequate representation for low-resource languages
{% cite pires-etal-2019-multilingual, lauscher-etal-2020-zero %}.

# 2. Research Objective & Contribution

In this thesis, we explore how one can alleviate the issues of big
multilingual transformers for low-resource languages, especially the
curse of multilinguality.
Specifically, our two main objectives are: (1) Improving the
cross-lingual alignment for low-resource languages and (2) improving
cross-lingual downstream task performance for low-resource languages. We
utilize Knowledge Distillation (KD) by distilling the multilingual model
into language-specialized (also called monolingual) language models. We
make the following contributions:

- We propose a novel setup to distill multilingual transformers into monolingual components. Based on the setup, we 
propose two KD strategies: One for improving the alignment between two languages and one to improve cross-lingual 
downstream task performance. We call the former `MonoAlignment` and the latter `MonoShot`.
- `MonoAlignment` uses a distillation strategy to distill multilingual transformer models into smaller monolingual 
  components which have an improved aligned representation space between a high-resource language and a low-resource 
  language. We demonstrate the effectiveness by distilling XLM-R and experimenting with aligning English with 
  Turkish, Swahili, Urdu, and Basque.
- We compare `MonoAlignment` to other Knowledge Distillation
    strategies showing that it outperforms them in the retrieval task
    for low-resource languages.
- Our work suggests that an increase in the cross-lingual alignment of
    a multilingual transformer model does not necessarily translate into
    an increase in cross-lingual downstream task performance.
- Therefore, we propose `MonoShot`, another Knowledge Distillation
    strategy to distill multilingual transformer models into smaller
    monolingual components but which have a strong cross-lingual
    downstream performance in the zero- and few-shot settings.
- We show that `MonoShot` performs best among many different Knowledge
    Distillation strategies, albeit still lacks behind the teacher
    performance. However, it outperforms models built upon the teacher
    architecture but is trimmed down to the same size as the distilled
    components and initialized from parts of the teacher.
- We demonstrate an effective fine-tuning strategy for the zero-shot
    scenario for aligned monolingual models and compare it against many
    other strategies.

To conduct our research, we will draw inspiration from the field of
Cross-Lingual Representation Learning, Knowledge Distillation, and
Parameter-Efficient Fine-tuning. Following different Knowledge
Distillation strategies, such as from DistilBert {% cite sanh2020distilbert %} 
or TinyBert {% cite jiao2020tinybert %} , we distill the aligned cross-lingual
representation space of the multilingual transformer model XLM-R
{% cite translation_conneau_2017 %}  into smaller monolingual students. To
fine-tune aligned monolingual models in a zero-shot scenario, we study
the field of parameter-efficient fine-tuning, i.e., Adapters
{% cite houlsby2019parameterefficient, pfeiffer2021adapterfusion %}, BitFit
{% cite bitfit_2021 %}  and Sparse Fine-Tuning {% cite guo_2020 %}. Finally, we evaluate
the general-purpose cross-lingual representation of our monolingual
models in the retrieval, classification, structured prediction, and
question-answering task.






# References

{% bibliography --cited %}


[^1]: <http://labs.theguardian.com/digital-language-divide/>

[^2]: Word embeddings and word representation are interchangeable in our
    thesis.

[^3]: Here: Measured in the number of free parameters in the model.




