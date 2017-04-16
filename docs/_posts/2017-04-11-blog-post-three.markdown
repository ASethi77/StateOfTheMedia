---
layout: post
title: "Blog Post 3 - Project Proposal"
date: 2017-04-11
description: 
image:        "http://placehold.it/400x200"
---

Below is our project proposal for the Spring 2017 NLP Capstone:

## Motivation
Laws are confusing. I’m an average person. I want to understand what my government is doing in layman’s terms.

## Product
We want to build a tool to help users understand complex congressional bills. We can do that by providing a summary of the key points of a bill written in simple language. Nasty bill text goes in, nice simple summaries come out.

#### Example
As an example of this, feel free to read (or not) the text of S.302 from the first session of the 109th Congress. This is an 8-page document “to make improvements in the Foundation for the National Institutes of Health.” But it’s hard to easily understand the core components of this bill, such as what problems it’s trying to address and how it’s attempting to solve them.

This bill can be summarized to the following:

> Foundation for the National Institutes of Health Improvement Act - Amends the
> Public Health Service Act to revise provisions regarding the National
> Foundation for Biomedical Research. Allows the National Institutes of Health
> (NIH) to accept transfers of funds from the Foundation. Requires the Director
> of NIH to transfer not less than $500,000 and not more than $1,125,000 to the
> Foundation from amounts appropriated to NIH for each fiscal year. Directs the
> Foundation to include in its annual report an accounting of the use of such
> amounts transferred from NIH and to make the report available to the
> appropriate congressional committees.

This is concise and enumerates key components of the bill in language that is easy enough for the layperson to understand.

## Methodologies/Challenges
The task of bill summarization can be broken down into 2 parts, key information identification and extraction simplification. We think pre-processing a bill and taking only the essential parts of the bill will make the resulting summary more clear and focused. Specifically, pre-processing the bill to identify blocks of text that describe the problems a bill is trying to solve or describe a method to approach said problems will eliminate any challenges of summarizing large texts, which may impact the likelihood of selecting to include text which is not important in the summary. The text summary itself is a relatively solved NLP problem, with many techniques available.

#### Key Information Identification
Most of the work will be spent here, trying to figure out an accurate/efficient/general method of obtaining relevant excerpts from congressional text to summarize. This is the most important part of our project, since we are able to utilize the structure of a bill to generate a more accurate summary than just using existing summarization techniques. For now we plan on using some extractive summarization techniques to classify how important certain sentences or phrases are. [3][5]

#### Excerpt Simplification
Once we have a collection of relevant excerpts from a bill, now we have to simplify and combine them into a coherent, but simple-sounding summary of the bill.

One thing we aim to do is reduce the complexity of the language used in the resulting summary. A naive way to approach this for a strawman/MVP would be to build a TF-IDF-based model of individual word complexity and a map of synonyms, followed by word-by-word substitution for any words that meet a certain “complexity” threshold. [4]

As a stretch goal, we are considering using abstractive/generative approaches to creating summaries. Abstractive approaches are less well-studied in research and poses challenges in terms of making generated text both lexically and semantically correct. For this reason, we don’t consider this to be an avenue we will take for our MVP; however, there has been more recent work from Facebook AI Research and Harvard to perform abstractive summarization using an encoder/decoder recurrent neural network architecture [6]. For the second iteration of our project, this could be a more interesting approach to evaluate.

## Evaluation
One way we can evaluate the quality of our summaries is via the ROUGE summary statistic package, which has been commonly-employed for comparing existing text summarization algorithms to summarizations written by humans. We can employ this as a way to test the quality of our summarization because GovTrack provides hand-written summaries for Congress bills. Another option we can consider is using the Word Mover’s Distance (WMD) to find the document distance between our summary and GovTrack’s hand written summary [1].

## References
1. [From Word Embeddings To Document Distances (Description of WMD)](http://mkusner.github.io/publications/WMD.pdf)
2. [Book on existing automatic summarization methods](https://www.cis.upenn.edu/~nenkova/1500000015-Nenkova.pdf)
3. [Previous survey of extractive text summarization methods](https://pdfs.semanticscholar.org/7e30/d0c7aaaed7fa2d04fc8cc0fd3af8e24ca385.pdf)
4. [An Architecture for a Text Simplification System](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1182292)
5. [Extractive Summarization Using Supervised and Semi-supervised Learning](http://anthology.aclweb.org/C/C08/C08-1124.pdf)
6. [Abstractive Sentence Summarization with Attentive Recurrent Neural Networks](http://nlp.seas.harvard.edu/papers/naacl16_summary.pdf)
