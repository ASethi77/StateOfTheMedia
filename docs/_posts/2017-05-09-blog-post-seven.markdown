---
layout: post
title: "Blog Post 6 - Advanced Model Update 1"
date: 2017-04-25
description:
image:        "http://placehold.it/400x200"
---

# Blog Post 7 - Advanced Model Update #1 (cont)
May 9th, 2017


## Summary of improvements over the past week:

1. Built out part of the front-end for a tool/demo
2. Analysis Cache
3. Started using NegMeanSquaredError as evaluation function
4. Experimented using LDA/NMF topic model
5. Non-linear regression model


## Demo: Aakash’s Face

Since we’re technically in startup mode, it makes sense to build some kind of *product* around the NLP work that we’re doing. We decided on making a simple web-app that takes articles, and responds with a visualization of the resulting sentiment analysis and topic model:

![Sad Aakash](/StateOfTheMedia/images/Aakash_Happy.png)

![Happy Aakash](/StateOfTheMedia/images/Aakash_Sad.png)


## Performance improvements: Analysis Cache
The process of loading in a corpus, generating topic/sentiment features, and training takes a long time. Generating topic/sentiment feature vectors takes particularly long, so we’ve implemented a cache system to store these features for a particular corpus, so we can instantly swap out regression models for experiments.


## New evaluation function: Negative Mean Squared Error
After noticing the discrepancy between our poor R^2 values and our plotted results from last week:

![Plot of predicted (dis)approval ratings vs. actual ratings for 8 different days](/StateOfTheMedia/images/Approval_Ratings_Linreg_1.png)

We’ve been using scikit-learn’s negative mean squared error function as our new evaluation function, and since then our results have been more promising.


## New topic model: LDA/NMF
Rather than using a hand-compiled set of topic words, we tried using LDA and NMF to generate topics and topic words for us. After running it over our political corpus from WebHose, we found that it did a pretty good job of grouping words for a while, but some of the topics it came up with were more scattered. We also found that there were some words present in all topics that were generated, such as “Mr.” Maybe after a bit of filtering we will be able to add the resulting topics/words into our topic model.


## New regressor: Multi-layered Perceptron
It’s there. We implemented it, but haven’t run experiments with it yet.

