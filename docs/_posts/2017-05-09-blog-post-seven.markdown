---
layout: post
title: "Blog Post 7 - Advanced Model Update #1 (cont)"
date: 2017-05-09
description:
image:        "http://placehold.it/400x200"
---

## Summary of improvements over the past week:

1. Built out part of the front-end for a tool/demo
2. Analysis Cache
3. Started using NegMeanSquaredError as evaluation function
4. Experimented using LDA/NMF topic model
5. Non-linear regression model


## Demo: Online sentiment analysis + topic extraction

Since we’re technically in startup mode, it makes sense to build some kind of *product* around the NLP work that we’re doing. We decided on making a simple web-app that takes articles, and responds with a visualization of the resulting sentiment analysis and topic model:


Here is our simple web app

![Overall App](http://i.imgur.com/VHbKZ0R.png)

It is run by our mascot Aakash! (He's very emotional and will react in kind to the kind of content that we feed our model)


![Sad Aakash](http://i.imgur.com/h3lCGvP.png)

![Happy Aakash](http://i.imgur.com/j6E9wgv.png)


Here is an example given a very strong toned article about immigrants choosing to avoid access to federal food resources.

![Immigrants Article](http://i.imgur.com/7wmkeRK.png)

In this article many trigger words like "fear", "avoid", and "deportation" convey a very strong negative sentiment, while words such as "nutrition" and "services" probably align well as a social issue topic. Although we do not specifically look for entities in our model, actors like "US", "Donald Trump", and "Wic" would give strong political/foreign relations context. Overall this article is not as difficult to analyze because unlike most news articles it 1) has a very strong tone/bias in the negative direction and 2) contains highly specialized terms that our model can infer topical context from (as a whole news articles tend to be very neutral and may contain a wide ranging amount of noisy filler words). After adding the article to the set for the day, we see some results like below.

![Immigrants Sentiment](http://i.imgur.com/h3lCGvP.png)

As you can see, the article tone is indicated as "Very Sad" and the strongest indicator is that of "Social Issues". Given more articles, the results will amalgamate the labels and create an overall impression for the day.

Below is another example for a more upbeat/less politically charged "article" about the tech industry.

![Tech Article](http://i.imgur.com/tnP7fJG.png)

![Tech Sentiment](http://i.imgur.com/Wdgn7tr.png)

![Tech Overall Page](http://i.imgur.com/VHbKZ0R.png)


## Performance improvements: Analysis Cache
The process of loading in a corpus, generating topic/sentiment features, and training takes a long time. Generating topic/sentiment feature vectors takes particularly long, so we’ve implemented a cache system to store these features for a particular corpus, so we can instantly swap out regression models for experiments.

This is also helpful for letting us test different hyperparameter configurations. So far, our model has
two main hyperparameters:

1. A delay between the day we read articles and the day we predict approval ratings for. We added this parameter because it is intuitively not likely that poll-based approval ratings for a president reflect new information published on the same day. So far, we have been testing our models with a "delay" of 1.

2. A "window" over which we aggregate sentiments and topic feature vectors. Intuitively, it makes sense that approval ratings would reflect news over (at least) several days, not just the news articles from one day. To implement this, each day's feature/topic vector is the average over the last **n** days' raw feature/topic vectors (that is, before averaging). We have been testing our models with a "window" of 15 days including the current day.

## New evaluation function: Negative Mean Squared Error
After noticing the discrepancy between our poor R^2 values and our plotted results from last week:

![Plot of predicted (dis)approval ratings vs. actual ratings for 8 different days](/StateOfTheMedia/images/Approval_Ratings_Linreg_1.png)

We’ve been using scikit-learn’s negative mean squared error function as our new evaluation function, and since then our results have been more promising.


## New topic model: LDA/NMF
Rather than using a hand-compiled set of topic words, we tried using LDA and NMF to generate topics and topic words for us. We believed this would give us a better sentiment analysis dictionary overall since each word returned by the LDA and NMF models are guaranteed to appear in some documents in our corpus. We trained both topic extraction models over all NYT articles from 1993-2001 (the Clinton administration) using 15 topics and 1000 input features.

We found that the topics discovered through NMF seem to be much clearer than those discovered through LDA. Below is an example of one of the 15 topics we discovered through LDA:

> city said new year state people million area drug car water official center building local mile national group plant mayor say york like project resident county park problem land food california...

Compare this to an example of the topics we are able to get through NMF:

> court judge case law justice federal supreme lawyers trial state legal ruling decision lawyer jury cases department rights police district criminal attorney evidence government charges states death ms investigation civil filed prison

In general, we find that there are far more low-quality topics discovered through LDA for our use case, and for topics such as foreign-relations, the LDA model often identifies multiple topics which could all be construed to be the same overall topic, which is not as much of a problem with LDA.

In either situation, however, the usage of the topics extracted through these methods require us to cherry-pick topics and keywords and put them in a more usable data format, which we have not had the time to do yet.

## New regressor: Multi-layered Perceptron
It’s there. We implemented it, but haven’t run experiments with it yet.

