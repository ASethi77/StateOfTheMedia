---
layout: post
title: "Blog Post 8 - Advanced Model Update #2"
date: 2017-05-16
description:
image:        "http://placehold.it/400x200"
---

# Progress With Demo
We’ve made some progress building out the back end of our demo with Flask. Last week, the entire demo was an angular app with stubbed out API calls, but now we’ve hooked it up to our model to make approval rating predictions based on the articles given throughout the day. 

# Hyperparameter Tuning
We’ve been messing around with our generic hyperparameters (that is, hyperparameters that are common to both our linear regression model and our feed-forward neural network regression model). One is averaging sentiment and topic strength over some window of days, this is because news from one day doesn’t get forgotten the next day (or period of days). Another is a lag hyperparameter, where we predict an approval rating N days after the news was published. After looking at a large number of combinations of these parameters, we’ve found that there is no significant correlation between these parameters and overall accuracy. 

We also tuned our hyperparameters for our specific to our feedforward network regressor - specifically the number of nodes in our hidden layer and the learning rate. We found that around 140 hidden nodes, our regression has the
minimum MSE of around 19 (testing on a validation set of 95 days to predict approval ratings for). The learning rate used did not have a significant impact on our MSE.

# Exploring final possibilities for advanced models
Right now, our model describes articles as a mixture of topics and assigns an overall sentiment value for that article. For our final advanced model, we had wanted to develop a model where our feature vectors are no longer just overal sentiments towards articles and weights of individual topics, but also sentiments towards individual topics and sentiments between relationships between topics. We felt that modeling such a graph would be an interesting way of testing our theory of whether having a better understanding of interactions between topics and entities in articles would help us make more accurate predictions. Another avenue we were exploring towards a more advanced sentiment analysis model
was a word-by-word LSTM regressor that would assign sentiments towards news articles based on the headline, or perhaps first 2 sentences. We hoped that this might be a better sentiment analysis compared to the MPQA lexicon, since it does not depend on a lexicon that was developed independently of the articles themselves.

We investigated the possibilites of implementing a topic relationship graph using DBPedia Spotlight for NER. We found a paper by Chen et. al, *[Probabilistic Topic Modelling
with Semantic Graph](https://fajieyuan.github.io/papers/ECIR16.pdf)* that could be helpful here, but are concerned that this will simply take more time than we have to develop and test the model implementation. Regarding the LSTM-based
sentiment analysis improvement, we discovered the GDELT dataset which assigns tone values from -100 to 100 for millions of articles, but we are concerned that scraping the articles for text will simply take too much time and
expose us to challenges we faced with the WebHose dataset we were using weeks ago.
