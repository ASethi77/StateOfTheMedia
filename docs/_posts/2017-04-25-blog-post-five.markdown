---
layout: post
title: "Blog Post 5 - Strawman 2 Update"
date: 2017-04-25
description:
image:        "http://placehold.it/400x200"
---

# Strawman II

## Summary of Strawman Methods

In case you missed our last blog post, we recently switched away from creating a project to summarize bill text,
and intead began working on a project to predict presidential approval ratings given news articles.

We used two different models as strawman approaches to the problem:

1. A decision-tree regressor and
2. A linear regression model

The first model is one we used mostly as a way of verifying that our pipeline of loading in a corpus, extracting
basic features, etc. was correct. We did not really expect to get good results out of the first model as a result.
The second model is actually based on prior research, and we believed that this model would perform reasonably well
(i.e. significantly better than chance, though not necessarily > 85% accuracy); more on this below.

## First model: Decision Tree Regression

## Second model: Linear Regression

We chose a linear regression model based on previous research from O'Connor et. al. [1]. In a nutshell, this paper
is able to fit Twitter messages and their corresponding sentiments to Gallup poll approval ratings via a linear
least-squares regression model. They break down the problem into the following tasks:

1. Identifying key topics that might be correlated with Gallup poll ratings, and subsequently identifying tweets that
   relate to those topics;
2. Identifying the sentiment of tweet with respect to that topic
3. Setting up a linear regression model where inputs are vectors of topics and the overall sentiment tweets express
   towards those topics for all tweets in a given day. Outputs to predict are, of course, the presidential approval
   rating for that day.

While we used this as guidance for our linear regression model, we chose to diverge in some key ways. The way we handle
each sub-problem and any differences between our model and the one presented in the paper is discussed below.

### Identifying key topics
The original paper uses hard-coded search terms for different inputs, depending on the topic they are trying to predict;
for example, if the paper is trying to predict president approval ratings specifically during the years 2008-2009,
they filter all tweets that contain the word "Obama" [1]. We followed a similar heuristic approach, but we chose to filter
topics based on topics we thought would be correlated to approval ratings; specifically, we added the following topics:

* world leaders
* countries
* politics
* economics
* foreign relations
* social issues
* environmental issues

For each topic, we assigned a set of keywords relevant to that topic and searched for them in each article; the
topic with the most matching keywords (in an absolute sense, i.e. not the most **unique** keywords matching)
was assigned to that article via a 1-hot encoding.

### Identifying the sentiment of tweets
# Identifying the 

## References

1. [O'Connor, B., Balasubramanyan, R., Routledge, B. R., & Smith, N. A. (2010). From tweets to polls: Linking text sentiment to public opinion time series. ICWSM, 11(122-129), 1-2.](https://homes.cs.washington.edu/~nasmith/papers/oconnor+balasubramanyan+routledge+smith.icwsm10.pdf)
