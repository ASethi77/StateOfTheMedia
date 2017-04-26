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

For our first strawman, we chose to use an aggregated bag of words for each day of articles as features to a decision tree regressor to predict presidential approval ratings. Each input vector was the count of each word that occurred during that day. Not surprisingly, this approach didn’t work very well.

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

### Identifying the sentiment of articles
For each article, we compute the overall sentiment of an article by first identifying “positive” and “negative” words in the article as defined by the MPQA subjectivity lexicon [2]. Once we count the number of positive and negative words, we take the tanh of the difference to measure the article sentiment between -1.0 and 1.0.

### Completing the model
Once we have identified the “main” topic of an article, we concatenate its one-hot encoding and the sentiment of the article as described above. This is the general format of our input vector.

The original paper’s dataset was gathered by collecting hundreds of thousands of *relevant* tweets per day over the course of a year. They aggregated topic/sentiment over the course of a day, but because we do not have nearly the scale of articles they do on a daily basis (we have ~100k relevant articles since Oct. 2015), we decided not to aggregate our examples in such a fashion.

## Results

Refined and tested our linear regression model on our 100k article dataset using 4-fold cross validation. After training and fitting our linear regression model, we computed the r^2 values of our model vs. the actual presidential approval ratings and found the following r^2 values for each fold:

-0.64290934, -0.76422136, -0.1494231, -0.3783654

This was initially surprising, since the original paper upon which we based our model was able to achieve an r^2 value of 0.73 in some cases (note that they also used a “lag” window hyperparameter, which we did not use). In contrast, our r^2 values suggest that there was no correlation between our features/topics or sentiments and presidential approval ratings. We believe that there are a number of factors that contribute to this result; they are described below.

## Error Analysis

The main limitation of our model is a significant lack of hits for sentiment signal words. When viewing a random sample of feature vectors that were being fed to our model as training data, a large majority of the sentiment values were heavily leaning towards the extremes. Even when we expanded the text being inputted to the full length of the articles, we saw no significant changes in R^2 values. One possible explanation is that news articles are inherently more neutral in their tone than tweets, so positive and negative lean is much more subtle. 
Another limitation is the sophistication to which we can accurately discern topics from signal words. At the moment we currently look for exact case insensitive matches of approximately 200 manually curated signal words for the topics. Some topics have disproportionately more signal words than others. There are also words that may have overlap between topics (such as “negotiation” between Foreign Relations and Economics). Open questions exist for where we should categorize utility words like “regulations”. And lastly, the signal words for topics themselves inherently have positive or negative sentiment skew that is unaccounted for in our model (for example both “peace accord” and “genocide” would contribute to the Foreign Relations label equally). Lastly, for topic extraction there is a question of what level of granularity should we branch our topics into (for example should “Race Relations” be a separate label or should its signal words be left under Social Issues).
Because of the variety of articles we have in our dataset, and to some extent variety of quality sourcing as well, there is quite a lot of noise that does not contribute to our model. Although we remove entries that have zero contribution to any of the topic labels, we theorize there are still plenty of articles in the corpora that are minimally related to politics or presidential actions. This is completely different from say a twitter dataset where filtering for something like the name of the president can remove most junk. In addition each twitter user has equal credibility and impact as any other twitter user. We currently do not consider reputability of the journalistic source or the disproportionate impact of large scale real life events (say for example a stock market crash or declaring war). Eventually we need to find some way of weighting articles as well as identifying key events being talked about through the context of the articles. 

# References

1. [O'Connor, B., Balasubramanyan, R., Routledge, B. R., & Smith, N. A. (2010). From tweets to polls: Linking text sentiment to public opinion time series. ICWSM, 11(122-129), 1-2.](https://homes.cs.washington.edu/~nasmith/papers/oconnor+balasubramanyan+routledge+smith.icwsm10.pdf)
2. [Theresa Wilson, Janyce Wiebe, and Paul Hoffmann (2005). Recognizing Contextual Polarity in Phrase-Level Sentiment Analysis. Proc. of HLT-EMNLP-2005.](https://asethi77.github.io/StateOfTheMedia/2017/04/25/blog-post-five/)
