---
layout: post
title: "Blog Post 6 - Advanced Model Update 1"
date: 2017-04-25
description:
image:        "http://placehold.it/400x200"
---

# Summary of improvements over the past week

Over the past week, we have worked a lot on:

1. Investigating potential shortcomings in our previous model/dataset
2. Fixing large pain-points in our training process due to third-party NLP libraries and
3. Expanding on the functionality of our linear regression model

## Observed pitfalls in previous work

### Dataset

One thing we have noticed is that many of the articles we have within the WebHose
political news article corpus are incomplete or are not related to politics entirely.
For example, out of a sample of 1000 documents in our corpus, we found articles such as
this in our dataset:

> 'TORONTO ? Chalk up another 30-plus night for DeMar DeRozan, and another victory for the Toronto Raptors.'

or this:

> Written by Jyoti Sharma Bawa | New Delhi | Published:November 9, 2016 6:00 pm Rock On 2 cast ?', 'Farhan Akhtra, S
hraddha Kapoor, Arjun Rampal, Purab Kohli and others came together to promote their upcoming film.', 'Intellectuals,
 thought leaders and all-around serious guys.

Out of a random sample of 20 articles from the *political* news dataset, we found 11 articles that were
about completely unrelated topics such as sports, movies, or even advertisements. Of course, this is
not necessarily stating that over half of our dataset is unrelated to politics, but our general observations
of the kinds of text contained in articles from the political news corpus suggest that the dataset is
extremely noisy.

To fix this problem, we worked on taking a dataset of 20 years' worth of New York Times articles, 
throwing out articles that are unlikely to be relevant (e.g. sports, lifestyle, arts), and preparing
the articles to be in a data format suitable for training. We finished preparing this corpus, but have not
finished training a model on this dataset yet.

### Sentiment analysis

As described in our last blog post, we follow a basic approach of computing sentiment as

$$ tanh(\frac{\texttt{count(positive words)}}{\texttt{count(negative words)}}) $$

where positive and negative words were "signal words" defined by the MPQA sentiment lexicon dataset. This was
problematic because the \\( tanh \\) function "saturates" quickly for values greater than 2
or less than -2. In order to appropriately normalize sentiment values for articles, we changed our
model to compute \\[ \texttt{count(positive words)} / \texttt{count(signal words)} \\] if there are
more positive words in the article, or \\[ - \texttt{count(negative words)} / \texttt{count(signal words)} \\]
if there are more negative words in the article. This allows us to normalize sentiment values between
-1 and 1, but prevents us from "saturating" the strength of the sentiment.

## Improvements in model and infrastructure

### Handling delays between reading articles and updating approval ratings

An important aspect we felt our model was lacking was the ability to accommodate for the fact that there is
likely a delay between the time news is released about political or economical events and the time that
significant events impact presidential polls/ratings. In order to handle this, we decided to add the
topic counts and sentiment measurements for all days within some window of time beforehand to the current
day's feature vector. We used a window of 15 days over which to aggregate feature vectors.

### Fixing pain-points in infrastructure

While working on this project, we found ourselves fighting with the textacy NLP library quite a lot.
Specifically, textacy turned out not to be very robust, and it crashed often while decompressing or processing
large corpora. This week, we wrote the functionality we needed from textacy from scratch, enabling us
to properly manage saving and loading large corpora and models. For some functions, such as sentence extraction,
we took advantage of NLTK. 

### Visualizing model predictions

Last week, we were relying entirely on `scikit-learn`'s `cross_val_score` functionality for performing
k-fold cross validation with our model and evaluate its prediction quality. However, this didn't give us
a clear way to understand in what sense our model might have been off. For example, it could be the case that
our model generally follows the trends that the actual presidential approval ratings followed but were offset,
or it could be the case that certain outliers in approval ratings skewed our model. To fix this, we
added plotting so that we could visualize a set of approval rating predictions against test cases and
see exactly how inaccurate our model might be.

## Results

Due to time constraints, we trained our updated model using the Webhose political news dataset again
instead of the NYT article dataset.

We again used 4-fold cross validation through `scikit-learn` to evaluate the quality of our updated
regression model. Again, our model attempted to predict both the approval *and* disapproval ratings
independently using feature vectors containing the weights of a fixed set of topics for the day, plus that
day's aggregate sentiment value.

Our 4-fold cross validation score showed the following R^2 values, with 1.0 being a perfect fit and
\\[ -\infty \\] being the worst possible fit:

* Split 1: -6.66
* Split 2: -0.25
* Split 3: -0.46
* Split 4: -25.83

With this, we thought our model had again failed at successfully predicting presidential approval ratings, but
when we plotted approval ratings, we found that our model's predictions often were very close to the actual
approval ratings for our test set of days (actual approval rating percentages are in red, predicted
approval rating percentages are in blue):

![Plot of predicted (dis)approval ratings vs. actual ratings for 8 different days](http://i.imgur.com/lDV7Uxu.jpg)

For several days - namely, days 2, 5, and 6 - the predicted approval ratings are actually very close to the
actual approval ratings. The major outlier in our dataset is the 7th day in our test data, where our
model predicts an ~10% higher approval rating than the actual approval rating. We see an even better fit
in general for the predicted disapproval ratings, save for the 7th day. We believe that such outliers are
skewing our `sklearn` cross-evaluation R^2 scores, since the scores appear to suggest no correlation between
our model predictions and the actual approval ratings, while we can see that our model is clearly able to
predict approval ratings with at least some accuracy.

We also tried adjusting our model to predict *only* the approval rating percentage. Our approval rating
dataset contains separate approval, disapproval, and neutral ratings per day, which is why we originally
predicted both approval and disapproval ratings separately, but we tried simplifying the problem. Plotting
our model's predictions of approval ratings vs. our test set, we found that our model was generally able to
follow the same patterns/trends that the actual approval rating had:

![Plot of prediction vs. actual approval ratings when model only predicts approval ratings](http://i.imgur.com/KLwiub2.png)

Again, our cross evaluation scores from `scikit-learn` showed no correlation between our model and the actual
approval rating values.  Later on this week, we will be looking more carefully at what evaluation metrics are more appropriate for our model,
since the cross-evaluation scores provided by `scikit-learn` do not seem to agree with the trends we see in
plots of the model predictions on our test set.
