---
layout: post
title: "Blog Post 9 - Advanced Model Update #2 (continued)"
date: 2017-05-23
description:
image:        "http://placehold.it/400x200"
---

# Progress with Demo
Since we had our presentation this week, we spent time working on improving
our demo tool. The interface for it is largely the same as it was when
we introduced it, but the backend has been fully implemented; users can
now submit articles and receive topic/sentiment breakdowns for articles as
well as a predicted approval rating given those articles. We also spent
time working on ways to allow users to inspect the feature vectors
(i.e. topics/sentiments) for individual articles to understand their
contribution to the overall day's prediction. Unfortunately, we were unable
to complete this portion of the demo tool this week due to time constraints,
but we plan to complete this and improve our demo tool over the next week.

We envision the final tool having the following new features:

1. Cards that enable users to see the individual contributions of articles
   to the final prediction value and how we classified it by topics/sentiments.
2. Live news stream for recent data (e.g. past 2 weeks) and a plot of our
   predictions against Trump's actual approval ratings. Having a live stream
   of news can also help buffer "user-submitted" articles. To do this, we
   are looking at taking advantage of the (Live News API)[https://newsapi.org]
   and/or RSS feeds for news sites.
3. A historical plot that allows us to look at any given window of time within
   our approval ratings dataset and overlay our model's predictions for that
   window if articles are available.

# LSTM-based regression model
One point that we've mentioned previously is that news is not forgotten
immediately; to account for this, our current models use a sliding window
average over `N` days' feature vectors. However, these are equally-weighted,
which is likely not true in practice. Also, there's a discrete drop-off where
the oldest days' feature vector is thrown away.

To solve this, we wanted to use an LSTM, where each LSTM cell's input is
a feature vector for one of those `N` days. Therefore, our new regression
model architecture is `N=5` LSTM cells, each taking the same feature vectors
as we have been using, coupled with a 3-output softmax layer to predict
approval, disapproval, and neutral ratings, respectively. We built this model
using CNTK, and trained it using ~7000 epochs over the Clinton dataset, which
is ~1000 days of training examples.

Below is a plot of our **training set** approval, disapproval, and neutral
predictions vs. actual values after our LSTM was done training.

![Training set prediction vs. actual (LSTM)](http://i.imgur.com/RjGKMKV.png)

We can see that, as with previous models, our LSTM appears to be capable
of following basic trends (i.e. did the approval rating increase or decrease?)
but fails to predict the appropriate magnitude. There are several reasons
this could happen:

1. Most large fluctuations of +/- 20% often recover back to the norm
   within three days, which could make it difficult for our LSTM to find
   consistent long-term trends.
2. Given the fact that we are using RMSE as a loss function, the LSTM is
   likely "discouraged" from making extreme predictions in case it is wrong.
   This could potentially be solved using a hinge-loss where the loss has
   a ceiling value for any given training example.

We also plotted our train, test, and validation loss (MSE):

![Training set prediction vs. actual (LSTM)](http://i.imgur.com/RjGKMKV.png)

Note that after about 500 epochs, the validation and test loss appear to
converge/stagnate. Our MSE for the validation loss appeared to drop by about
20% down to 1.2 in 134 validation examples. The MSE for the test loss dropped
by about 18% over 578 test examples. This is a relatively low improvement
overall, and can at least partially be attributed to the fact that we have
so few training examples. This is a limitation of our domain, since we can
only have one example per day at best given how we frame the problem.

# Moving forward

One thing we haven't done with the LSTM that we had originally wanted to
do is make the LSTM variable length so that we can eliminate the hyperparameter
of how many days of articles we compound into a single feature vector.

We also intend to explore other potential feature vectors, such as training
an LSTM regression model based on embeddings of words in documents, and
allowing the embeddings to "discover" important relationships between
words or entities in articles as the model trains. We realize that there
is a lot of information loss as a result of the way we select features
and aggregate those features for distinct articles into just one days' feature
vector. For example, since we might have hundreds of diverse articles for
any given day, after averaging out all topic weights and sentiments, we may
not have particularly distinguishable feature vectors for different approval
ratings. In general, finding a way to:

1. Expand the number of training examples available to us by changing our
   problem framing
2. Changing our feature space to capture nuances of individual articles

is a high priority for us at this point.
