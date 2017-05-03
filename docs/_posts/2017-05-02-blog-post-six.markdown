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

# Observed pitfalls in previous work

## Dataset

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

## Sentiment analysis

As described in our last blog post, we follow a basic approach of computing sentiment as

$$ tanh(\frac{\texttt{count(positive words)}}{\texttt{count(negative words)}}) $$

where positive and negative words were "signal words" defined by the MPQA sentiment lexicon dataset. This was
problematic because the \\( tanh \\) function "saturates" quickly for values greater than 2
or less than -2. In order to appropriately normalize sentiment values for articles, we changed our
model to compute \\[ \texttt{count(positive words)} / \texttt{count(signal words)} \\] if there are
more positive words in the article, or \\[ - \texttt{count(negative words)} / \texttt{count(signal words)} \\]
if there are more negative words in the article. This allows us to normalize sentiment values between
-1 and 1, but prevents us from "saturating" the strength of the sentiment.
