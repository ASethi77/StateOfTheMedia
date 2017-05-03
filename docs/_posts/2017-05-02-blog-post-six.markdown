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

# Observed pitfalls

## Dataset

One thing we have noticed is that many of the articles we have within the WebHose
political news article corpus are incomplete or are not related to politics entirely.
For example, out of a sample of 1000 documents in our corpus, we found articles such as
this in our dataset:

> 'TORONTO ? Chalk up another 30-plus night for DeMar DeRozan, and another victory for the Toronto Raptors.'

