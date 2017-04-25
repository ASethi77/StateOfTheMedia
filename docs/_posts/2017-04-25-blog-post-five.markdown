---
layout: post
title: "Blog Post 5 - Strawman 2 Update"
date: 2017-04-25
description:
image:        "http://placehold.it/400x200"
---

# Strawman II

## Part 1, the Strawmannining
 Not surprisingly, a bag of words in articles for a day doesn't seem to correlate well with presidential approval ratings. We made several changes to try and improve the performance of our regressor. We decided to use a LinearRegression model with better features: normalized politic-centered-topic counts and sentiments in articles throughout a given day.

## Part 2, the Model
 Not much to say here. We were using a sklearn DecisionTreeRegressor in Strawman I and now we're using a sklearn LinearRegressor

## Part 3, the Features
 We've replaced the shitty bag of words representation with more reasonable features. We're looking at polictical-centered-topics such as {{put some topic names here}}, and using untargeted sentiment analysis.
