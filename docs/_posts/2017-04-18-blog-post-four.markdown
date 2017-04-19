---
layout: post
title: "Blog Post 4 - Strawman 1 Update"
date: 2017-04-11
description: 
image:        "http://placehold.it/400x200"
---

# Strawman I

## Project Pivot

Midway through the previous week we decide that our previous project had a very unclear progression. We initially planned on having an MVP that would focus on reducing HR Bill text using excerpting strategies, but upon further investigation into the literature it seemed to us that this was a rather trivial problem that had already been solved with other open source algorithms/libraries (just passing an HR Bill through smmry.com gave a very reasonable result). By contrast, our extensions from that included abstraction-based summarization and rephrasing, which fit into the other end of the spectrum as largely unsolved problems with little research traction so far. Given our experience and time constraints, we decided it would be better to abandon the project idea of summarizing HR Bills.

## New Project
Given that we have access to datasets of news articles (NYTimes and various other news organizations), we are instead deciding to explore the problem of training a model to predict presidential approval given current news articles. We are using presidential approval ratings from the past couple of decades as our labels (which have been sourced from various polling organizations by the Roper Center at Cornell University). We foresee that this is a more tractable problem with various different routes and approaches to try (such as performing sentiment analysis, sequencing the language of an article to determine subjects, actions, and events, and/or building topic graphs to understand the interrelationship between subjects in the article).

## New Project Progress Update
Since we pivoted rather recently, our goals for this week were to 1) start brainstorming lines of thought for techniques we could apply to our model and 2) build a simple pipeline to parse our data/labels, train a simple model via the Scikit Learn framework, and gather some baseline data (we assume that initial results will be very bad given the simplicity of our first model).

### Data Collection

#### News Articles
We are using a dataset of ~500,000 English news articles from 2016. We pulled these articles from [WebHose.io](http://webhose.io/datasets). These contain not only the text of articles, but also metadata such as entities in the article, sentiments, and more. However, we are focusing on using raw text from articles for our purposes. Hannah also gave us access to her dataset of ~20 years' worth of NYT articles, which we hope to take advantage of soon. Thanks Hannah!

#### Presidential Approval Ratings
We were able to collect polls of presidential approval ratings from the F.D.R to the Trump Administration via the [Cornell Roper Center](https://presidential.roper.center/). Each poll in the dataset spans multiple days, and for any given day there may be multiple approval rating polls. We took advantage of this to get "average" approval ratings for each day, and then mapped these approval ratings to contents of articles for that same day.

### Data Pipeline
We spent a good chunk of the past week looking into frameworks that would be helpful for our new project, such as tools for NER, sentiment analysis, etc. In the process, we ran into a library called [textacy](http://textacy.readthedocs.io/en/latest/), which provides LOTS of helpful functionality, including:

1. Loading and saving corpuses in a consistent format
2. Performing POS-tagging, NER, and extracting key terms from documents using algorithms like TextRank
3. Building semantic-network relationships between entities in documents

While we were playing around with textacy, however, we ran into a number of issues with decoding UTF-8 characters in articles, serializing/saving corpora, and minimizing a corpus' memory footprint so that we could take advantage of all our data (using textacy's existing memory management model, we ran out of memory loading a 3 GB dataset on a machine with 32 GB of memory). Overall, we believe that the time we spent learning (and fixing) textacy will enable us to take advantage of the complex high-level functionality that textacy provides for more interesting prediction models.

### Basic Strawman Model
A trivial model for attempting to predict presidential approval ratings based on articles is to use a bag-of-words for a days' worth of articles as features for a regression model, coupled with the presidential approval/disapproval ratings as percentages as output labels for each day. We implemented this using scikit-learn's DecisionTreeRegressor to test this out.

As expected, this model shows no significant correlations between word frequencies for a days' worth of articles and the president's approval rating for that given day. We tried tuning the max-depth of our decision tree regression model as well, but (unsurprisingly) found that this did not significantly improve our results. Our model also tended to overfit our training examples extremely quickly due to the sparsity of our inputs (> 1000 unique words for the set of articles within a given day).

### Moving Forward: Better Models

As a result of the work necessary to set up our data pipeline, we're now able to utilize textacy's semantic graph features in the way we originally wanted to.
