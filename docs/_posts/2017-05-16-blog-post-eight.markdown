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
We’re doing it. It doesn’t do anything. We’ve been messing around with our generic hyperparameters. One is averaging sentiment and topic strength over some window of days, this is because news from one day doesn’t get forgotten the next day (or period of days). Another is a lag hyperparameter, where we predict an approval rating N days after the news was published. After looking at a large number of combinations of these parameters, we’ve found that there is no significant correlation between these parameters and overall accuracy:

<surface plot with good heatmap here>

# Exploring final possibilities for advanced models
Right now, our model describes articles as a mixture of topics and assigns an overall sentiment value for that article.

