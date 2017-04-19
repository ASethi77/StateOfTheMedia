---
layout: post
title: "Blog Post 3 - Project Proposal"
date: 2017-04-11
description: 
image:        "http://placehold.it/400x200"
---

### Strawman I

### Project Pivot

Midway through the previous week we decide that our previous project had a very unclear progression. We initially planned on having an MVP that would focus on reducing HR Bill text using excerpting strategies, but upon further investigation into the literature it seemed to us that this was a rather trivial problem that had already been solved with other open source algorithms/libraries (just passing an HR Bill through smmry.com gave a very reasonable result). By contrast, our extensions from that included abstraction-based summarization and rephrasing, which fit into the other end of the spectrum as largely unsolved problems with little research traction so far. Given our experience and time constraints, we decided it would be better to abandon the project idea of summarizing HR Bills.

### New Project
Given that we have access to datasets of news articles (NYTimes and various other news organizations), we are instead deciding to explore the problem of training a model to predict presidential approval given current news articles. We are using presidential approval ratings from the past couple of decades as our labels (which have been sourced from various polling organizations by the Roper Center at Cornell University). We foresee that this is a more tractable problem with various different routes and approaches to try (such as performing sentiment analysis, sequencing the language of an article to determine subjects, actions, and events, and/or building topic graphs to understand the interrelationship between subjects in the article).

### Our Progress
Since we pivoted rather recently, our goals for this week were to 1) start brainstorming lines of thought for techniques we could apply to our model and 2) build a simple pipeline to parse our data/labels, train a simple model via the Scikit Learn framework, and gather some baseline data (we assume that initial results will be very bad given the simplicity of our first model).
