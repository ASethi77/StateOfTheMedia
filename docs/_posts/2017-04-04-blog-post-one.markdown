---
layout: post
title: "Blog Post 1 - Project Ideas"
date: 2017-04-04
categories:
  - NLP
description: 
image: https://unsplash.it/2000/1200?image=1003
image-sm: https://unsplash.it/500/300?image=1003
comments: true
---

We are team **????????**, consisting of Aakash Sethi, Drew Dowling, and Kevin Liang. This blog will document our progress through the
UW CSE 481n NLP Capstone. We are interested in pursuing startup-oriented capstone projects; this post summarizes our current project ideas.

### Idea 1: Understanding International Tensions via Twitter

#### Description

The idea behind this project is to map out international relations based on tweets between people, foreign leaders, etc. For example, given a dataset of international tweets within the span of a year, we can map high levels of tension between Russia and Ukraine, and lower levels of tension elsewhere. As a final deliverable, we are considering creating a tool to visualize countries which have large amounts of tension between them. It's important to recognize that tension in this setting can be not only due to diplomatic relations, but also other cultural factors such as xenophobia; however, this tool is agnostic to that distinction.

#### Minimum Viable Action Plan

Design a model for extracting whether tweets/text are directed towards an entity such as a country, as well as a model for sentiment analysis so that we can determine whether positive or negative emotions are directed towards a country. A simple interface for using this tool to start working with is a CLI where users can select a pair of countries A and B and receive a score of how hostile A's sentiment is towards B, normalized between 1.0 to 10.0. Once this much is complete and we are satisified with our model's results, we can work on packaging this functionality into a map-based visualization.

Stretch goal: Look at relations between regions within countries. For example, in Saudi Arabia, divides might exist in the country due to religion while in the Phillippines divides exist because of poverty. However, our tool cannot
make assumptions in terms of causes of divides; enabling the tool to identify polarizing topics within countries as well as the stakeholders within such topics could be a very interesting exploratory addition to this project.

#### Challenges
One key challenge to this project is ensuring that we can pull Twitter data that is relevant to our project. This is especially challenging since Twitter has limits regarding fetching data in certain time windows (e.g. 15 minutes).
We will either need to find a way to guarantee that whatever data we scrape is likely to be relevant for our project, or consider alternative data sources for understanding international relations.

### Idea 2: Describing Congressional Bill's Content in Layman's Terms

#### Description

This project centers around text simplification as a way of summarizing and highlighting key aspects of bills proposed in Congress. We hope creating a tool that extracts the main points of a bill proposal, such as the motivation for the bill and key changes proposed inside bill, makes it easier for the average person to learn about what's happening in their government.

#### Minimum Viable Action Plan

Design a model to identify where key points of a bill exist within a document, specifically what problems the bill is trying to address as well as what *key* changes it's proposing to address those problems. It's also necessary to
create a model that can summarize each of those points and "translate" them into layman's terms. As a starting point, we can look at existing algorithms for creating summaries from excerpts of text, such as SMMRY, as well as TF-IDF-based models for mapping complex language to easier-to-parse equivalents. Once this is complete, we can focus on more generative models for replacing complex text with simpler phrases, rather than using an individual-word-based substitution scheme.

Stretch goals: One stretch goal could be taking advantage of a bill's document structure, such as cross-references for definitions, explanations, etc. for better simplifications. It would also be interesting to consider the possibility of making a more generic language simplification model that can be transferred to other domains beyond bills or legal text.

#### Challenges
One problem we forsee is that bills have lots of domain-specific text. Finding a way to make a model that creates good results despite all the domain-specific language will be an interesting challenge to explore.

### Idea 3: Developing Reading Comprehension Quizzes

This project is intended to identify key aspects of some text, e.g. an article, fictional passage, etc. and generate questions for testing a reader's comprehension of that text. For example, this tool could be used to generate questions such as "Where did the protagonist x go in part y of the text?"

#### Minimum Viable Action Plan

Design a model that identifies key entities or events within a text, as well as a model that can generate natural-sounding questions based on such key events. We are constraining the MVP to ask questions that are based on relatively
self-contained events in text, but a stretch goal could definitely be to expand this tool to generate questions that are based on a more complete/overall understanding of the text rather than a relatively narrow section.

#### Challenges

After our discussion with Yejin, we discovered that this domain is actually an open project within NLP research, so it would certainly be important to get up to speed with state-of-the-art models for reading comprehension and
understand where room for improvement exists in this space.

*Note*: As students coming into the NLP capstone without prior NLP experience, we are working on learning more about classical and emerging NLP methods in order to tackle these projects. For right now, it is challenging for us
to state explicitly which models or techniques we will be using for each project's minimum viable action plan, but are working to flesh this information out very soon.

Thanks for reading! You will be able to follow the progress of our project [via this Github Repo](https://github.com/ASethi77/musical-octo-parakeet).

{% if page.comments %}
<div id="disqus_thread"></div>
<script>

/**
*  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
*  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables*/

var PAGE_URL = "https://asethi77.github.io/musical-octo-parakeet";
var PAGE_ID = "BLOG_POST_1";

var disqus_config = function () {
this.page.url = PAGE_URL;
this.page.identifier = PAGE_IDENTIFIER;
};

(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = 'https://https-asethi77-github-io-musical-octo-parakeet.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>

{% endif %}
