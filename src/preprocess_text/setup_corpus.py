# coding: utf-8
from article_parsers.webhose_article_parser import WebhoseArticleParser
import textacy

def setup_dev_corpus(max_articles=100, use_big_data=False):
	myparser = WebhoseArticleParser("/opt/nlp_shared/data/news_articles/webhose_english_dataset/")
	i = 0
	doc_list = []
	for doc in myparser.yield_articles():
	    i += 1
	    doc_list.append(doc)
	    if i >= max_articles:
		break
	    
	corpus = textacy.Corpus("en", docs=doc_list, big_ass_data=use_big_data)
