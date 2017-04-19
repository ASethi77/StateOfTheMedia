# coding: utf-8
from .article_parsers.webhose_article_parser import WebhoseArticleParser
import textacy
from collections import defaultdict
from dateutil.parser import parse
import pickle
import os

def setup_corpus(article_parser_type, article_dir, corpus_name, max_articles, per_date, use_big_data):
    myparser = article_parser_type(article_dir)
    i = 0
    doc_list = []
    date_to_docs = defaultdict(list)
    for doc in myparser.yield_articles():
        i += 1
        doc_list.append(doc)

        if per_date:
            datestr = doc.metadata['published']
            datetime_object = parse(datestr)

            date_to_docs[datetime_object.date()].append(doc)

        if i >= max_articles:
            break

    if per_date:
        corpra = []
        dates = set()
        corpus_dir = "../data/" + corpus_name

        if not os.path.exists(corpus_dir):
            os.makedirs(corpus_dir)

        for date, docs in date_to_docs.items():
            dates.add(date)
            corpus_for_today = textacy.Corpus('en', docs=docs)
            corpra.append((date, corpus_for_today))
            corpus_for_today.save(corpus_dir, str(date))

        with open(corpus_dir + "/dates.json", "wb") as dates_file:
            pickle.dump(dates, dates_file)

        return corpra
    else:
        return textacy.Corpus("en", docs=doc_list)

def setup_dev_corpus(corpus_name="WebHoseDevCorpus", max_articles=100, per_date=True, use_big_data=False):
    return setup_corpus(WebhoseArticleParser, "../data/Articles/WebHoseDevCorpus", corpus_name, max_articles, per_date, use_big_data)

def setup_100M_corpus(corpus_name="HundredMegsCorpus", per_date=True, use_big_data=True):
    return setup_corpus(WebhoseArticleParser, "/opt/nlp_shared/data/news_articles/webhose_english_dataset/WebHoseDataset-16275-Articles", corpus_name, 17000, per_date, use_big_data)

if __name__ == '__main__':
    setup_dev_corpus()
