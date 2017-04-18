# coding: utf-8
from .article_parsers.webhose_article_parser import WebhoseArticleParser
import textacy
from collections import defaultdict
from dateutil.parser import parse

def setup_dev_corpus(max_articles=100, per_date=True, use_big_data=False):
    myparser = WebhoseArticleParser("../data/Articles/WebHoseDevCorpus/")
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
        for date, docs in date_to_docs.items():
            corpus_for_today = textacy.Corpus('en', docs=docs)
            corpra.append((date, corpus_for_today))
            corpus_for_today.save("../data/TextacyCorpra", str(date))
        return corpra
    else:
        return textacy.Corpus("en", docs=doc_list)
