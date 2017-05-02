# coding: utf-8
from .article_parsers.webhose_article_parser import WebhoseArticleParser
import textacy
from collections import defaultdict
from dateutil.parser import parse
from optparse import OptionParser
import pickle
import os
import sys

def setup_corpus(article_parser_type, article_dir, corpus_name, max_articles, per_date, use_big_data, min_length=5):
    myparser = article_parser_type(article_dir)
    i = 0
    doc_list = []
    date_to_docs = defaultdict(list)
    # where doc is a textacy doc object
    for doc in myparser.yield_articles():
        if (doc.n_sents >= min_length): # ignore junk news (articles that are too short)
            i += 1
            print("Processing doc #: " + str(i))
            if per_date:
                datestr = doc.metadata['published']
                datetime_object = parse(datestr)
                date_to_docs[datetime_object.date()].append(doc)
            else:
                doc_list.append(doc)

        print("Adding articles to corpus: {0}% complete.\r".format(i * 100.0 / max_articles))
        sys.stdout.flush()

        if i >= max_articles:
            break

    if per_date:
        corpora = {}
        dates = set()
        corpus_dir = os.path.join("../data/", corpus_name)

        if not os.path.exists(corpus_dir):
            os.makedirs(corpus_dir)

        for date, docs in date_to_docs.items():
            dates.add(date)
            corpus_for_today = textacy.Corpus('en', docs=docs, big_ass_data=use_big_data)
            corpora[date] = corpus_for_today
            corpus_for_today.save(corpus_dir, str(date))

        with open(os.path.join(corpus_dir, "dates.json"), "wb") as dates_file:
            pickle.dump(dates, dates_file)

        return corpora
    else:
        return textacy.Corpus("en", docs=doc_list)

def setup_dev_corpus(corpus_name="WebHoseDevCorpus", max_articles=100, per_date=True, use_big_data=False):
    return setup_corpus(WebhoseArticleParser, "../data/Articles/WebHoseDevCorpus", corpus_name, max_articles, per_date, use_big_data)

def setup_100M_corpus(corpus_name="HundredMegsCorpus", per_date=True, use_big_data=True):
    return setup_corpus(WebhoseArticleParser, "/opt/nlp_shared/data/news_articles/webhose_english_dataset/WebHoseDataset-16275-Articles", corpus_name, 17000, per_date, use_big_data)

if __name__ == '__main__':
    # add command-line flags
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="dir", help="directory containing corpus data", metavar="DIRECTORY") # by default if not specified
    parser.add_option("-n", "--name", dest="name", help="name of corpus", metavar="NAME")
    parser.add_option("-m", "--max", dest="max_articles", help="max # of articles to pull", metavar="MAX_NUM")
    parser.add_option("-t", "--date", dest="per_date", action="store_true", help="partition data by date (true or false)")
    parser.add_option("-b", "--big", dest="big_data", action="store_true", help="big data, use disk (true or false)")
    (options, args) = parser.parse_args()

    # default values    
    if options.dir != None:
        name = "TEMP_CORPUS"
        data_dir = options.dir
        max_articles = 1000
        per_date = options.per_date
        big_data = options.big_data
        if options.dir != None:
            data_dir = options.dir
        if options.max_articles != None:
            max_articles = int(options.max_articles)
        if options.name != None:
            name = options.name

        setup_corpus(WebhoseArticleParser, data_dir, name, max_articles, per_date, big_data)
    else:
        setup_dev_corpus()
