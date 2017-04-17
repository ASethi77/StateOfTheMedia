from preprocess_text.setup_corpus import setup_dev_corpus
from dateutil.parser import parse
from collections import defaultdict
from random import shuffle
import textacy

corpus = textacy.Corpus.load(path="../../data/TextacyCorpra", name="WebHoseDevCorpus")

date_to_docs = defaultdict(list)

for doc in corpus.docs:
    datestr = doc.metadata['published']
    datetime_object = parse(datestr)

    date_to_docs[datetime_object.date()].append(doc)

# print(date_to_docs)
for date, docs in date_to_docs.items():
    # print(docs)
    # corpus_for_today = textacy.Corpus('en', docs=docs)
    # for doc in corpus_for_today:
    #     print(doc.text)
    # bag_o_words = corpus_for_today.word_freqs(as_strings=True)
    # print(bag_o_words)
    # print()
    bag_o_terms_today = dict()
    bot_list = []
    for doc in docs:
        bag_o_terms = doc.to_bag_of_terms(named_entities=False, as_strings=True)
        bot_list.append(bag_o_terms)

    for bag_o_terms in bot_list:
        for term, count in bag_o_terms.items():
            if term not in bag_o_terms_today:
                bag_o_terms_today[term] = count
            else:
                bag_o_terms_today[term] += count

    print(bag_o_terms_today)

