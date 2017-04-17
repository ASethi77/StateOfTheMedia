from preprocess_text.setup_corpus import setup_dev_corpus
from dateutil.parser import parse
from collections import defaultdict
from random import shuffle
import textacy

corpus = setup_dev_corpus(10)

# docs = corpus.docs[2:5]
# print(docs)
# textacy.Corpus('en', docs=docs)

date_to_docs = defaultdict(list)

for doc in corpus.docs:
    datestr = doc.metadata['published']
    datetime_object = parse(datestr)

    date_to_docs[datetime_object].append(doc)

print(date_to_docs)
corpra = []
for date, docs in date_to_docs.items():
    print(docs)
    corpus_for_today = textacy.Corpus('en', docs=docs)
    # print (corpus_for_today)
    # corpra.append(corpus_for_today)
    corpus_for_today.save("../../data/TextacyCorpra", str(date))

