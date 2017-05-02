import pickle
import os
from .corpus import Corpus

def load_corpora(corpus_name, corpus_dir="../data/"):
    corpus_path = os.path.join(corpus_dir, corpus_name)
    with open(os.path.join(corpus_path, "dates.json"), "rb") as dates_file:
        dates = pickle.load(dates_file)

        corpora = {}
        for date in sorted(dates):
            #print(date)
            corpus = Corpus.load(corpus_path, str(date))
            #print(corpus)
            corpora[date] = corpus

        return corpora

if __name__ == '__main__':
    load_corpora("WebHoseDevCorpus")
