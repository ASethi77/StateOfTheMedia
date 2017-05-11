import pickle
import os
import glob
import datetime
from dateutil.parser import parse
from .corpus import Corpus

def load_corpora(corpus_name, corpus_dir="../data/", years=[]):
    corpus_path = os.path.join(corpus_dir, corpus_name)
    if corpus_path[-1] != "/":
        corpus_path += "/"

    if len(years) == 0:
        with open(os.path.join(corpus_path, "dates.json"), "rb") as dates_file:
            dates = pickle.load(dates_file)

            corpora = {}
            for date in sorted(dates):
                corpus = Corpus.load(corpus_path, str(date))
                corpora[date] = corpus

            return corpora

    else:
        print("Loading corpora for years between {0} and {1}".format(years[0], years[-1]))
        corpora = {}
        for year in range(years[0], years[-1]):
            print(os.path.join(corpus_path, "{}*industrial*".format(year)))
            corpora_files_for_year = glob.glob(os.path.join(corpus_path, "{}*industrial*".format(year)))
            corpora_for_year = 0
            for corpora_file in corpora_files_for_year:
                corpus_date = corpora_file.split('/')[-1][:10]
                corpus_for_date = Corpus.load(os.path.dirname(corpora_file), corpus_date)
                corpora[parse(corpus_date)] = corpus_for_date
                corpora_for_year += 1
            print("Found articles for {} days in the year {}".format(corpora_for_year, year))

        return corpora

if __name__ == '__main__':
    load_corpora("WebHoseDevCorpus")
