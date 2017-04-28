import pickle
import textacy

def load_corpora(corpus_name, use_big_data=False):
    corpus_dir = "../data/" + corpus_name
    with open(corpus_dir + "/dates.json", "rb") as dates_file:
        dates = pickle.load(dates_file)

        corpora = {}
        for date in sorted(dates):
            #print(date)
            corpus = textacy.Corpus.load(corpus_dir, str(date), big_ass_data=use_big_data)
            #print(corpus)
            corpora[date] = corpus

        return corpora

if __name__ == '__main__':
    load_corpora("WebHoseDevCorpus")
