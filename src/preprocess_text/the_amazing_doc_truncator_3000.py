import textacy
import itertools

from preprocess_text.load_corpora import load_corpora


def truncate_docs_in_daily_corpora(corpora):
    trunc_corpora = []
    for corpus in corpora:
        trunc_corpus = textacy.Corpus('en')
        for doc in corpus:
            first_sents_spans = list(itertools.islice(doc.sents, 2))
            # print(first_sents_spans)
            first_sents = ""
            for span in first_sents_spans:
                first_sents += str(span) + " "
            # print(first_sents)
            trunc_doc = textacy.Doc(first_sents, doc.metadata, 'en')
            # print(trunc_doc)
            trunc_corpus.add_doc(trunc_doc, doc.metadata)
        trunc_corpora.append(trunc_corpus)

    return trunc_corpora

if __name__ == '__main__':
    corp = load_corpora("WebHoseDevCorpus")
    print(corp)
    trunc_corp = truncate_docs_in_daily_corpora(corp)
    print(trunc_corp)