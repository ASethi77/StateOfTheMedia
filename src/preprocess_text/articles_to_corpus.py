import textacy
articles = []
for filename in textacy.fileio.utils.get_filenames('data/Articles/'):
    article = textacy.fileio.read_file(filename, mode=u'rt', encoding=None)
    articles.append(article)

article_corpus = textacy.Corpus('en', texts=articles, metadatas=None)
article_corpus.save(path='data/TextacyCorpra/', name='temp_corpa')