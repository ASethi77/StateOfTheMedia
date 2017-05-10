# coding: utf-8
import glob
from time import time
from preprocess_text import corpus
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

NYT_CORPUS_PATH = '/opt/nlp_shared/corpora/NytCorpora/NYTCorpus/'
YEAR = 2007

N_FEATURES = 1000
N_TOPICS = 25
N_TOP_WORDS = 100

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

corpora_files_for_year = glob.glob("{}{}*industrial*".format(NYT_CORPUS_PATH, YEAR))
corpora_for_year = []
for corpus_path in corpora_files_for_year:
    corpus_date = corpus_path.split('/')[-1][:10]
    corpus_for_date = corpus.Corpus.load(NYT_CORPUS_PATH, corpus_date)
    corpora_for_year.append(corpus_for_date)
print("Found articles for {} days in the year {}".format(len(corpora_for_year), YEAR))

dataset = []
for corpus_for_day in corpora_for_year:
    for article in corpus_for_day:
        dataset.append('\n'.join(article.sents))

N_SAMPLES=len(dataset)
        
average_num_sentences = 0.0
for article in dataset:
    average_num_sentences += article.count('\n')
average_num_sentences /= len(dataset)
print("average number of sentences in articles for {} is {}".format(YEAR, average_num_sentences))
    
data_samples = []
data_samples = dataset[:N_SAMPLES]
# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=N_FEATURES,
                                   stop_words='english')
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Fit the NMF model
print("Fitting the NMF model with tf-idf features, "
      "N_SAMPLES=%d and N_FEATURES=%d..."
      % (N_SAMPLES, N_FEATURES))
t0 = time()
nmf = NMF(n_components=N_TOPICS, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, N_TOP_WORDS)