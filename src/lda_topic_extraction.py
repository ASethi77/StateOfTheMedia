# coding: utf-8
import glob
from time import time
from preprocess_text import corpus
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import re

NYT_CORPUS_PATH = '/opt/nlp_shared/corpora/NytCorpora/NYTCorpus/'

# Compute topics for Bill Clinton's terms
YEARS = range(1993, 2002)

N_FEATURES = 1000
N_TOPICS = 15
N_TOP_WORDS = 100

corpora_all = []

def print_top_words(model, feature_names, N_TOP_WORDS):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        words_for_topic = [feature_names[i] for i in topic.argsort()[:-N_TOP_WORDS - 1:-1]]
        words_for_topic_filtered = []
        for word in words_for_topic:
            if re.match('[^a-zA-Z]+', word) is None:
                words_for_topic_filtered.append(word)
        print(" ".join(words_for_topic_filtered))
        print()

for year in YEARS:
    corpora_files_for_year = glob.glob("{}{}*industrial*".format(NYT_CORPUS_PATH, year))
    corpora_for_year = []
    for corpus_path in corpora_files_for_year:
        corpus_date = corpus_path.split('/')[-1][:10]
        corpus_for_date = corpus.Corpus.load(NYT_CORPUS_PATH, corpus_date)
        corpora_for_year.append(corpus_for_date)
    print("Found articles for {} days in the year {}".format(len(corpora_for_year), year))

    corpora_all += corpora_for_year

dataset = []
for corpus_for_day in corpora_all:
    for article in corpus_for_day:
        dataset.append('\n'.join(article.sents))

N_SAMPLES=len(dataset)
        
average_num_sentences = 0.0
for article in dataset:
    average_num_sentences += article.count('\n')
average_num_sentences /= len(dataset)
print("average number of sentences in the {} articles is {}".format(len(dataset), average_num_sentences))
    
data_samples = []
data_samples = dataset[:N_SAMPLES]

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.90, min_df=2,
                                max_features=N_FEATURES,
                                stop_words='english',
                                tokenizer=LemmaTokenizer())
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

print("Fitting LDA models with tf features, "
      "N_SAMPLES=%d and N_FEATURES=%d..."
      % (N_SAMPLES, N_FEATURES))
lda = LatentDirichletAllocation(n_topics=N_TOPICS, max_iter=20,
                                learning_method='batch',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, N_TOP_WORDS)
