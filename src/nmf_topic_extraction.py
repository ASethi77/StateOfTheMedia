# coding: utf-8
import glob
from time import time
from preprocess_text import corpus
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import re
import pickle

NYT_CORPUS_PATH = '/opt/nlp_shared/corpora/NytCorpora/NYTCorpus/'

# Compute topics for Bill Clinton's terms
YEARS = range(1993, 2002)

N_FEATURES = 1000
N_TOPICS = 100
N_TOP_WORDS = 100

corpora_all = []

class LemmaTokenizer(object):
        def __init__(self):
            self.wnl = WordNetLemmatizer()
        def __call__(self, doc):
            output = []
            for t in word_tokenize(doc):
                if re.match('[^a-zA-Z]+', t):
                    output.append(self.wnl.lemmatize(t))

def get_top_words(model, feature_names, n=0):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        words_for_topic_filtered = []
        words_for_topic = [feature_names[i] for i in topic.argsort()[:-n - 1:-1]]
        for word in words_for_topic:
            if re.match('[^a-zA-Z]+', word) is None:
                words_for_topic_filtered.append(word)
    topics.append(words_for_topic_filtered)
    return topics 

# returns an nmf_model given a corpora
def build_nmf_model(corpora, num_features=N_FEATURES, num_topics=N_TOPICS):
    dataset = []
    for date, corpus_for_day in corpora.items():
        for article in corpus_for_day:
            dataset.append('\n'.join(article.sents))

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=num_features,
                                       tokenizer=LemmaTokenizer(),
                                       stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(dataset)
    nmf = NMF(n_components=num_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    return (nmf, tfidf_feature_names)

def write_model_to_disk(nmf_file_name, features_file_name, num_features=N_FEATURES, num_topics=N_TOPICS):
    nmf, feature_names = build_nmf_model(corpora, num_features, num_topics)
    pickle.dump(feature_names, features_file_name)
    pickle.dump(nmf, nmf_file_name)

def load_model_from_disk(nmf_file_name, features_file_name):
    return (pickle.load(nmf_file_name), pickle.load(features_file_name))
    
if __name__ == '__main__':
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
 
    # Use tf-idf features for NMF.
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=N_FEATURES,
                                       tokenizer=LemmaTokenizer(),
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
    get_top_words(nmf, tfidf_feature_names, N_TOP_WORDS)
