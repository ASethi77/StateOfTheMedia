import numpy as np
from sklearn.model_selection import cross_val_score
from nltk.stem.lancaster import LancasterStemmer
import textacy

import model
import model.feature_util
import model.sentiment_analysis
import model.topic_extractor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import timedelta
from model.linear_regression_model import LinearRegressionModel
from preprocess_text.load_corpora import load_corpora
from preprocess_text.setup_corpus import setup_corpus
from preprocess_text.article_parsers.webhose_article_parser import WebhoseArticleParser
from util.topic_matchers import topic_labels, label_index

def doc_to_text(doc, max_sentences=-1):
    sentences = ""
    num_sentences = 1
    for sent in doc.sents:
        if max_sentences > 0 and num_sentences > max_sentences:
            break

        sentences += str(sent).strip()
        num_sentences += 1

    return sentences

# run topic extraction/sentiment analysis on the corpora
# to build feature vectors per day
# we expect corpora to be a map of {datetime: corpus}
def corpora_to_day_features(corpora, sentiment_corpus):
    output = {}
    for date, corpus_for_day in corpora.items():
        day_feature_vector = [0.0] * (len(label_index.keys()) + 1) # features are topic labels plus sentiment value
        for doc in corpus_for_day:
            doc_topic = model.topic_extractor.topic_vectorize(doc_to_text(doc))
            doc_sentiment = model.sentiment_analysis.get_doc_sentiment_by_words(doc, sentiment_corpus)
            for indx in range(len(doc_topic)):
                day_feature_vector[indx] += doc_topic[indx]
            day_feature_vector[-1] += doc_sentiment
        for i in range(len(day_feature_vector)):
            day_feature_vector[i] = day_feature_vector[i] / len(corpus_for_day) # normalize our features
        output[date] = day_feature_vector
    return output

if __name__ == '__main__':
    print("Loading daily approval ratings...")
    obama_approval_ratings = model.feature_util.get_approval_poll_data()
    print("done.")

    print("Loading sentiment corpus...")
    sentiment_corpus = model.sentiment_analysis.load_sentiment_corpus()
    print("done.")

    print("Loading corpus of political articles...")
    political_article_corpora = setup_corpus(WebhoseArticleParser, "/opt/nlp_shared/data/news_articles/webhose_political_news_dataset", "WebhosePoliticalNewsArticles", 10000, per_date=True, use_big_data=True)
    print("done.")

    # TODO: Get inputs, which should look like:
    X = []
    Y = []
<<<<<<< HEAD
    for date, corpus_for_day in political_article_corpora: 
        if date not in obama_approval_ratings :
            print("Unable to find approval rating data for {}, skipping".format(date))
            continue

        approval_ratings_for_date = obama_approval_ratings[date]
        doc_input_vectors = []
        for doc in corpus_for_day:
            doc_text = doc_to_text(doc, max_sentences=5)
            doc_topic = model.topic_extractor.topic_vectorize(doc_text)
            doc_sentiment = model.sentiment_analysis.get_doc_sentiment_by_words(textacy.Doc(doc_text, lang='en'), sentiment_corpus)
            doc_topic.append(doc_sentiment)
            doc_input_features = doc_topic
            doc_input_vectors.append(doc_input_features)
            X.append(doc_input_features)
            Y.append(obama_approval_ratings[date])
            print()
            print("Doc text:")
            print(doc_text)
            print("Sentiment score:")
            print(doc_sentiment)

    print(len(X))
    test_partition = -20

    X_train = X[:test_partition]
    Y_train = Y[:test_partition]

    X_test = X[test_partition:]
    Y_test = Y[test_partition:]

    dev_corpus_regression_model = LinearRegressionModel([X_train, Y_train])
    dev_corpus_regression_model.train()

    input_sanity = X_train[0]
    label_sanity = Y_train[0]
    approval_rating_prediction = dev_corpus_regression_model.predict([input_sanity])[0]
    print(approval_rating_prediction)

    print("Sanity checking regression on trained example")
    print("Predicted approval ratings:\n\tApprove: {0}%\n\tDisapprove: {1}%".format(approval_rating_prediction[0],
                                                                                    approval_rating_prediction[1]))
    print("Actual approval ratings:\n\tApprove: {0}%\n\tDisapprove: {1}%".format(label_sanity[0],
                                                                                 label_sanity[1]))

    k_fold_scores = cross_val_score(dev_corpus_regression_model.model, X, Y, n_jobs=-1, cv=4)
    print(k_fold_scores)

