import numpy as np
from sklearn.model_selection import cross_val_score

import model
import model.feature_util
import model.sentiment_analysis
import model.topic_extractor
import matplotlib.pyplot as plt
from datetime import timedelta
from model.linear_regression_model import LinearRegressionModel
from preprocess_text.load_corpora import load_corpora
from preprocess_text.setup_corpus import setup_corpus
from preprocess_text.article_parsers.webhose_article_parser import WebhoseArticleParser
from util.topic_matchers import topic_labels, label_index

def doc_to_text(doc):
    sentences = ""
    for sent in doc.sents:
        sentences += str(sent).strip()

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
            for i in range(len(doc_topic):
                day_feature_vector[i] += doc_topic[i]
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
    #political_article_corpora = setup_corpus(WebhoseArticleParser, "/opt/nlp_shared/data/news_articles/webhose_political_news_dataset", "WebhosePoliticalNewsArticles", 100000, per_date=True, use_big_data=True)
    political_article_corpora = load_corpora("WebhosePoliticalNewsArticles", True)
    print("done.")

    # TODO: Get inputs, which should look like:
    X = []
    Y = []
    poll_lag = 1  # modify how displaced the label should be from the features
    moving_range_size = 15 # how big of a range should we combine sentiments

    features_by_day = corpora_to_day_features(political_article_corpora, sentiment_corpus)
    # combine individual days' features into one feature vector a range of days
    for date, features in features_by_day.items():
        range_features = [0.0] * (len(label_index.keys()) + 1)
        days_with_data = 0 # count how many days in this range actually provided us data
        # TODO: this might be biased since days with different # of articles are weighted the same
        for i in range(0, moving_range_size):
            days_away = timedelta(days=i)
            target_day = date - days_away
            curr_day_features = features_by_day.get(target_day)
            if curr_day_features is not None:
                days_with_data += 1
                for i in range(len(curr_day_features)):
                    range_features[i] += curr_day_features[i]
        for i in range(len(curr_day_features)):
            range_features[i] = range_features[i] / days_with_data
        
        # match up inputs (range features) w/ output label
        approval_label = obama_approval_ratings.get(date + timedelta(days=poll_lag)) # approval label should be 'poll_lag' days into the future
        if approval_label is not None:
            X.append(range_features)
            Y.append(approval_label)

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

    '''print("Sanity check on training example:")
    print(X_test)
    print(Y_test)
    print(dev_corpus_regression_model.predict(X_test))
    print(dev_corpus_regression_model.evaluate(X_test, Y_test))'''

    k_fold_scores = cross_val_score(dev_corpus_regression_model.model, X, Y, n_jobs=-1, cv=4)
    print(k_fold_scores)
