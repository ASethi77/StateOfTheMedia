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
from preprocess_text.document import Document
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
        print("processing day with {0} articles".format(len(corpus_for_day)))
        day_feature_vector = [0.0] * (len(label_index.keys()) + 1) # features are topic labels plus sentiment value
        doc_num = 0
        for doc in corpus_for_day:
            print("\tprocessing doc {0}".format(doc_num))
            doc_num += 1
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
    num_articles = 100000
    corpus_name = "WebhosePoliticalArticles-{}-Docs".format(num_articles)
    political_article_corpora = load_corpora("WebhosePoliticalArticles-100000-Docs", "/opt/nlp_shared/corpora/WebhosePoliticalNewsCorpora/")
    print("done.")

    X = []
    Y = []
    poll_lag = 1  # modify how displaced the label should be from the features
    moving_range_size = 15 # how big of a range should we combine sentiments

    features_by_day = corpora_to_day_features(political_article_corpora, sentiment_corpus)
    # combine individual days' features into one feature vector a range of days
    print("Number of days of data: " + str(len(features_by_day.items())))
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
        for i in range(len(range_features)):
            range_features[i] = range_features[i] / days_with_data
        # TEST ADDING PREVIOUS DAY'S APPROVAL AS FEATURES
        '''prev_label = obama_approval_ratings.get(date - timedelta(days=1)) # get yesterday's labels
        if prev_label is not None:
            range_features.append(prev_label[0])
            range_features.append(prev_label[1])
        else:
            # no information so just provide a 50/50 guess...
            # TODO: Provide a better justification for default dates with no label
            range_features.append(50.0)
            range_features.append(50.0) '''
        # match up inputs (range features) w/ output label
        approval_label = obama_approval_ratings.get(date + timedelta(days=poll_lag)) # approval label should be 'poll_lag' days into the future
        if approval_label is not None:
            X.append(range_features)
            Y.append(approval_label)

    print("Number of feature vectors (ideally this is # days - moving_range_size + 1): " + str(len(X)))
    test_partition = -1 * int(0.35 * len(X)) # Use a percentage of data as validation set

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

    # ------------------------ Plotting Results ----------------------------------------
    actual_approval = []
    actual_disapproval = []
    predict_approval = []
    predict_disapproval = []
    axis_vals = []
    
    for label in Y_test:
        actual_approval.append(label[0])
        actual_disapproval.append(label[1])

    for i in range(len(X_test)):
        print("Predicting day " + str(i) + " given: " + str(X_test[i]))
        prediction = dev_corpus_regression_model.predict(X_test[i])
        print("Output: " + str(prediction))
        predict_approval.append(prediction[0][0])
        predict_disapproval.append(prediction[0][1])
        axis_vals.append(i)
        

    plt.figure(1)
    print("RED VALUES ARE ACTUAL - BLUE VALUES ARE PREDICTED") # just a nice console reminder
    # red is actual, blue is predicted
    plt.subplot(211)
    approval_actual, = plt.plot(axis_vals, actual_approval, 'ro')
    approval_predicted, = plt.plot(axis_vals, predict_approval, 'bo')
    plt.legend([approval_actual, approval_predicted], ["Actual", "Predicted"], loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.ylabel('Approval percentage')

    plt.subplot(212)
    disapproval_actual, = plt.plot(axis_vals, actual_disapproval, 'ro')
    disapproval_predicted, = plt.plot(axis_vals, predict_disapproval, 'bo')
    plt.legend([disapproval_actual, approval_predicted], ["Actual", "Predicted"], loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.ylabel('Disapproval percentage')
    plt.show()
