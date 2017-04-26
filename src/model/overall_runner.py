import numpy as np
from sklearn.model_selection import cross_val_score

import model
import model.feature_util
import model.sentiment_analysis
import model.topic_extractor
from model.linear_regression_model import LinearRegressionModel
from preprocess_text.load_corpora import load_corpora
from preprocess_text.setup_corpus import setup_corpus
from preprocess_text.article_parsers.webhose_article_parser import WebhoseArticleParser

def doc_to_text(doc):
    sentences = ""
    for sent in doc.sents:
        sentences += str(sent).strip()

    return sentences

if __name__ == '__main__':
    print("Loading daily approval ratings...")
    obama_approval_ratings = model.feature_util.get_approval_poll_data()
    print("done.")

    print("Loading sentiment corpus...")
    sentiment_corpus = model.sentiment_analysis.load_sentiment_corpus()
    print("done.")

    print("Loading corpus of political articles...")
    political_article_corpora = setup_corpus(WebhoseArticleParser, "/opt/nlp_shared/data/news_articles/webhose_political_news_dataset", "WebhosePoliticalNewsArticles", 100000, per_date=True, use_big_data=True)
    print("done.")

    # TODO: Get inputs, which should look like:
    X = []
    Y = []
    for date, corpus_for_day in political_article_corpora: 
        if date not in obama_approval_ratings :
            print("Unable to find approval rating data for {}, skipping".format(date))
            continue

        approval_ratings_for_date = obama_approval_ratings[date]
        doc_input_vectors = []
        for doc in corpus_for_day:
            doc_topic = model.topic_extractor.topic_vectorize(doc_to_text(doc))
            doc_sentiment = model.sentiment_analysis.get_doc_sentiment_by_words(doc, sentiment_corpus)
            doc_topic.append(doc_sentiment)
            doc_input_features = doc_topic
            doc_input_vectors.append(doc_input_features)
            X.append(doc_input_features)
            Y.append(obama_approval_ratings[date])

        '''
        agg_features = []
        num_examples = len(doc_input_vectors)
        for feature_idx in range(len(doc_input_vectors[0])):
            feature_sum = 0.0
            for feature_vec in doc_input_vectors:
                feature_sum += float(feature_vec[feature_idx])
            feature_avg = feature_sum / num_examples
            agg_features.append(feature_avg) 
        '''

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
