# adding this to suppress sklearn DeprecationWarnings...
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import time
from datetime import timedelta
from optparse import OptionParser
from sklearn.model_selection import cross_val_score, train_test_split
import textacy
from multiprocessing.dummy import Pool as ThreadPool 

import model
import model.feature_util
import model.sentiment_analysis
import model.topic_extractor
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from model.linear_regression_model import LinearRegressionModel
from preprocess_text.load_corpora import load_corpora
from preprocess_text.setup_corpus import setup_corpus
from preprocess_text.article_parsers.webhose_article_parser import WebhoseArticleParser
from util.config import Config, Models, Paths, RegressionModels
from preprocess_text.document import Document

current_milli_time = lambda: int(round(time.time() * 1000))

def doc_to_text(doc, max_sentences=-1):
    sentences = ""
    num_sentences = 1
    for sent in doc.sents:
        if max_sentences > 0 and num_sentences > max_sentences:
            break

        sentences += str(sent).strip()
        num_sentences += 1

    return sentences

def corpus_to_day_features(date, corpus_for_day, output):
    print("processing day {0} with {1} articles".format(date, len(corpus_for_day)))
    day_feature_vector = [0.0] * (Config.NUM_TOPICS.value + 1) # features are topic labels plus sentiment value
    doc_num = 0
    t0_day = current_milli_time() 
    for doc in corpus_for_day:
        t0_doc = current_milli_time()
        doc_num += 1
        doc_topic = Config.TOPIC_EXTRACTION_METHOD.value.value(doc_to_text(doc, max_sentences=3))
        doc_sentiment = Config.SENTIMENT_ANALYSIS_METHOD.value.value(doc)
        for indx in range(len(doc_topic)):
            day_feature_vector[indx] += doc_topic[indx]
        t1_doc = current_milli_time()
        print("\tprocessing doc {0} took {1} milliseconds".format(doc_num, t1_doc - t0_doc))
        day_feature_vector[-1] += doc_sentiment
    for i in range(len(day_feature_vector)):
        day_feature_vector[i] = day_feature_vector[i] / len(corpus_for_day) # normalize our features
    t1_day = current_milli_time()
    print("\tprocessing day {0} took {1} milliseconds".format(date, t1_day - t0_day))
    output[date] = day_feature_vector

# run topic extraction/sentiment analysis on the corpora
# to build feature vectors per day
# we expect corpora to be a map of {datetime: corpus}
def corpora_to_day_features(corpora):
    output = {}
    threadpool = ThreadPool(4)
    arg_list = [(item[0], item[1], output) for item in corpora.items()]
    threadpool.starmap(corpus_to_day_features, arg_list)
    threadpool.close()
    threadpool.join()
    return output

def init_corpora():
    print("Loading daily approval ratings...")
    approval_ratings = model.feature_util.get_approval_poll_data()
    print("done.")

    print("Loading corpus of political articles...")
    num_articles = 100
    corpus_name = Config.CORPUS_NAME.value
    article_corpora = load_corpora(corpus_name, "/opt/nlp_shared/corpora/{}/".format(Config.CORPUS_SUBDIR.value))
    print("done.")
    
    return (approval_ratings, article_corpora)

# takes the features for individual days and does a running average for
# a shifting range of days (specified in config)
def combine_day_ranges(features_by_day):
    output = {}
    for date, features in features_by_day.items():
        range_features = [0.0] * (Config.NUM_TOPICS.value + 1)
        days_with_data = 0 # count how many days in this range actually provided us data
        # TODO: this might be biased since days with different # of articles are weighted the same
        for i in range(0, Config.DAY_RANGE.value):
            days_away = timedelta(days=i)
            target_day = date - days_away
            curr_day_features = features_by_day.get(target_day)
            if curr_day_features is not None:
                days_with_data += 1
                for i in range(len(curr_day_features)):
                    range_features[i] += curr_day_features[i]
        for i in range(len(range_features)):
            range_features[i] = range_features[i] / days_with_data

        '''
        prev_label = obama_approval_ratings.get(date - timedelta(days=1)) # get yesterday's labels
        if prev_label is not None:
            range_features.append(prev_label[0])
            range_features.append(prev_label[1])
        else:
            # no information so just provide a 50/50 guess...
            range_features.append(50.0)
            range_features.append(50.0)
        '''
        output[date] = range_features
    return output

def match_features_to_labels(features_by_range, approval_ratings):
        X = []
        Y = []
        # match up inputs (range features) w/ output label
        for date, features in features_by_range.items():
            approval_label = approval_ratings.get(date + timedelta(days=Config.POLL_DELAY.value)) # approval label should be 'poll_lag' days into the future
            if approval_label is not None:
                X.append(features)
                Y.append(approval_label[:-1])  # remove count of number of polls contributing to daily rating
        return (X, Y)


if __name__ == '__main__':
    # add command-line flags
    # NOTE: Set hyper-parameters in util/Config.py
    parser = OptionParser()
    parser.add_option("-s", "--save", dest="save", action="store_true", help="save the model to disk with a default name")
    parser.add_option("-l", "--load", dest="load_file", help="load the model from the given file", metavar="MODEL_FILE")
    parser.add_option("-p", "--plot", dest="plot_results", action="store_true", help="plot the eval results")
    parser.add_option("-d", "--dump_predictions", dest="dump_predictions", action="store_true", help="print feature vectors and prediction vectors for test set")
    parser.add_option("-e", "--evaluate", dest="evaluate", action="store_true", help="run k-fold cross validation on the data")
    parser.add_option("-m", "--model", dest="model_type", help="run with the given model type", metavar="MODEL_TYPE")
    (options, args) = parser.parse_args()
 
    # load various corpora and labels  
    approval_ratings, political_article_corpora = init_corpora() 
 
    features_by_day = corpora_to_day_features(political_article_corpora)
    print("Number of days of data: " + str(len(features_by_day.items())))
    features_by_range = combine_day_ranges(features_by_day)
    X, Y = match_features_to_labels(features_by_range, approval_ratings)

    print("Number of feature vectors (ideally this is # days - moving_range_size + 1): " + str(len(X))) 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=Config.TRAINING_PARTITION)

    # setup model and configurations
    model = None

    # train new model (only if no model is loaded)
    if options.model_type is None or options.model_type == RegressionModels.LINEAR_REGRESSION:
        if !options.evaluate:
            model = LinearRegressionModel([X, Y]) # when not evaluating, use entire data
        else:
            model = LinearRegressionModel([X_train, Y_train])
    elif options.model_type == RegressionModels.MLP:
        if !options.evaluate:
            model = MLPRegressionModel([X, Y]) # when not evaluating, use entire data
        else:
            model = MLPRegressionModel([X_train, Y_train])


    model_name = None

    if options.load is not None:
        model.load(options.load)
        model_name = options.load
    else:
        model.train()
    if options.save:
        model_name = "TEMP_MODEL_" + str(datetime.datetime.now())
        model.save(model_name)
    
    if options.evaluate:
        '''
        eval_file = open(Paths.EVAL_RESULTS_PATH + model_name + ".txt")
        
        for i in range(len(X_train)):
            prediction = model.predict([X_train[i]])[0]
        for i in range(len(X_test)):
            prediction = model.predict([X_test[i]])[0]
        '''
        input_sanity = X_train[0]
        label_sanity = Y_train[0]
        approval_rating_prediction = model.predict([input_sanity])[0]
        print(approval_rating_prediction)

        print("Sanity checking regression on trained example")
        print("Predicted approval ratings:\n\tApprove: {0}".format(approval_rating_prediction[0]))
        print("Actual approval ratings:\n\tApprove: {0}%".format(label_sanity[0]))

        k_fold_scores = cross_val_score(model.model, X, Y, n_jobs=-1, cv=4, scoring="neg_mean_squared_error")
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
        prediction = model.predict(X_test[i])
        if options.dump_predictions:
            print("Predicting day " + str(i) + " given: " + str(X_test[i]))
            print("Output: " + str(prediction))
        predict_approval.append(prediction[0][0])
        predict_disapproval.append(prediction[0][1])
        if options.plot_results:
            axis_vals.append(i)
            plt.figure(1)

    # red is actual, blue is predicted
    if options.plot_results:
        print("RED VALUES ARE ACTUAL - BLUE VALUES ARE PREDICTED") # just a nice console reminder
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
        
