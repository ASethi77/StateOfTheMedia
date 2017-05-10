# adding this to suppress sklearn DeprecationWarnings...
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import os
import time
import pickle
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
from util.config import Config, Paths, RegressionModels
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
    topic_extraction_cache_filename = "_".join([str(date), Config.CORPUS_NAME.value, Config.TOPIC_EXTRACTION_METHOD.value.name])
    sentiment_analysis_cache_filename = "_".join([str(date), Config.CORPUS_NAME.value, Config.SENTIMENT_ANALYSIS_METHOD.value.name])

    topic_extraction_cache_filename = os.path.join(Config.FEATURE_CACHE_DIR.value, topic_extraction_cache_filename)
    sentiment_analysis_cache_filename = os.path.join(Config.FEATURE_CACHE_DIR.value, sentiment_analysis_cache_filename)

    topics_precomputed = os.path.exists(topic_extraction_cache_filename)
    sentiments_precomputed = os.path.exists(sentiment_analysis_cache_filename)

    print("processing day {0} with {1} articles".format(date, len(corpus_for_day)))
    day_feature_vector = [0.0] * (Config.NUM_TOPICS.value + 1) # features are topic labels plus sentiment value
    day_sentiments = 0
    day_topics = [0.0] * (Config.NUM_TOPICS.value)

    if topics_precomputed:
        day_topics = pickle.load(open(topic_extraction_cache_filename, "rb"))

    if sentiments_precomputed:
        day_sentiments = pickle.load(open(sentiment_analysis_cache_filename, "rb"))

    t0_day = current_milli_time() 
    if not topics_precomputed or not sentiments_precomputed:
        doc_num = 0
        for doc in corpus_for_day:
            t0_doc = current_milli_time()
            doc_num += 1
            
            if not topics_precomputed:
                doc_topic = Config.TOPIC_EXTRACTION_METHOD.value.value(doc_to_text(doc, max_sentences=3))
                for indx in range(len(doc_topic)):
                    day_topics[indx] += doc_topic[indx]

            if not sentiments_precomputed:
                doc_sentiment = Config.SENTIMENT_ANALYSIS_METHOD.value.value(doc)
                day_sentiments += doc_sentiment

            t1_doc = current_milli_time()
            print("\tprocessing doc {0} took {1} milliseconds".format(doc_num, t1_doc - t0_doc))

    if not topics_precomputed:
        for i in range(len(day_topics)):
            day_topics[i] = day_topics[i] / len(corpus_for_day) # normalize our features

    if not sentiments_precomputed:
        day_sentiments /= float(len(corpus_for_day))

    t1_day = current_milli_time()
    print("processing day {0} took {1} milliseconds".format(date, t1_day - t0_day))
    output[date] = day_topics + [ day_sentiments ]
   
    if not topics_precomputed: 
        pickle.dump(day_topics, open(topic_extraction_cache_filename, "wb"))

    if not sentiments_precomputed:
        pickle.dump(day_sentiments, open(sentiment_analysis_cache_filename, "wb"))


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
    article_corpora = load_corpora(corpus_name, "/opt/nlp_shared/corpora/{}/".format(Config.CORPUS_SUBDIR.value), Config.CORPUS_YEARS.value)
    print(len(article_corpora))
    print()
    print()
    print()
    print()
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
            #approval_label = approval_ratings.get("hello from the other side") # approval label should be 'poll_lag' days into the future
            actual_date = date + timedelta(days=Config.POLL_DELAY.value)
            approval_label = approval_ratings.get(actual_date.date()) # approval label should be 'poll_lag' days into the future
            for rating in approval_ratings.keys():
                print(str(rating))
            print(type(rating))
            print(type(actual_date.date()))
            print("approval label for day {} is {}".format(str(actual_date), approval_label))
            if approval_label is not None:
                X.append(features)
                Y.append(approval_label[:-1])  # remove count of number of polls contributing to daily rating
            else:
                print("UNABLE TO FIND APPROVAL RATINGS FOR DAY {}".format(str(actual_date.date())))
        return (X, Y)

# print the given message to console
# and write it to file
def pw(output_file, message):
    output_file.write(message + "\n")
    print(message)

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

    print(len(political_article_corpora.keys())) 
    features_by_day = corpora_to_day_features(political_article_corpora)
    print("Number of days of data: " + str(len(features_by_day.items())))
    features_by_range = combine_day_ranges(features_by_day)
    X, Y = match_features_to_labels(features_by_range, approval_ratings)

    print("Number of feature vectors (ideally this is # days - moving_range_size + 1): " + str(len(X))) 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=Config.TRAINING_PARTITION.value)

    # setup model and configurations
    model = None

    # train new model (only if no model is loaded)
    if options.model_type is None or options.model_type == RegressionModels.LINEAR_REGRESSION:
        if not options.evaluate:
            model = LinearRegressionModel([X, Y]) # when not evaluating, use entire data
        else:
            model = LinearRegressionModel([X_train, Y_train])
    elif options.model_type == RegressionModels.MLP:
        if not options.evaluate:
            model = MLPRegressionModel([X, Y]) # when not evaluating, use entire data
        else:
            model = MLPRegressionModel([X_train, Y_train])


    model_name = None

    if options.load_file is not None:
        model.load(options.load_file)
        model_name = options.load_file
    else:
        model.train()
    if options.save:
        model_name = "TEMP_MODEL_" + str(datetime.datetime.now())
        model.save(model_name)
    
    if options.evaluate:
        eval_file = open(Paths.EVAL_RESULTS_PATH + model_name + ".txt")
        pw(eval_file, "BEGIN MODEL ANALYSIS FOR: " + model_name + " of type " + options.model_type)
        pw(eval_file, "================================================")
        pw(eval_file, "")
        pw(eval_file, "")
        pw(eval_file, "")

        MSE_approval = 0.0
        MSE_disapproval = 0.0

        total_diff_approval = 0.0
        total_diff_disapproval = 0.0

        first_approval_group_count = 0
        second_approval_group_count = 0
        third_approval_group_count = 0
        fourth_approval_group_count = 0
        fifth_approval_group_count = 0

        first_disapproval_group_count = 0
        second_disapproval_group_count = 0
        third_disapproval_group_count = 0
        fourth_disapproval_group_count = 0
        fifth_disapproval_group_count = 0

        approval_over_count = 0
        approval_under_count = 0
        disapproval_over_count = 0
        disapproval_under_count = 0

        # keep track of outliers: tuples of the form (feature_vector, label, prediction)
        approval_outliers = []
        disapproval_outliers = []

        for i in range(len(X_train)):
            prediction = model.predict([X_train[i]])[0]
            diff_approval_signed = prediction[0] - Y_train[i][0]
            diff_disapproval_signed = prediction[1] - Y_train[i][1]
            diff_approval = math.fabs(diff_approval_signed)
            diff_disapproval = math.fabs(diff_disapproval_signed)

            percent_diff_approval_signed = diff_approval / Y_train[i][0]
            percent_diff_disapproval_signed = diff_disapproval / Y_train[i][1]
            percent_diff_approval = math.fabs(percent_diff_approval_signed)
            percent_diff_disapproval_signed = math.fabs(percent_diff_disapproval_signed)

            MSE_approval += math.pow(diff_approval, 2)
            MSE_disapproval += math.pow(diff_disapproval, 2)

            total_diff_approval += diff_approval
            total_diff_disapproval += diff_disapproval

            # count which 'percentiile' results fall in
            if percent_diff_approval < Config.FIRST_CUTOFF:
                first_approval_group_count += 1
            if percent_diff_approval < Config.SECOND_CUTOFF:
                second_approval_group_count += 1
            if percent_diff_approval < Config.THIRD_CUTOFF:
                third_approval_group_count += 1
            if percent_diff_approval < Config.FOURTH_CUTOFF:
                fourth_approval_group_count += 1
            if percent_diff_approval < Config.FIFTH_CUTOFF:
                fifth_approval_group_count += 1
 
            # count which 'percentiile' results fall in
            if percent_diff_disapproval < Config.FIRST_CUTOFF:
                first_disapproval_group_count += 1
            if percent_diff_disapproval < Config.SECOND_CUTOFF:
                second_disapproval_group_count += 1
            if percent_diff_disapproval < Config.THIRD_CUTOFF:
                third_disapproval_group_count += 1
            if percent_diff_disapproval < Config.FOURTH_CUTOFF:
                fourth_disapproval_group_count += 1
            if percent_diff_disapproval < Config.FIFTH_CUTOFF:
                fifth_disapproval_group_count += 1 

            # count over/understimates
            if diff_approval > Config.LENIENCY:
                if diff_approval_signed > 0:
                    approval_over_count += 1
                else:
                    approval_under_count += 1
            if diff_disapproval > Config.LENIENCY:
                if diff_disapproval_signed > 0:
                    disapproval_over_count += 1
                else:
                    disapproval_under_count += 1
                
          
            # handle outliers
            if diff_approval >= Config.OUTLIER_THRESHOLD_HARD:
                approval_outliers.append((X_train[i], Y_train[i], prediction[0]))
            if diff_disapproval >= Config.OUTLIER_THRESHOLD_HARD:
                disapproval_outliers.append((X_train[i], Y_train[i], prediction[1]))

            
            #TODO: Check trend matching (does the directionality/magnitude change correlate with the actual labels)
            # This might be difficult given random partitioning
        
        RMSE_approval = math.sqrt(MSE_approval / len(Y_train))
        RMSE_disapproval = math.sqrt(MSE_disapproval / len(Y_train))

        avg_diff_approval = total_diff_approval / len(Y_train)
        avg_diff_disapproval = total_diff_disapproval / len(Y_train)

        # print out results:
        pw(eval_file, "Evaluation results on {} points of Training Data\n".format(len(X_train))) 
        pw("==================================================")
        pw(eval_file, "Root Mean Squared Error (Approval): " + str(RMSE_approval))
        pw(eval_file, "Root Mean Squared Error (Disapproval): " + str(RMSE_disapproval))
        pw(eval_file, "")
        pw(eval_file, "Average distance (Approval): " + str(avg_diff_approval))
        pw(eval_file, "Average distance (Disapproval): " + str(avg_diff_disapproval))
        pw(eval_file, "")
        pw(eval_file, "# of approval data points within " + str(Config.FIRST_CUTOFF) + "% of actual: " + first_approval_group_count)               
        pw(eval_file, "# of approval data points within " + str(Config.SECOND_CUTOFF) + "% of actual: " + second_approval_group_count)
        pw(eval_file, "# of approval data points within " + str(Config.THIRD_CUTOFF) + "% of actual: " + third_approval_group_count)
        pw(eval_file, "# of approval data points within " + str(Config.FOURTH_CUTOFF) + "% of actual: " + fourth_approval_group_count)
        pw(eval_file, "# of approval data points within " + str(Config.FIFTH_CUTOFF) + "% of actual: " + fifth_approval_group_count)
        pw(eval_file, "")
        pw(eval_file, "# of disapproval data points within " + str(Config.FIRST_CUTOFF) + "% of actual: " + first_disapproval_group_count)               
        pw(eval_file, "# of disapproval data points within " + str(Config.SECOND_CUTOFF) + "% of actual: " + second_disapproval_group_count)
        pw(eval_file, "# of disapproval data points within " + str(Config.THIRD_CUTOFF) + "% of actual: " + third_disapproval_group_count)
        pw(eval_file, "# of disapproval data points within " + str(Config.FOURTH_CUTOFF) + "% of actual: " + fourth_disapproval_group_count)
        pw(eval_file, "# of disapproval data points within " + str(Config.FIFTH_CUTOFF) + "% of actual: " + fifth_disapproval_group_count)
        pw(eval_file, "")
        pw(eval_file, "# of approval over-estimates: " + str(approval_over_count))
        pw(eval_file, "# of approval under-estimates: " + str(approval_under_count))
        pw(eval_file, "Ratio of over to under (Approval): " + str(approval_over_count * 1.0 / approval_under_count))
        pw(eval_file, "# of disapproval over-estimates: " + str(disapproval_over_count))
        pw(eval_file, "# of disapproval under-estimates: " + str(disapproval_under_count))
        pw(eval_file, "Ratio of over to under (Disapproval): " + str(disapproval_over_count * 1.0 / disapproval_under_count))
        pw(eval_file, "")
        pw(eval_file, "# of Outliers (Approval): " + len(approval_outliers))
        pw(eval_file, "---------------------------------------------------")
        for i in range(len(approval_outliers)):
            features, actual, predicted  = approval_outliers[i]
            pw(eval_file, "Outlier " + str(i) + ": " + str(features) + " => " + str(predicted) + "(when actual is " + str(actual) + ")")
        pw(eval_file, "")
        pw(eval_file, "# of Outliers (Disapproval): " + len(approval_outliers))
        pw(eval_file, "---------------------------------------------------")
        for i in range(len(disapproval_outliers)):
            features, actual, predicted  = disapproval_outliers[i]
            pw(eval_file, "Outlier " + str(i) + ": " + str(features) + " => " + str(predicted) + "(when actual is " + str(actual) + ")")


        MSE_approval = 0.0
        MSE_disapproval = 0.0

        total_diff_approval = 0.0
        total_diff_disapproval = 0.0

        first_approval_group_count = 0
        second_approval_group_count = 0
        third_approval_group_count = 0
        fourth_approval_group_count = 0
        fifth_approval_group_count = 0

        first_disapproval_group_count = 0
        second_disapproval_group_count = 0
        third_disapproval_group_count = 0
        fourth_disapproval_group_count = 0
        fifth_disapproval_group_count = 0

        approval_over_count = 0
        approval_under_count = 0
        disapproval_over_count = 0
        disapproval_under_count = 0

        # keep track of outliers: tuples of the form (feature_vector, label, prediction)
        approval_outliers = []
        disapproval_outliers = []

        for i in range(len(X_test)):
            prediction = model.predict([X_test[i]])[0]
            diff_approval_signed = prediction[0] - Y_test[i][0]
            diff_disapproval_signed = prediction[1] - Y_test[i][1]
            diff_approval = math.fabs(diff_approval_signed)
            diff_disapproval = math.fabs(diff_disapproval_signed)

            percent_diff_approval_signed = diff_approval / Y_test[i][0]
            percent_diff_disapproval_signed = diff_disapproval / Y_test[i][1]
            percent_diff_approval = math.fabs(percent_diff_approval_signed)
            percent_diff_disapproval_signed = math.fabs(percent_diff_disapproval_signed)

            MSE_approval += math.pow(diff_approval, 2)
            MSE_disapproval += math.pow(diff_disapproval, 2)

            total_diff_approval += diff_approval
            total_diff_disapproval += diff_disapproval

            # count which 'percentiile' results fall in
            if percent_diff_approval < Config.FIRST_CUTOFF:
                first_approval_group_count += 1
            if percent_diff_approval < Config.SECOND_CUTOFF:
                second_approval_group_count += 1
            if percent_diff_approval < Config.THIRD_CUTOFF:
                third_approval_group_count += 1
            if percent_diff_approval < Config.FOURTH_CUTOFF:
                fourth_approval_group_count += 1
            if percent_diff_approval < Config.FIFTH_CUTOFF:
                fifth_approval_group_count += 1
 
            # count which 'percentiile' results fall in
            if percent_diff_disapproval < Config.FIRST_CUTOFF:
                first_disapproval_group_count += 1
            if percent_diff_disapproval < Config.SECOND_CUTOFF:
                second_disapproval_group_count += 1
            if percent_diff_disapproval < Config.THIRD_CUTOFF:
                third_disapproval_group_count += 1
            if percent_diff_disapproval < Config.FOURTH_CUTOFF:
                fourth_disapproval_group_count += 1
            if percent_diff_disapproval < Config.FIFTH_CUTOFF:
                fifth_disapproval_group_count += 1 

            # count over/understimates
            if diff_approval > Config.LENIENCY:
                if diff_approval_signed > 0:
                    approval_over_count += 1
                else:
                    approval_under_count += 1
            if diff_disapproval > Config.LENIENCY:
                if diff_disapproval_signed > 0:
                    disapproval_over_count += 1
                else:
                    disapproval_under_count += 1
                
          
            # handle outliers
            if diff_approval >= Config.OUTLIER_THRESHOLD_HARD:
                approval_outliers.append((X_test[i], Y_test[i], prediction[0]))
            if diff_disapproval >= Config.OUTLIER_THRESHOLD_HARD:
                disapproval_outliers.append((X_test[i], Y_test[i], prediction[1]))

            
            #TODO: Check trend matching (does the directionality/magnitude change correlate with the actual labels)
            # This might be difficult given random partitioning
        
        RMSE_approval = math.sqrt(MSE_approval / len(Y_test))
        RMSE_disapproval = math.sqrt(MSE_disapproval / len(Y_test))

        avg_diff_approval = total_diff_approval / len(Y_test)
        avg_diff_disapproval = total_diff_disapproval / len(Y_test)

        # print out results:
        pw(eval_file, "Evaluation results on {} points of Test Data\n".format(len(X_test))) 
        pw("==================================================")
        pw(eval_file, "Root Mean Squared Error (Approval): " + str(RMSE_approval))
        pw(eval_file, "Root Mean Squared Error (Disapproval): " + str(RMSE_disapproval))
        pw(eval_file, "")
        pw(eval_file, "Average distance (Approval): " + str(avg_diff_approval))
        pw(eval_file, "Average distance (Disapproval): " + str(avg_diff_disapproval))
        pw(eval_file, "")
        pw(eval_file, "# of approval data points within " + str(Config.FIRST_CUTOFF) + "% of actual: " + first_approval_group_count)               
        pw(eval_file, "# of approval data points within " + str(Config.SECOND_CUTOFF) + "% of actual: " + second_approval_group_count)
        pw(eval_file, "# of approval data points within " + str(Config.THIRD_CUTOFF) + "% of actual: " + third_approval_group_count)
        pw(eval_file, "# of approval data points within " + str(Config.FOURTH_CUTOFF) + "% of actual: " + fourth_approval_group_count)
        pw(eval_file, "# of approval data points within " + str(Config.FIFTH_CUTOFF) + "% of actual: " + fifth_approval_group_count)
        pw(eval_file, "")
        pw(eval_file, "# of disapproval data points within " + str(Config.FIRST_CUTOFF) + "% of actual: " + first_disapproval_group_count)               
        pw(eval_file, "# of disapproval data points within " + str(Config.SECOND_CUTOFF) + "% of actual: " + second_disapproval_group_count)
        pw(eval_file, "# of disapproval data points within " + str(Config.THIRD_CUTOFF) + "% of actual: " + third_disapproval_group_count)
        pw(eval_file, "# of disapproval data points within " + str(Config.FOURTH_CUTOFF) + "% of actual: " + fourth_disapproval_group_count)
        pw(eval_file, "# of disapproval data points within " + str(Config.FIFTH_CUTOFF) + "% of actual: " + fifth_disapproval_group_count)
        pw(eval_file, "")
        pw(eval_file, "# of approval over-estimates: " + str(approval_over_count))
        pw(eval_file, "# of approval under-estimates: " + str(approval_under_count))
        pw(eval_file, "Ratio of over to under (Approval): " + str(approval_over_count * 1.0 / approval_under_count))
        pw(eval_file, "# of disapproval over-estimates: " + str(disapproval_over_count))
        pw(eval_file, "# of disapproval under-estimates: " + str(disapproval_under_count))
        pw(eval_file, "Ratio of over to under (Disapproval): " + str(disapproval_over_count * 1.0 / disapproval_under_count))
        pw(eval_file, "")
        pw(eval_file, "# of Outliers (Approval): " + len(approval_outliers))
        pw(eval_file, "---------------------------------------------------")
        for i in range(len(approval_outliers)):
            features, actual, predicted  = approval_outliers[i]
            pw(eval_file, "Outlier " + str(i) + ": " + str(features) + " => " + str(predicted) + "(when actual is " + str(actual) + ")")
        pw(eval_file, "")
        pw(eval_file, "# of Outliers (Disapproval): " + len(approval_outliers))
        pw(eval_file, "---------------------------------------------------")
        for i in range(len(disapproval_outliers)):
            features, actual, predicted  = disapproval_outliers[i]
            pw(eval_file, "Outlier " + str(i) + ": " + str(features) + " => " + str(predicted) + "(when actual is " + str(actual) + ")")
        
        pw(eval_file, "")
        pw(eval_file, "")
        pw(eval_file, "========================================================")
        pw(eval_file, "K-fold cross validation scores: ")
        k_fold_scores = cross_val_score(model.model, X, Y, n_jobs=-1, cv=4, scoring="neg_mean_squared_error")
        pw(eval_file, k_fold_scores)
        
        eval_file.close()

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

        # plt.show()
        config_params = [
            "CORPUS_NAME",
            "POLL_DELAY",
            "DAY_RANGE",
            "SENTIMENT_ANALYSIS_METHOD",
            "TOPIC_EXTRACTION_METHOD",
            "NUM_TOPICS",
            "REGRESSION_MODEL",
            "NUM_LAYERS"
        ]
        plt.savefig(os.path.join(Config.PLOT_DIR.value, (Config.dump_config(config_params) + ".png")))
        pickle.dump(k_fold_scores, open(os.path.join(Config.PLOT_DIR.value, Config.dump_config(config_params) + "_k_fold_scores_negmse.txt"), "wb"))
