# adding this to suppress sklearn DeprecationWarnings...
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import os
import time
import pickle
import datetime
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
from model.MLPRegressionModel import MLPRegressionModel
from preprocess_text.load_corpora import load_corpora
from preprocess_text.setup_corpus import setup_corpus
from util.config import Config, Paths, RegressionModels
from preprocess_text.document import Document
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from model.overall_runner import doc_to_text, corpus_to_day_features, corpora_to_day_features, \
                                 init_corpora, combine_day_ranges, match_features_to_labels

current_milli_time = lambda: int(round(time.time() * 1000))

if __name__ == '__main__':
    PREDICT_DELAY_RANGE = range(1, 15)
    DAY_RANGE_RANGE = range(1, 30)

    plot_x = []
    plot_y = []
    plot_z = []

    approval_ratings, political_article_corpora = init_corpora() 

    for delay in PREDICT_DELAY_RANGE:
        plot_x.append([])
        plot_y.append([])
        plot_z.append([])
        for day_range in DAY_RANGE_RANGE:
            plot_x[delay - 1].append(delay)
            plot_y[delay - 1].append(day_range)
            Config.POLL_DELAY = delay
            Config.DAY_RANGE = day_range
            
            features_by_day = corpora_to_day_features(political_article_corpora)
            features_by_range = combine_day_ranges(features_by_day)
            X, Y = match_features_to_labels(features_by_range, approval_ratings)

            X_train_and_val, X_test, Y_train_and_val, Y_test = \
                    train_test_split(X, Y, test_size=Config.TRAINING_PARTITION, random_state=2)

            X_train, X_val, Y_train, Y_val = \
                    train_test_split(X_train_and_val, Y_train_and_val, test_size=0.125, random_state=2)

            # setup model and configurations

            if Config.REGRESSION_MODEL == RegressionModels.LINEAR_REGRESSION:
                model = LinearRegressionModel([X_train, Y_train])
            elif Config.REGRESSION_MODEL == RegressionModels.MLP:
                model = MLPRegressionModel([X_train, Y_train])

            print(model)

            model.train()
            mse = model.evaluate(X_val, Y_val)
            print("MSE is {}".format(mse))
            plot_z[delay - 1].append(mse)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(plot_x, plot_y, plot_z, cmap=cm.coolwarm, antialiased=True, rstride=2, cstride=2)
    plt.show()
