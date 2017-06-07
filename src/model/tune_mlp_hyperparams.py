# adding this to suppress sklearn DeprecationWarnings...
import numpy


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import time
from sklearn.model_selection import train_test_split

from model.MLPRegressionModel import MLPRegressionModel
from util.config import Config
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from model.overall_runner import corpora_to_day_features, \
                                 init_corpora, combine_day_ranges, match_features_to_labels

current_milli_time = lambda: int(round(time.time() * 1000))

if __name__ == '__main__':
    hidden_layer_sizes = range(50, 200, 10)
    alpha_sizes = numpy.arange(0.00001, 0.001, 0.00010)
    plot_x = []
    plot_y = []
    plot_z = []

    approval_ratings, political_article_corpora = init_corpora()

    for index, hidden_layer_size in enumerate(hidden_layer_sizes):
        plot_x.append([])
        plot_y.append([])
        plot_z.append([])
        for alpha in alpha_sizes:
            print("Testing with hidden layer size {} and alpha {}".format(hidden_layer_size, alpha))
            plot_x[index].append(hidden_layer_size)
            plot_y[index].append(alpha)
            # print("Prediction delay is {}, day_range is {}".format(delay, day_range))
            features_by_day = corpora_to_day_features(political_article_corpora)
            #print("Number of days of data: " + str(len(features_by_day.items())))
            features_by_range = combine_day_ranges(features_by_day)
            X, Y = match_features_to_labels(features_by_range, approval_ratings)
            #print("Number of feature vectors (ideally this is # days - moving_range_size + 1): " + str(len(X)))

            X_train_and_val, X_test, Y_train_and_val, Y_test = \
                    train_test_split(X, Y, test_size=Config.TRAINING_PARTITION, random_state=2)

            X_train, X_val, Y_train, Y_val = \
                    train_test_split(X_train_and_val, Y_train_and_val, test_size=0.125, random_state=2)

            model = MLPRegressionModel([X_train, Y_train], hidden_layer_sizes=(hidden_layer_size,), alpha=alpha)
            model.train()
            mse = model.evaluate(X_val, Y_val)
            print("MSE is {}".format(mse))
            plot_z[index].append(mse)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_wireframe(plot_x, plot_y, plot_z, cmap=cm.coolwarm, antialiased=True)
    plt.savefig('mlp_hyperparam_tuning.png')
