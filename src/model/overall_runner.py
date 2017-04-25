import json

import numpy as np

import model.feature_util
from preprocess_text.load_corpora import load_corpora
from model.linear_regression_model import LinearRegressionModel

from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
    # use mock of input
    input = []
    labels = []
    with open("../data/regression_mock_input.json", "rb") as f:
        input = json.load(f)
    # print(input)

    with open("../data/regression_mock_labels.json", "rb") as f:
        labels = json.load(f)
    # print(labels)

    X = input
    y = labels

    test_partition = -10

    X_train = X[:test_partition]
    y_train = y[:test_partition]

    X_test = X[test_partition:]
    y_test = y[test_partition:]

    # print(np.array(X_train).shape)

    # print(X_train)
    # print(y_train)
    dev_corpus_regression_model = LinearRegressionModel([X_train, y_train])
    dev_corpus_regression_model.train()

    input_sanity = X_train[0]
    label_sanity = y_train[0]
    approval_rating_prediction = dev_corpus_regression_model.predict([input_sanity])[0]
    print(approval_rating_prediction)

    print("Sanity checking regression on trained example")
    print("Predicted approval ratings:\n\tApprove: {0}%\n\tDisapprove: {1}%".format(approval_rating_prediction[0],
                                                                                    approval_rating_prediction[1]))
    print("Actual approval ratings:\n\tApprove: {0}%\n\tDisapprove: {1}%".format(label_sanity[0],
                                                                                 label_sanity[1]))

    print("Sanity check on training example:")
    print(X_test)
    print(y_test)
    print(dev_corpus_regression_model.predict(X_test))
    print(dev_corpus_regression_model.evaluate(X_test, y_test))
