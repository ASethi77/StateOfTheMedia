from model.overall_runner import init_corpora, corpora_to_day_features, match_features_to_labels, multid_combine_day_ranges, split_data
from model.keras_lstm_regression_model import LSTMRegressionModel 
from util.config import Config
import math

if __name__ == "__main__":
    print("Initializing corpus...")
    approval_ratings, article_corpora = init_corpora()
    print("Got " + str(len(article_corpora.items())) + " days of articles.")
    print("Got " + str(len(approval_ratings.items())) + " days of approval ratings.")
    print("Done.")
    print("Building feature vectors...")
    features_by_day = corpora_to_day_features(article_corpora)
    print("Found " + str(len(features_by_day)) + " number of days with articles")
    range_features = multid_combine_day_ranges(features_by_day)
    X, Y = match_features_to_labels(range_features, approval_ratings)
    print("There are " + str(len(X)) + " ranges of features")
    print("There are " + str(len(Y)) + " labels")
    #partition data into train/test
    X_data, Y_data = split_data(X, Y)
    X_train, X_test, X_val = X_data
    Y_train, Y_test, Y_val = Y_data 
    print("Done.")
    print("Training set of size: " + str(len(X_train)))
    print("Test set of size: " + str(len(X_test)))
    print("Creating model...")
    model = LSTMRegressionModel(train_data=[X_train, Y_train], val_data=[X_test, Y_test])
    print("Done.")
    print("Training model...")
    model.train()
    print("Done.")
    print("Printing architecture...")
    model.plot_model("lstm.png")
    print("Done.")
    print("Evaluating...")
    predict_train = model.predict(X_train)
    print(predict_train)
    print(predict_train.shape)
    total_error = 0.0
    for i in range(len(predict_train)):
        predict_approval = predict_train[i][0][0] * 100
        diff_approval = predict_approval - Y_train[i][0]
        total_error += math.pow(math.fabs(diff_approval), 2)
    print("MSE (train): " + str(total_error / len(Y_train)))
    print("RMSE (train): " + str(math.sqrt(total_error / len(Y_train))))
    
    predict_test = model.predict(X_test)
    print(predict_test)
    print(predict_test.shape)
    total_error = 0.0
    for i in range(len(predict_test)):
        predict_approval = predict_test[i][0][0] * 100
        diff_approval = predict_approval - Y_test[i][0]
        total_error += math.pow(math.fabs(diff_approval), 2)
    print("MSE (test): " + str(total_error / len(Y_test)))
    print("RMSE (test): " + str(math.sqrt(total_error / len(Y_test))))
    #print(predict_test)
    print("Done.") 
