from model.overall_runner import init_corpora, train_test_split, corpora_to_day_features, match_features_to_labels
from model.keras_lstm_regression_model import LSTMRegressionModel 
from util.config import Config
import math

if __name__ == "__main__":
    print("Initializing corpus...")
    approval_ratings, article_corpora = init_corpora()
    print("Done.")
    print("Building feature vectors...")
    features_by_day = corpora_to_day_features(article_corpora)
    print("Found " + str(len(features_by_day)) + " number of days with articles")
    X, Y = match_features_to_labels(features_by_day, approval_ratings) # note we do not use day ranges for LSTM
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=Config.TRAINING_PARTITION)
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
    total_error = 0.0
    for i in range(len(predict_train)):
        predict_approval = predict_train[i][0][0] * 100
        diff_approval = predict_approval - Y_train[i][0]
        total_error += math.pow(math.fabs(diff_approval), 2)
    print("MSE: " + str(total_error / len(Y_train)))
    print("RMSE: " + str(math.sqrt(total_error / len(Y_train))))
    #predict_test = model.predict(X_test)
    #print(predict_test)
    print("Done.") 
