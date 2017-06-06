from model.overall_runner import init_corpora, train_test_split, corpora_to_day_features, match_features_to_labels
from model.keras_lstm_regression_model import LSTMRegressionModel 
from util.config import Config

if __name__ == "__main__":
    print("Initializing corpus...")
    approval_ratings, article_corpora = init_corpora()
    print("Done.")
    print("Building feature vectors...")
    features_by_day = corpora_to_day_features(article_corpora)
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
