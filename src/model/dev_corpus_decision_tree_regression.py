import model.feature_util
from preprocess_text.load_corpora import load_corpora
from model.decision_tree_model import DecisionTreeRegressionModel

import numpy
#import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
	dev_corpus_100_articles = load_corpora("WebHoseDevCorpus")

	daily_bag_of_words = model.feature_util.webhose_corpus_to_daily_bag_of_words(dev_corpus_100_articles)
	bag_of_words_per_day, presidential_approval_ratings = model.feature_util.daily_bag_of_words_to_regression_data(daily_bag_of_words)

	bag_of_words_train, bag_of_words_test, approval_ratings_train, approval_ratings_test = \
			model.feature_util.train_test_split(bag_of_words_per_day, presidential_approval_ratings)

	dev_corpus_regression_model = DecisionTreeRegressionModel([bag_of_words_train, approval_ratings_train], max_depth=2)
	dev_corpus_regression_model.train()

	bag_of_words_sanity = bag_of_words_train[0]
	approval_rating_sanity = approval_ratings_train[0]
	approval_rating_prediction = dev_corpus_regression_model.predict([bag_of_words_sanity])[0]

	print("Sanity checking regression on trained example")
	print("Predicted approval ratings:\n\tApprove: {0}%\n\tDisapprove: {1}%".format(approval_rating_prediction[0], approval_rating_prediction[1]))
	print("Actual approval ratings:\n\tApprove: {0}%\n\tDisapprove: {1}%".format(approval_rating_sanity[0], approval_rating_sanity[1]))

	print("Sanity check on training example:")
	print(dev_corpus_regression_model.evaluate(bag_of_words_test, approval_ratings_test))

	print("Computing k-fold loss with different max decision tree depths:")
	depth_min = 1
	depth_max = 25
	score_per_depth = {}
	for i in range(depth_min, depth_max + 1, 1):
		print(i)
		dev_corpus_regression_model = DecisionTreeRegressionModel([bag_of_words_train, approval_ratings_train], max_depth=i)
		k_fold_scores = cross_val_score(dev_corpus_regression_model.model, bag_of_words_per_day, presidential_approval_ratings, n_jobs=-1, cv=4)
		score_per_depth[i] = numpy.mean(k_fold_scores)
		print("mean score is {0}".format(score_per_depth[i]))
