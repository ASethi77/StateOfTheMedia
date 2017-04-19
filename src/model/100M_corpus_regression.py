import model.feature_util
from preprocess_text.setup_corpus import setup_100M_corpus
from model.decision_tree_model import DecisionTreeRegressionModel

if __name__ == '__main__':
	bigger_corpus = setup_100M_corpus(corpus_name="BiggerCorpus")

	daily_bag_of_words = model.feature_util.webhose_corpus_to_daily_bag_of_words(bigger_corpus)
	bag_of_words_per_day, presidential_approval_ratings = model.feature_util.daily_bag_of_words_to_regression_data(daily_bag_of_words)

	bag_of_words_train, bag_of_words_test, approval_ratings_train, approval_ratings_test = \
			model.feature_util.train_test_split(bag_of_words_per_day, presidential_approval_ratings)

	dev_corpus_regression_model = DecisionTreeRegressionModel([bag_of_words_train, approval_ratings_train])
	dev_corpus_regression_model.train()

	bag_of_words_sanity = bag_of_words_train[0]
	approval_rating_sanity = approval_ratings_train[0]
	approval_rating_prediction = dev_corpus_regression_model.predict([bag_of_words_sanity])[0]

	print("Sanity checking regression on trained example")
	print("Predicted approval ratings:\n\tApprove: {0}%\n\tDisapprove: {1}%".format(approval_rating_prediction[0], approval_rating_prediction[1]))
	print("Actual approval ratings:\n\tApprove: {0}%\n\tDisapprove: {1}%".format(approval_rating_sanity[0], approval_rating_sanity[1]))
