from numpy import tanh
from util.consts import Paths

def get_doc_sentiment_by_words(doc, sentiment_corpus):
	sum_sentiments = 0.0
	positive_count = 0.0
	negative_count = 0.0
	for token in doc:
		token = str(token)
		if token in sentiment_corpus:
			'''sum_sentiments += sentiment_corpus[token]''' # I theorize that the difference actually forces extremes. Use ratios instead
			if sentiment_corpus[token] > 0.0:
				positive_count += 1.0
			else:
				negative_count += 1.0
	if negative_count < 1.0:
		return 1.0 # avoid divide by zero errors. Only positive sentiment 
	return tanh(positive_count / negative_count)

def load_sentiment_corpus(tff_path=Paths.WORD_SENTIMENT_CORPUS_PATH.value):
	words = []
	with open(tff_path, "r") as tff:
		for word in tff:
			words.append(word.strip())
	sentiment_per_word = {}
	tff_entries = []
	for word in words:
		attributes = [[kvpair for kvpair in word.split("=")] for word in word.split(" ")]
		attributes_dict = {}
		for attribute in attributes:
			attributes_dict[attribute[0]] = attribute[1]
			tff_entries.append(attributes_dict)
	for word_attrs in tff_entries:
		word = word_attrs["word1"]
		sentiment = word_attrs["priorpolarity"]
		sentiment_value = 0.0
		if sentiment == "positive":
			sentiment_value = 1.0
		else:
			sentiment_value = -1.0
		sentiment_per_word[word] = sentiment_value
	return sentiment_per_word
