from numpy import tanh
#from util.config import Config

def get_doc_sentiment_by_words(doc, sentiment_corpus):
    sum_sentiments = 0.0
    pos_sentiments = 0.0
    neg_sentiments = 0.0
    for token in doc:
        token = str(token)
        #if Config.DEBUG_FUNCTION.value:
        #    print("Checking Token: " + token)
        if token in sentiment_corpus.keys():
            sentiment = sentiment_corpus[token]
            if sentiment > 0:
                pos_sentiments += 1.0
            else:
                neg_sentiments += 1.0	
            sum_sentiments += sentiment_corpus[token]
        #if Config.DEBUG_FUNCTION.value:
        #    print("P: " + str(pos_sentiments) + " N: " + str(neg_sentiments))
    total_sentiments = pos_sentiments + neg_sentiments
    sentiment_ratio = 0.0
    if total_sentiments > 0:
        sentiment_ratio = float((pos_sentiments if pos_sentiments > neg_sentiments else -1 * neg_sentiments) / total_sentiments)
    return sentiment_ratio

def load_mpqa_sentiment_corpus(tff_path):
	words = []
	with open(tff_path, "r") as tff:
		for word in tff:
			words.append(word.strip())
	sentiment_per_word = {}
	tff_entries = []
	for index, word in enumerate(words):
		attributes = [[kvpair for kvpair in word.split("=")] for word in word.split(" ")]
		attributes_dict = {}
		for attribute in attributes:
			try:
				attributes_dict[attribute[0]] = attribute[1]
			except IndexError as e:
				print(e)
				print(index)
				print(attribute)
				print(attribute[0])
				print(attribute[1])
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
