def webhose_corpus_to_daily_bag_of_words(webhose_corpus):
	from collections import Counter, defaultdict
	import copy

	set_of_all_words = set()

	for day, corpus_for_day in webhose_corpus:
	    word_freqs_for_day = corpus_for_day.word_freqs(normalize='lower', as_strings=True)
	    set_of_all_words = set_of_all_words.union(set(word_freqs_for_day.keys()))
   
	count_of_all_words = Counter(set_of_all_words)

	for word in count_of_all_words:
	    count_of_all_words[word] = 0

	freqs_per_day = defaultdict(lambda: copy.deepcopy(count_of_all_words))
  
	for day, corpus_for_day in webhose_corpus:
	    word_freqs_for_day = corpus_for_day.word_freqs(normalize='lower', as_strings=True)
	    day_freq_counter = Counter(word_freqs_for_day)
	    freqs_per_day[day].update(day_freq_counter)

	return freqs_per_day

def daily_bag_of_words_to_regression_data(daily_bag_of_words):
	from evaluation.load_labels import LabelLoader

	label_loader = LabelLoader()
	label_loader.load_json(president_surname="Obama")
	labels = label_loader.get_labels()

	training_input_output_pairs = []
	for day in daily_bag_of_words:
	    approve, disapprove, neutral, _ = labels[day]
	    input_features = [daily_bag_of_words[day][word] for word in sorted(daily_bag_of_words[day])]
	    output_approval_ratings = [approve, disapprove]
	    training_input_output_pairs.append([input_features, output_approval_ratings])

	x, y = zip(*training_input_output_pairs)

	return x, y

def train_test_split(x, y, percent_test=0.25, random_seed=42):
	from sklearn.model_selection import train_test_split
	return train_test_split(x, y, test_size=percent_test, random_state=random_seed)
