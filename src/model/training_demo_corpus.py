# coding: utf-8
from preprocess_text.setup_corpus import setup_dev_corpus

mycorpus = setup_dev_corpus()

from collections import Counter, defaultdict
import copy
from evaluation.load_labels import LabelLoader
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor

set_of_all_words = set()

for day, corpus_for_day in mycorpus:
    word_freqs_for_day = corpus_for_day.word_freqs(normalize='lower', as_strings=True)
    set_of_all_words = set_of_all_words.union(set(word_freqs_for_day.keys()))

count_of_all_words = Counter(set_of_all_words)

for word in count_of_all_words:
    count_of_all_words[word] = 0

freqs_per_day = defaultdict(lambda: copy.deepcopy(count_of_all_words))

for day, corpus_for_day in mycorpus:
    word_freqs_for_day = corpus_for_day.word_freqs(normalize='lower', as_strings=True)
    day_freq_counter = Counter(word_freqs_for_day)
    freqs_per_day[day].update(day_freq_counter)

label_loader = LabelLoader()
label_loader.load_json()
labels = label_loader.get_labels()

training_input_output_pairs = []
for day in freqs_per_day:
    # create a mapping between per-day bag-of-words and approval ratings
    approve, disapprove, neutral, _ = labels[day]
    input_features = [freqs_per_day[day][word] for word in sorted(freqs_per_day[day])]
    output_approval_ratings = [approve, disapprove]
    training_input_output_pairs.append([input_features, output_approval_ratings])

x, y = zip(*training_input_output_pairs)

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_1.fit(x, y)

x_input_sanity = x[0]
y_output_sanity = y[0]
x_test = [x_input_sanity]
y_test = regr_1.predict(x_test)

print("Sanity checking regression on trained example")
print("Predicted approval ratings:\n\tApprove: {0}%\n\tDisapprove: {1}%".format(y_test[0][0], y_test[0][1]))
print("Actual approval ratings:\n\tApprove: {0}%\n\tDisapprove: {1}%".format(y_output_sanity[0], y_output_sanity[1]))
