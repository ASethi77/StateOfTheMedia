# this class is used to turn an article into a feature vector by topic

from util.topic_matchers import topic_labels, label_index
import textacy

def topic_vectorize(text):
    count_result = __count_signal_words(__pre_process_text(text))
    output_vector = count_result[0]
    num_signals = count_result[1]
    __normalize(output_vector, num_signals)
    return output_vector


def one_hot_topic_vectorize(text):
    count_result = __count_signal_words(__pre_process_text(text))
    output_vector = count_result[0]
    num_signals = count_result[1]
    max_indx = __max_indx(output_vector)
    output_vector[max_indx] = 1.0 # designate the main topic label
    for indx in range(len(output_vector)):
        if indx != max_indx:
             output_vector[indx] = 0.0 # zero out the secondary topic labels
    return output_vector

# returns a list of all words in the text
def __pre_process_text(text):
    text = textacy.preprocess.remove_punct(text)
    text = textacy.preprocess.normalize_whitespace(text)
    return text.split()

# returns a tuple of the non-normalized vector and the total count of signal words
def __count_signal_words(words):
    output = [0.0] * len(label_index.keys())
    signal_word_count = 0.0
    for word in words:
        if word.lower() in topic_labels.keys():
            topic_label = topic_labels[word.lower()]
            output[label_index[topic_label]] += 1.0
            signal_word_count += 1.0
    return (output, signal_word_count)

# returns the indx of the maximum value in the list    
def __max_indx(vector):
    max_val = vector[0]
    max_indx = 0
    for indx in range(len(vector)):
        if vector[indx] > max_val:
            max_indx = indx
            max_val = vector[indx]
    return max_indx

# returns the normalized vector based on the given total # of values
def __normalize(vector, total):
    for indx in range(len(vector)):
        vector[indx] = vector[indx] / total

if __name__ == "__main__": # used for basic testing
    test_phrase = "Barack Obama has pursued the most aggressive 'war on leaks' since the Nixon administration, according to a report published on Thursday that says the administration's attempts to control the flow of information is hampering the ability of journalists to do their jobs. The author of the study, the former Washington Post executive editor Leonard Downie, says the administration's actions have severely hindered the release of information that could be used to hold it to account."
    print(topic_vectorize(test_phrase))
    print(one_hot_topic_vectorize(test_phrase))
