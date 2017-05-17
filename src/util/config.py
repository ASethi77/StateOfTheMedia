'''
Module to store application-wide constants, such as paths for different datasets, etc.

Available consts:

WORD_SENTIMENT_CORPUS_PATH: Path to the tff file available through the 2005 MPQA Subjectivity Lexicon
(Theresa Wilson, Janyce Wiebe, and Paul Hoffmann (2005). Recognizing Contextual Polarity in Phrase-Level Sentiment Analysis. Proc. of HLT-EMNLP-2005.)
'''

import os
from enum import Enum, IntEnum
from functools import partial

from model.sentiment_analysis import get_doc_sentiment_by_words, load_mpqa_sentiment_corpus
from model.topic_extractor import manual_topic_vectorize, manual_one_hot_topic_vectorize
from util.topic_matchers import hand_selected_topic_labels, hand_selected_label_index

class Paths(Enum):
    WORD_SENTIMENT_CORPUS_PATH = os.environ.get("WORD_SENTIMENT_PATH", "/opt/nlp_shared/data/subjectivity_lexicon/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff")
    EVAL_RESULTS_PATH = "./evaluation/results/"

class RegressionModels(Enum):
    LINEAR_REGRESSION = "linear"
    MLP = "mlp"

class SentimentAnalysisMethod(Enum):
    # Sentiment analysis via MPQA lexicon
    MPQA_LEXICON = load_mpqa_sentiment_corpus(Paths.WORD_SENTIMENT_CORPUS_PATH.value)
    MPQA = partial(lambda doc: get_doc_sentiment_by_words(doc, SentimentAnalysisMethod.MPQA_LEXICON.value))

    def __str__(self):
        return self.name

class TopicExtractionMethod(Enum):
    MANUAL_TOPIC_EXTRACTION_MIXTURE = partial(lambda text: manual_topic_vectorize(text))
    MANUAL_TOPIC_EXTRACTION_ONE_HOT = partial(lambda text: manual_one_hot_topic_vectorize(text))

    def __str__(self):
        return self.name

class Config():
    # General model testing params
    # ----------------------------------------------
    CORPUS_NAME = "NYTCorpus"
    CORPUS_SUBDIR = "NytCorpora"
    PLOT_DIR = "/opt/nlp_shared/plot/"
    CORPUS_YEARS = [1993, 1998]

    # Feature computation configuration params
    # ----------------------------------------------
    POLL_DELAY = 1 # how far into the future we should predict
    DAY_RANGE = 15 # how many days of articles we should compound into one feature vector
    SENTIMENT_ANALYSIS_METHOD = SentimentAnalysisMethod.MPQA
    TOPIC_EXTRACTION_METHOD = TopicExtractionMethod.MANUAL_TOPIC_EXTRACTION_MIXTURE
    TOPIC_NAMES = list(hand_selected_label_index.keys())
    NUM_TOPICS = len(hand_selected_label_index.keys())

    FEATURE_CACHE_DIR = os.environ.get("FEATURE_CACHE_DIR", "/opt/nlp_shared/analysis_cache/")

    # Regression model selection configuration params
    # -----------------------------------------------
    REGRESSION_MODEL = RegressionModels.MLP

    # Neural-net-specific configuration params
    # -----------------------------------------------
    NUM_LAYERS = 3 # number of layers to use in neural network

    # Model evaluation configuration params
    # -----------------------------------------------
    TRAINING_PARTITION = 0.20 # fraction of data to use for testing
    OUTLIER_THRESHOLD_HARD = 15 # percentage (as a decimal) of how much above or below the actual is considered an extreme outlier
    OUTLIER_THRESHOLD_PERCENT = 0.25 # percentage (as a decimal) relative to the actual is considered an extreme outlier
    FIRST_CUTOFF = 0.02
    SECOND_CUTOFF = 0.05
    THIRD_CUTOFF = 0.10
    FOURTH_CUTOFF = 0.20
    FIFTH_CUTOFF = 0.30
    LENIENCY = 2 # this is how much above or below the actual label do we allow before considering something an over/under estimate

    def dump_config(items=[]):
        if len(items) == 0:
            for config_item in Config:
                items.append(config_item.name)
        return '_'.join("{}={}".format(config_item.name, str(config_item.value)) for config_item in Config if config_item.name in items)
