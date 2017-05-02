'''
Module to store application-wide constants, such as paths for different datasets, etc.

Available consts:

WORD_SENTIMENT_CORPUS_PATH: Path to the tff file available through the 2005 MPQA Subjectivity Lexicon
(Theresa Wilson, Janyce Wiebe, and Paul Hoffmann (2005). Recognizing Contextual Polarity in Phrase-Level Sentiment Analysis. Proc. of HLT-EMNLP-2005.)
'''

import os
from enum import Enum, IntEnum

class Paths(Enum):
	WORD_SENTIMENT_CORPUS_PATH = os.environ.get("WORD_SENTIMENT_PATH", "/opt/nlp_shared/data/subjectivity_lexicon/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff")

class Config(IntEnum):
    POLL_DELAY = 1 # how far into the future we should predict
    DAY_RANGE = 15 # how many days of articles we should compound into one feature vector
    NUM_LAYERS = 3 # number of layers to use in neural network
