#!flask/bin/python

# This is the back-end REST api for making predictions about text
# make a GET request to the following endpoints to receive JSON data

# to populate the web-app.
#
# /model/predict
# /model/sentiment
# /model/topic
#
# RUN WITH: python3 ./server.py

# code to fix PYTHONPATH
import sys
import os
PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from flask import Flask, request
from flask import jsonify

import pickle
import model.overall_runner as Runner
import model.sentiment_analysis as Sentiment
import model.topic_extractor as Topic
from nltk.tokenize import word_tokenize
from model.linear_regression_model import LinearRegressionModel
from model.MLPRegressionModel import MLPRegressionModel
from util.config import Config, Paths, RegressionModels

app = Flask(__name__)

sentiment_corpus = None
topic_corpus = None
model = None

def init_server():
    '''topic_extraction_cache_filename = "_".join([str(date), Config.CORPUS_NAME.value, Config.TOPIC_EXTRACTION_METHOD.value.name])
    sentiment_analysis_cache_filename = "_".join([str(date), Config.CORPUS_NAME.value, Config.SENTIMENT_ANALYSIS_METHOD.value.name])

    topic_extraction_cache_filename = os.path.join(Config.FEATURE_CACHE_DIR.value, topic_extraction_cache_filename)
    sentiment_analysis_cache_flename = os.path.join(Config.FEATURE_CACHE_DIR.value, sentiment_analysis_cache_filename)

    topics_precomputed = os.path.exists(topic_extraction_cache_filename)
    sentiments_precomputed = os.path.exists(sentiment_analysis_cache_filename)'''
    #TODO: If we load pre-built models from disk, we can avoid all this work on start up
    print("Loading corpora...")
    approval_ratings, political_article_corpora = Runner.init_corpora()
    print("Done.")
    print("Building features...")
    features_by_day = Runner.corpora_to_day_features(political_article_corpora)
    print("Done.")
    print("Combining features...")
    features_by_range = Runner.combine_day_ranges(features_by_day)
    print("Done.")
    print("Matching features to labels...")
    X, Y = Runner.match_features_to_labels(features_by_range, approval_ratings)
    print("Done.")
    #TODO: use model type specified in config
    model = LinearRegressionModel([X, Y]) # Train using all data.
    print("Training model...")
    model.train()
    print("Done.")
    print("Server set up. Ready to go!")
    pass

# -------------End Points-------------------
@app.route('/')
def index():
    return "MAIN PAGE UNDER CONSTRUCTION"

# expects a GET request attribute "text"
# outputs {sentiment: double}
@app.route('/model/sentiment', methods=['GET'])
def get_sentiment():
    text = request.args.get('text')
    tokens = word_tokenize(text)
    if Config.DEBUG_WEBAPP.value:
        print("RECEIVED text: " + text)
    sentiment_ratio = Config.SENTIMENT_ANALYSIS_METHOD.value.value(tokens)
    if Config.DEBUG_WEBAPP.value:
        print("GOT SENTIMENT OF: " + str(sentiment_ratio))
    return jsonify({'sentiment': sentiment_ratio})

# expects a GET request attribute "text"
# outputs {topic: [...]}
@app.route('/model/topic', methods=['GET'])
def get_topic():
    text = request.args.get('text')
    tokens = word_tokenize(text)
    if Config.DEBUG_WEBAPP.value:
        print("RECEIVED text: " + text)
    topics = Config.TOPIC_EXTRACTION_METHOD.value.value(tokens)
    if Config.DEBUG_WEBAPP.value:
        print("GOT TOPIC WEIGHTS OF: " + str(topics))
    return jsonify({'topics': topics})

# TODO: Should we move the calculation all to the server side?
# expects a GET request attribute "topics" as an array of doubles
# and an attribute "sentiment" as a double
# outputs {result: [...]} where the array indices correspond to approve, disapprove, neutral
@app.route('/model/predict', methods=['GET'])
def get_predict():
    topics = request.args.get('topics')
    sentiment = request.args.get('sentiment')
    features = []
    features += topics
    features.append(sentiment)
    if model is not None:
        output = model.predict(features)
        return jsonify({'result': output})
    return jsonify({'error': 'No suitable model loaded'})

if __name__ == '__main__':
    #init_server()
    app.run(debug=True)
