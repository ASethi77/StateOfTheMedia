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

from flask import Flask, request, jsonify

# import pickle
#
# import model.sentiment_analysis as Sentiment
# import model.topic_extractor as Topic
# from model.linear_regression_model import LinearRegressionModel

# from model.MLPRegressionModel import MLPRegressionModel
from util.config import Config #, Paths, RegressionModels
from preprocess_text.document import Document

from operator import add


app = Flask(__name__)

sentiment_corpus = None
topic_corpus = None
model = None

# def init_server():
#     '''topic_extraction_cache_filename = "_".join([str(date), Config.CORPUS_NAME.value, Config.TOPIC_EXTRACTION_METHOD.value.name])
#     sentiment_analysis_cache_filename = "_".join([str(date), Config.CORPUS_NAME.value, Config.SENTIMENT_ANALYSIS_METHOD.value.name])
#
#     topic_extraction_cache_filename = os.path.join(Config.FEATURE_CACHE_DIR.value, topic_extraction_cache_filename)
#     sentiment_analysis_cache_filename = os.path.join(Config.FEATURE_CACHE_DIR.value, sentiment_analysis_cache_filename)
#
#     topics_precomputed = os.path.exists(topic_extraction_cache_filename)
#     sentiments_precomputed = os.path.exists(sentiment_analysis_cache_filename)'''
#     #TODO: If we load pre-built models from disk, we can avoid all this work on start up
#     print("Loading corpora...")
#     approval_ratings, political_article_corpora = Runner.init_corpora()
#     print("Done.")
#     print("Building features...")
#     features_by_day = Runner.corpora_to_day_features(political_article_corpora)
#     print("Done.")
#     print("Combining features...")
#     features_by_range = Runner.combine_day_ranges(features_by_day)
#     print("Done.")
#     print("Matching features to labels...")
#     X, Y = Runner.match_features_to_labels(features_by_range, approval_ratings)
#     print("Done.")
#     #TODO: use model type specified in config
#     model = LinearRegressionModel([X, Y]) # Train using all data.
#     print("Training model...")
#     model.train()
#     print("Done.")
#     print("Server set up. Ready to go!")
#     pass

# -------------End Points-------------------
@app.route('/')
def index():
    return "MAIN PAGE UNDER CONSTRUCTION"

# expects a GET request attribute "text"
# outputs {sentiment: double}
@app.route('/model/sentiment', methods=['GET'])
def get_sentiment():
    text = request.args.get('text')
    print(text)
    sentiment_ratio = Config.SENTIMENT_ANALYSIS_METHOD.value.value(Document(content=text))
    return jsonify({'sentiment': sentiment_ratio})

# expects a GET request attribute "text"
# outputs {topic: [...]}
@app.route('/model/topic', methods=['GET'])
def get_topic():
    text = request.args.get('text')
    topics = Config.TOPIC_EXTRACTION_METHOD.value.value(text)
    return jsonify({'topics': topics})

@app.route('/nlp', methods=['POST'])
def do_nlp():
    print("doing the nlp")
    pass

# TODO: Should we move the calculation all to the server side?
# expects a GET request attribute "docs" which is an array of strings
# and an attribute "sentiment" as a double
# outputs {result: [...]} where the array indices correspond to approve, disapprove, neutral
@app.route('/model/predict', methods=['POST'])
def get_predict():
    # TODO: use actual topic labels here
    TOPIC_LABELS = ["A", "B", "C", "D", "E", "F"]
    request_json = request.get_json()
    data = json.loads(request_json)
    total_sentiment = 0.0
    total_topics = [0.0]
    for text in data:
        doc_topics = Config.TOPIC_EXTRACTION_METHOD.value.value(text)
    doc_sentiment = Config.SENTIMENT_ANALYSIS_METHOD.value.value(text)
    total_sentiment = map(add, total_sentiment, doc_topics)
    total_sentiment += doc_sentiment
    for indx in range(len(total_sentiment)):
        total_sentiment[indx] = total_sentiment[indx] / len(data)
    total_sentiment = total_sentiment / len(data)
  
    if model is not None:
        features = total_topics + total_sentiment
        output = model.predict(features)
    return jsonify({'sentiment': total_sentiment, 'topicStrengths': total_topics, 'topicLabels': TOPIC_LABELS, 'approval': output[0][0]})
    return jsonify({'error': 'No suitable model loaded'})

if __name__ == '__main__':
    #init_server()
    app.run(debug=True)
    print("App is running")