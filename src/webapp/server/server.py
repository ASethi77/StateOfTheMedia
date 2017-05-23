#!flask/bin/python

# This is the back-end REST api for making predictions about text
# make a GET request to the following endpoints to receive JSON data

# to populate the web-app.
#
# /model/predict
# /model/sentiment
# /model/topic
#
# RUN WITH: python3 -m webapp.server.server (from src directory)

# code to fix PYTHONPATH
import pickle
import sys
import os

import datetime
from dateutil.parser import parse
from sklearn.externals import joblib

PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from flask import Flask, request, jsonify, json
from flask_cors import CORS, cross_origin

import pickle
import json
import model.overall_runner as Runner
import model.sentiment_analysis as Sentiment
import model.topic_extractor as Topic
from nltk.tokenize import word_tokenize

from model.linear_regression_model import LinearRegressionModel
from evaluation.load_labels import LabelLoader
from model.regression_model import RegressionModel
# from model.MLPRegressionModel import MLPRegressionModel
from util.config import Config #, Paths, RegressionModels
from preprocess_text.document import Document
from model.overall_runner import corpus_to_day_features
from preprocess_text.corpus import Corpus
from operator import add


app = Flask(__name__, static_url_path='')
CORS(app)

sentiment_corpus = None
topic_corpus = None
model = None
labels = None
clients = {}

def init_server():
    global model
    global labels
    # '''topic_extraction_cache_filename = "_".join([str(date), Config.CORPUS_NAME.value, Config.TOPIC_EXTRACTION_METHOD.value.name])
    # sentiment_analysis_cache_filename = "_".join([str(date), Config.CORPUS_NAME.value, Config.SENTIMENT_ANALYSIS_METHOD.value.name])
    #
    # topic_extraction_cache_filename = os.path.join(Config.FEATURE_CACHE_DIR.value, topic_extraction_cache_filename)
    # sentiment_analysis_cache_filename = os.path.join(Config.FEATURE_CACHE_DIR.value, sentiment_analysis_cache_filename)
    #
    # topics_precomputed = os.path.exists(topic_extraction_cache_filename)
    # sentiments_precomputed = os.path.exists(sentiment_analysis_cache_filename)'''
    # #TODO: If we load pre-built models from disk, we can avoid all this work on start up
    # print("Loading corpora...")
    # approval_ratings, political_article_corpora = Runner.init_corpora()
    # print("Done.")
    # print("Building features...")
    # features_by_day = Runner.corpora_to_day_features(political_article_corpora)
    # print("Done.")
    # print("Combining features...")
    # features_by_range = Runner.combine_day_ranges(features_by_day)
    # print("Done.")
    # print("Matching features to labels...")
    # X, Y = Runner.match_features_to_labels(features_by_range, approval_ratings)
    # print("Done.")
    # #TODO: use model type specified in config
    # model = LinearRegressionModel([X, Y]) # Train using all data.
    # print("Training model...")
    # model.train()
    # print("Done.")
    # print("Server set up. Ready to go!")
    print("Initializing server")
    print("Loading model from disk: " + Config.TARGET_MODEL)
    print("Working...")
    model = LinearRegressionModel.load(Config.MODEL_DIR + Config.TARGET_MODEL)
    print("Done.")
    # label_loader = LabelLoader()
    # label_loader.load_json()
    # labels = label_loader.get_labels()
    # print(labels)
    print("Loading labels from disk: " + Config.TARGET_LABELS)
    print("Working...")
    with open(Config.DATA_DIR + Config.TARGET_LABELS, mode="rb") as f:
        labels = pickle.load(f)
    print("Done.")

# -------------End Points-------------------
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/article/add', methods=['POST'])
def add_article():
    data = json.loads(request.data.decode('utf-8'))
    if data['id'] not in clients.keys():
        clients[data['id']] = []
    clients[data['id']].append(data['text'])
    return "Successfully added article", 200

@app.route('/article/remove', methods=['POST'])
def remove_article():
    data = json.loads(request.data.decode('utf-8'))
    if Config.DEBUG_WEBAPP:
        print("REMOVING ARTICLE " + str(data['index']) + " for client #" + str(data['id']))
    if data['id'] not in clients.keys():
        return "Nothing to remove, client not found", 400
    else:
        articles = clients[data['id']]
        if Config.DEBUG_WEBAPP:
            print("PRE-REMOVE: " + str(articles))
        if data['index'] >= len(articles):
            return "Index out of bounds", 400
        del articles[data['index']]
        if Config.DEBUG_WEBAPP:
            print("POST-REMOVE: " + str(clients[data['id']]))
        return "Successfully deleted article #" + str(data['index']), 200
    

# expects a GET request attribute "text"
# outputs {sentiment: double}
@app.route('/model/sentiment', methods=['GET'])
def get_sentiment():
    client_id = request.args.get('id')
    article_index = request.args.get('index')
    if client_id not in clients.keys():
        return "No record for id: " + str(client_id), 400
    articles = clients[client_id]
    else if article_index >= len(articles):
        return "No article for given index: " + str(article_index), 400
    text = articles[article_index]
    tokens = word_tokenize(text)
    sentiment_ratio = Config.SENTIMENT_ANALYSIS.value(tokens)
    return jsonify({'sentiment': sentiment_ratio})

# expects a GET request attribute "text"
# outputs {topic: [...]}
@app.route('/model/topic', methods=['GET'])
def get_topic():
    client_id = request.args.get('id')
    article_index = request.args.get('index')
    if client_id not in clients.keys():
        return "No record for id: " + str(client_id), 400
    articles = clients[client_id]
    else if article_index >= len(articles):
        return "No article for given index: " + str(article_index), 400
    text = articles[article_index]
    tokens = word_tokenize(text)
    topics = Config.TOPIC_EXTRACTION_METHOD.value(tokens)
    return jsonify({'topics': topics})

@app.route('/approvalRatings', methods=['GET'])
def get_approval_ratings():
    date = request.args.get('date')
    base_date = parse(date)
    date_list = [str(base_date - datetime.timedelta(days=x)) for x in range(-5, 5, 1)]
    approval_ratings = {'approvalRatings': [70, 75, 70, 68, 80, 70, 72, 50, 80, 90, 20],
                        'labels': date_list}
    return jsonify(approval_ratings)

@app.route('/nlp', methods=['POST'])
def do_nlp():
    global model

    article_list = json.loads(request.data.decode('utf-8'))
    doc_list = []
    date = None
    for article in article_list:
        doc_list.append(Document(article['text']))
        date = article['date']

    day_corpus = Corpus(docs=doc_list)
    output = dict()
    corpus_to_day_features(date, day_corpus, output)

    x_in = None
    for v in output.values():
        x_in = v # only one value. this is dumb

    prediction = model.predict(x_in)
    response = dict()
    response['sentiment'] = x_in[-1]
    response['topicLabels'] = Config.TOPIC_NAMES
    response['topicStrings'] = x_in[0:-1]
    response['approval'] = prediction.tolist()
    return jsonify(response)

# TODO: Should we move the calculation all to the server side?
# expects a GET request attribute "docs" which is an array of strings
# and an attribute "sentiment" as a double
# outputs {result: [...]} where the array indices correspond to approve, disapprove, neutral
@app.route('/model/predict', methods=['POST'])
def get_predict():
    data = json.loads(request.data.decode('utf-8'))
    '''# TODO: use actual topic labels here
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
    return jsonify({'error': 'No suitable model loaded'})'''

# for registering a client with the server
# returns 403 error if no id passed or id already exists on server
@app.route('/register', methods=['POST'])
def register():
    global clients
    data = json.loads(request.data.decode('utf-8'))
    if data['id'] in clients.keys():
        return "There is already an entry for " + str(data['id']), 400
    else:
        clients[data['id']] = []
        return "Registered ID: " + str(data['id']), 200
    return "NO ID FOUND", 400

if __name__ == '__main__':
    init_server()
    app.run(debug=True, use_reloader=False)
    print("App is running")
