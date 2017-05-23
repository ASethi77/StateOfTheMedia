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
approval_ratings = {}
features_by_day = {}
features_by_range = {}
clients = {}

def init_server():
    global model
    global labels
    global approval_ratings
    global features_by_day
    global features_by_range

    print("Initializing server")
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
    approval_ratings, political_article_corpora = Runner.init_corpora()
    # print("Done.")
    # print("Building features...")
    features_by_day = Runner.corpora_to_day_features(political_article_corpora)
    # print("Done.")
    # print("Combining features...")
    features_by_range = Runner.combine_day_ranges(features_by_day)
    # print("Done.")
    # print("Matching features to labels...")
    #X, Y = Runner.match_features_to_labels(features_by_range, approval_ratings)
    # print("Done.")
    # #TODO: use model type specified in config
    # model = LinearRegressionModel([X, Y]) # Train using all data.
    # print("Training model...")
    # model.train()
    # print("Done.")
    # print("Server set up. Ready to go!")
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
    global clients
    data = json.loads(request.data.decode('utf-8'))
    if data['id'] not in clients.keys():
        clients[data['id']] = []
    clients[data['id']]['articles'].append(data['text'])
    #for now use the first 8 words
    title = ""
    tokens = word_tokenize(data['text'])
    for i in range(len(8)):
        if i < len(tokens):
            title = title + tokens[i] + " "
    clients[data['id']]['titles'].append(title)
    #clients[data['id']]['titles'].append(data['title'])
    return "Successfully added article", 200

@app.route('/article/remove', methods=['POST'])
def remove_article():
    global clients
    data = json.loads(request.data.decode('utf-8'))
    if Config.DEBUG_WEBAPP:
        print("REMOVING ARTICLE " + str(data['index']) + " for client #" + str(data['id']))
    if data['id'] not in clients.keys():
        return "Nothing to remove, client not found", 400
    else:
        articles = clients[data['id']]['articles']
        titles = clients[data['id']]['titles']
        if Config.DEBUG_WEBAPP:
            print("PRE-REMOVE: " + str(titles))
        if data['index'] >= len(articles):
            return "Index out of bounds", 400
        del articles[data['index']]
        del titles[data['index']]
        if Config.DEBUG_WEBAPP:
            print("POST-REMOVE: " + str(clients[data['id']]['titles']))
        return "Successfully deleted article #" + str(data['index']), 200

@app.route('/article/get', methods=['POST'])
def get_article():
    global clients
    data = json.loads(request.data.decode('utf-8'))
    client_id = data['id']
    if client_id not in clients.keys():
        return "Nothing to get, client not found", 400
    articles = clients[client_id]['articles']
    titles = clients[client_id]['titles']
    article_index = data['index']
    if article_index >= len(articles) or article_index < 0:
        return "Index out of bounds", 400
    else:
        return jsonify({'title': titles[article_index], 'text': articles[article_index]})   

@app.route('/article/titles', methods=['POST'])
def get_titles():
    global clients
    data = json.loads(request.data.decode('utf-8'))
    client_id = data['id']
    if client_id not in clients.keys():
        return "Client not found", 400
    titles = clients[client_id]['titles']
    return jsonify({'titles': titles})

# expects a GET request attribute "id" and "index"
# outputs {sentiment: double}
@app.route('/model/sentiment', methods=['GET'])
def get_sentiment():
    global clients
    client_id = request.args.get('id')
    article_index = request.args.get('index')
    if client_id not in clients.keys():
        return "No record for id: " + str(client_id), 400
    articles = clients[client_id]['articles']
    if article_index >= len(articles):
        return "No article for given index: " + str(article_index), 400
    text = articles[article_index]
    tokens = word_tokenize(text)
    sentiment_ratio = Config.SENTIMENT_ANALYSIS.value(tokens)
    return jsonify({'sentiment': sentiment_ratio})

# expects a GET request attribute "id" and "index"
# outputs {topic: [...]}
@app.route('/model/topic', methods=['GET'])
def get_topic():
    global clients
    client_id = request.args.get('id')
    article_index = request.args.get('index')
    if client_id not in clients.keys():
        return "No record for id: " + str(client_id), 400
    articles = clients[client_id]['articles']
    if article_index >= len(articles):
        return "No article for given index: " + str(article_index), 400
    text = articles[article_index]
    tokens = word_tokenize(text)
    topics = Config.TOPIC_EXTRACTION_METHOD.value(tokens)
    return jsonify({'topics': topics})

@app.route('/model/history', methods=['GET'])
def get_prediction_history():
    global approval_ratings
    global features_by_day
    global features_by_range
    data = json.loads(request.data.decode('utf-8'))
    start_date = parse(data['start'])
    num_days = data['days']
    actual_labels = []
    predicted_labels = []
    for i in range(num_days):
        offset = timedelta(days=i)
        target_date = start_date + offset
        feature = features_by_range[target_date]
        actual_label = approval_ratings[target_date + timedelta(days=Config.POLL_DELAY)]
        if feature is not None and actual_label is not None:
           predict_label = model.predict(feature)
           actual_labels.append(actual_label)
           predicted_labels.append(predict_label[0])
        else:
            actual_labels.append([0.0, 0.0, 0.0])
            predicted_labels.append([0.0, 0.0, 0.0])
    return jsonify()

# TODO: Should we move the calculation all to the server side?
# expects a GET request attribute "docs" which is an array of strings
# and an attribute "sentiment" as a double
# outputs {result: [...]} where the array indices correspond to approve, disapprove, neutral
@app.route('/model/predict', methods=['POST'])
def get_predict():
    global clients
    data = json.loads(request.data.decode('utf-8'))
    client_id = data['id']
    if client_id not in clients.keys():
        return "No client found for id: " + str(client_id), 400
    articles = clients[data['id']]['articles']
    total_sentiment = 0.0
    total_topics = [0.0]
    article_count = 0
    for index in data['articles']:
        if index < len() and index >= 0:
            text = articles[index]
            tokens = word_tokenize(text)
            doc_topics = Config.TOPIC_EXTRACTION_METHOD.value(text)
            doc_sentiment = Config.SENTIMENT_ANALYSIS_METHOD.value(text) 
            total_sentiment += doc_sentiment
            total_topics = map(add, total_topics, doc_topics)
            article_count += 1
    # normalize data
    for i in range(len(totel_sentiment)):
        total_sentiment[i] = total_sentiment[i] / article_count
    total_sentiment = total_sentiment / article_count
    if model is not None:
        features = total_topics + total_sentiment
        output = model.predict(features)
        return jsonify({'prediction': output[0]})
    else:
        return "No model loaded", 400

# for registering a client with the server
# returns 403 error if no id passed or id already exists on server
@app.route('/register', methods=['POST'])
def register():
    global clients
    data = json.loads(request.data.decode('utf-8'))
    if data['id'] in clients.keys():
        return "There is already an entry for " + str(data['id']), 400
    else:
        clients[data['id']] = {'articles': [], 'titles': []}
        return "Registered ID: " + str(data['id']), 200
    return "NO ID FOUND", 400

if __name__ == '__main__':
    init_server()
    app.run(debug=True, use_reloader=False)
    print("App is running")
