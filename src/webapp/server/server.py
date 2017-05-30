# uncompyle6 version 2.9.11
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 |Anaconda custom (64-bit)| (default, Jul  2 2016, 17:53:06) 
# [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
# Embedded file name: /home/aakash/Code/StateOfTheMedia/src/webapp/server/server.py
# Compiled at: 2017-05-23 04:46:02
# Size of source mod 2**32: 12797 bytes
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
from util.config import Config
from preprocess_text.document import Document
from model.overall_runner import corpus_to_day_features
from preprocess_text.corpus import Corpus
from operator import add
from collections import defaultdict
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
articles_for_day = defaultdict(list)

def init_server():
    global labels
    global model
    global features_by_range
    global approval_ratings
    global features_by_day
    global articles_for_day
    print('Initializing server')
    approval_ratings, political_article_corpora = Runner.init_corpora()
    features_by_day = Runner.corpora_to_day_features(political_article_corpora)
    features_by_range = Runner.combine_day_ranges(features_by_day)
    print('Loading model from disk: ' + Config.TARGET_MODEL)
    print('Working...')
    model = LinearRegressionModel.load(Config.MODEL_DIR + Config.TARGET_MODEL)
    print('Done.')
    print('Loading labels from disk: ' + Config.TARGET_LABELS)
    print('Working...')
    with open(Config.DATA_DIR + Config.TARGET_LABELS, mode='rb') as f:
        labels = pickle.load(f)
    print('Done.')

    print('Loading historical articles...')
    nyt_article_archive = json.load(open('../data/NYT_Articles_October_1998.json', 'rb'))
    article_list = nyt_article_archive['response']['docs']
    for article in article_list:
        article_date = parse(article['pub_date'])
        articles_for_day[article_date.day].append({
            'headline': article['headline']['main'],
            'content': article['lead_paragraph']
        })

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
    title = ''
    tokens = word_tokenize(data['text'])
    for i in range(8):
        if i < len(tokens):
            title = title + tokens[i] + ' '

    clients[data['id']]['titles'].append(title)
    return (
     jsonify('Successfully added article'), 200)


@app.route('/article/remove', methods=['POST'])
def remove_article():
    data = json.loads(request.data.decode('utf-8'))
    if Config.DEBUG_WEBAPP:
        print('REMOVING ARTICLE ' + str(data['index']) + ' for client #' + str(data['id']))
    if data['id'] not in clients.keys():
        return ('Nothing to remove, client not found', 400)
    articles = clients[data['id']]['articles']
    titles = clients[data['id']]['titles']
    if Config.DEBUG_WEBAPP:
        print('PRE-REMOVE: ' + str(titles))
    if data['index'] >= len(articles):
        return ('Index out of bounds', 400)
    del articles[data['index']]
    del titles[data['index']]
    if Config.DEBUG_WEBAPP:
        print('POST-REMOVE: ' + str(clients[data['id']]['titles']))
    return ('Successfully deleted article #' + str(data['index']), 200)


@app.route('/article/get', methods=['POST'])
def get_article():
    data = json.loads(request.data.decode('utf-8'))
    client_id = data['id']
    if client_id not in clients.keys():
        return ('Nothing to get, client not found', 400)
    articles = clients[client_id]['articles']
    titles = clients[client_id]['titles']
    article_index = data['index']
    if article_index >= len(articles) or article_index < 0:
        return ('Index out of bounds', 400)
    return jsonify({'title': titles[article_index],'text': articles[article_index]})


@app.route('/article/titles', methods=['POST'])
def get_titles():
    data = json.loads(request.data.decode('utf-8'))
    client_id = data['id']
    if client_id not in clients.keys():
        return ('Client not found', 400)
    titles = clients[client_id]['titles']
    return jsonify({'titles': titles})


@app.route('/model/sentiment', methods=['GET'])
def get_sentiment():
    client_id = request.args.get('id')
    article_index = request.args.get('index')
    if client_id not in clients.keys():
        return ('No record for id: ' + str(client_id), 400)
    articles = clients[client_id]['articles']
    if article_index >= len(articles):
        return ('No article for given index: ' + str(article_index), 400)
    text = articles[article_index]
    tokens = word_tokenize(text)
    sentiment_ratio = Config.SENTIMENT_ANALYSIS_METHOD.value(tokens)
    return jsonify({'sentiment': sentiment_ratio})


@app.route('/model/sentimentForDay', methods=['POST'])
def get_sentiment_list():
    print("SENTIMENT ANALYSIS CALLED")
    data = json.loads(request.data.decode('utf-8'))
    client_id = data['id']
    article_day = data['day']
    article_list = data['articles']
    if client_id not in clients.keys():
        return ('No record for id: ' + str(client_id), 400)
    sentiment_ratio_day = 0.0
    for article in article_list:
        text = article['articleText']
        tokens = word_tokenize(text)
        sentiment_ratio = Config.SENTIMENT_ANALYSIS_METHOD.value(tokens)
        sentiment_ratio_day += sentiment_ratio

    sentiment_ratio_day /= len(article_list)
    return jsonify({'sentiment': sentiment_ratio_day})


@app.route('/model/topic', methods=['GET'])
def get_topic():
    client_id = request.args.get('id')
    article_index = request.args.get('index')
    if client_id not in clients.keys():
        return ('No record for id: ' + str(client_id), 400)
    articles = clients[client_id]['articles']
    if article_index >= len(articles):
        return ('No article for given index: ' + str(article_index), 400)
    text = articles[article_index]
    tokens = word_tokenize(text)
    topics = Config.TOPIC_EXTRACTION_METHOD.value(tokens)
    return jsonify({'topics': topics})


@app.route('/model/topicMixtureForDay', methods=['POST'])
def get_topic_list():
    print("TOPIC ANALYSIS CALLED")
    data = json.loads(request.data.decode('utf-8'))
    client_id = data['id']
    article_day = data['day']
    article_list = data['articles']
    if client_id not in clients.keys():
        return ('No record for id: ' + str(client_id), 400)
    topic_mixtures = None
    for article in article_list:
        text = article['articleText']
        tokens = word_tokenize(text)
        topics = Config.TOPIC_EXTRACTION_METHOD.value(' '.join(tokens))
        if topic_mixtures is None:
            topic_mixtures = topics
        else:
            for topic_idx, topic_val in enumerate(topic_mixtures):
                topic_mixtures[topic_idx] += topics[topic_idx]

    for topic_idx, topic_val in enumerate(topic_mixtures):
        topic_mixtures[topic_idx] /= len(topic_mixtures)

    return jsonify(
        {
            'topicStrengths': topic_mixtures,
            'topicLabels': ['World Leaders', 'Countries', 'Politics',
                            'Economics', 'Foreign', 'Social', 'Environment']
        })


@app.route('/model/history', methods=['GET'])
def get_prediction_history():
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


@app.route('/model/predict', methods=['POST'])
def get_predict():
    print("PREDICT CALLED")
    if model is None:
        return ('No model loaded', 400)

    data = json.loads(request.data.decode('utf-8'))
    client_id = data['id']
    if client_id not in clients.keys():
        return ('No client found for id: ' + str(client_id), 400)
    article_day = data['day']
    article_list = data['articles']
    total_sentiment = 0.0
    total_topics = None
    article_count = 0
    for article in article_list:
        text = article['articleText']
        tokens = word_tokenize(text)
        doc_topics = Config.TOPIC_EXTRACTION_METHOD.value(text)
        doc_sentiment = Config.SENTIMENT_ANALYSIS_METHOD.value(text)
        total_sentiment += doc_sentiment
        if total_topics is None:
            total_topics = doc_topics
        else:
            for i in range(len(total_topics)):
                total_topics[i] += doc_topics[i]
        article_count += 1

    for i in range(len(total_topics)):
        total_topics[i] = total_topics[i] / article_count

    total_sentiment = total_sentiment / article_count

    features = total_topics + [total_sentiment]
    output = model.predict(features)
    print(output)
    return jsonify({'prediction': output[0].tolist()[0]})


@app.route('/register', methods=['POST'])
def register():
    data = json.loads(request.data.decode('utf-8'))
    if data['id'] in clients.keys():
        return ('There is already an entry for ' + str(data['id']), 400)
    clients[data['id']] = {'articles': [],'titles': []}
    return (
     'Registered ID: ' + str(data['id']), 200)
    return ('NO ID FOUND', 400)

@app.route('/news/<int:year>/<int:month>/<int:day>', methods=['GET'])
def fetch_news(year, month, day):
    if year != 1998:
        return ('INVALID YEAR', 400)
    elif month != 10:
        return ('INVALID MONTH', 400)
    else:
        return jsonify(articles_for_day[day])

if __name__ == '__main__':
    init_server()
    app.run(debug=True, use_reloader=False)
    print('App is running')
# okay decompiling ../__pycache__/server.cpython-35.pyc
