# main function
import json
import logging
import logging.config
import os

from flask import Flask, request, jsonify

from scripts import detection
from scripts.train_dataset import train_data

application = Flask(__name__)


def setup_logging(
        default_path='./config/logging.json',
        default_level=logging.INFO,
        env_key='LOG_CFG'
):
    """Setup logging configuration"""
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


@application.route("/db_calculation", methods=['GET'])
def db_calculate_data():
    """
        calculate sentiment and emotion for data
       Giving input as data from Redshift database
    """

    logger = logging.getLogger(__name__)
    response_obj = detection.calculate_db_data()

    return jsonify(response_obj)


@application.route("/sentiment", methods=['GET', 'POST'])
def get_sentiment():
    """
     calculate sentiment using SentimentIntensityAnalyzer for each passed input text
     """
    text = ""
    logger = logging.getLogger(__name__)
    if request.method == 'GET':
        """return the information for <text>"""
        text = request.args.get('text')
        logger.info("Input text : " + text)
    if request.method == 'POST':
        """Post method request {text : <text data>}"""
        input_json = request.get_json()
        logger.info(input_json)
        text = input_json['text']
    sentiment = detection.calculate_sentiment(text)
    # sentiment_obj = {'text': text, 'sentiment': sentiment}
    # logger.info(sentiment_obj)
    logger.info(sentiment)
    return jsonify(sentiment)


@application.route("/emotion", methods=['GET', 'POST'])
def get_emotion():
    """
    calculate emotion for each passed input text
    """
    logger = logging.getLogger(__name__)
    text = request.args.get('text')
    emotion = detection.calculate_emotion(text)
    logger.info("Input text : " + text)
    # emotion_obj = {'text': text, 'emotion': emotion}
    # logger.info(emotion_obj)
    logger.info(emotion)
    return jsonify(emotion)


@application.route("/", methods=['GET'])
@application.route('/index')
def default():
    return "<h1> Sentimizer Service is up & running <h1>"


if __name__ == "__main__":
    setup_logging()
    train_data()
    application.debug = True
    application.run()
