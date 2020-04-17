# Emotion and sentiment detection
import logging
import logging.config

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils import db_utils
from .train_dataset import predicts


def calculate_db_data():
    """redshift database connection"""
    logger = logging.getLogger(__name__)
    conn = db_utils.redshift_db_connect()
    cur = conn.cursor()
    cur.execute("select * from deepwaterprod.t_test_emotions order by indexedTimestamp limit 100 ")
    results = cur.fetchall()

    response_obj = []
    for result in results:
        text = result[6]
        response = {'text': text, 'sentiment': calculate_sentiment(text),
                    'emotion': calculate_emotion(text)}
        response_obj.append(response)
    logger.info(response_obj)
    return response_obj


def calculate_emotion(input_data):
    """ Anger = 1, Fear = 2, Joy = 3, Love = 4, Sadness = 5, Surprise = 6 """
    logger = logging.getLogger(__name__)
    a = predicts(input_data)
    if a[0] == max(a):
        emotion = {"label": "anger", "value": 1}
        logger.info("Emotion : ", emotion)
        return emotion
    elif a[1] == max(a):
        emotion = {"label": "fear", "value": 2}
        logger.info("Emotion : ", emotion)
        return emotion
    elif a[2] == max(a):
        emotion = {"label": "joy", "value": 3}
        logger.info("Emotion : ", emotion)
        return emotion
    elif a[3] == max(a):
        emotion = {"label": "love", "value": 4}
        logger.info("Emotion : ", emotion)
        return emotion
    elif a[4] == max(a):
        emotion = {"label": "sadness", "value": 5}
        logger.info("Emotion : ", emotion)
        return emotion
    else:
        emotion = {"label": "surprise", "value": 6}
        logger.info("Emotion : ", emotion)
        return emotion


def calculate_sentiment(input_text):
    """ Positive = 1, Neutral = 0, Negative = -1 """
    logger = logging.getLogger(__name__)
    analysis = SentimentIntensityAnalyzer()
    vs = analysis.polarity_scores(input_text)
    if vs['compound'] > 0:
        sentiment = {"label": "positive", "value": 1}
        logger.info("Sentiment : ", sentiment)
        return sentiment
    elif vs['compound'] == 0:
        sentiment = {"label": "neutral", "value": 0}
        logger.info("Sentiment : ", sentiment)
        return sentiment
    else:
        sentiment = {"label": "negative", "value": -1}
        logger.info("Sentiment : ", sentiment)
        return sentiment
