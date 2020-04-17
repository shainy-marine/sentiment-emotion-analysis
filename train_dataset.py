# Trained data with emotions using Logistic Regression Algorithm
import logging
import warnings
from warnings import simplefilter
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
simplefilter(action='ignore', category=FutureWarning)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
sns.set_context("talk")
model = None
vectorizer = None
def train_data():
    """
    Train data to calculate Emotion
    """
    logger = logging.getLogger(__name__)
    global model
    global vectorizer
    data = pd.read_csv("./resources/emotion.data")
    data = data.drop(['slNo'], axis=1)
    X = data['text']
    y = data['emotions']
    train, test = train_test_split(data, test_size=0.2, random_state=12, stratify=data['emotions'])
    X_train = train.text
    y_train = train.emotions
    X_test = test.text
    y_test = test.emotions
    vectorizer = TfidfVectorizer(max_df=0.9).fit(X_train)
    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)
    logger.info(X_train.shape)
    encoder = LabelEncoder().fit(y_train)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)
    model = LogisticRegression(C=.1, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    logger.info("Training Accuracy Score:" + str(accuracy_score(y_train, y_pred_train)))
    logger.info("Testing Accuracy Score" + str(accuracy_score(y_test, y_pred_test)))
def predicts(x):
    tfidf = vectorizer.transform([x])
    preds = model.predict_proba(tfidf)[0]
    return preds