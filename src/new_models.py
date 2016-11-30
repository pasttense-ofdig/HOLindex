import pandas as pd
import numpy as np
import pickle
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from unidecode import unidecode
from chardet.universaldetector import UniversalDetector
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from pymongo import MongoClient
from textblob import TextBlob
from mongo_tweet_reader import get_tweets, get_full_tweet_info
from itertools import izip
from sklearn.decomposition import NMF

def create_xgb(Xfile,yfile):
    X = pd.read_pickle(Xfile)
    y = pd.read_pickle(yfile)
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    print 'creating vectorizer...'
    vect = TfidfVectorizer(stop_words='english', decode_error='ignore')
    print 'vectorizing...'
    tfidf_X = vect.fit_transform(X_train)
    tfidf_Xtest = vect.transform(X_test)
    print 'converting data...'
    # tfidf_Xtest = tfidf_Xtest.todense()
    # tfidf_X = tfidf_X.todense()
    xg_train = xgb.DMatrix(tfidf_X,label=y_train)
    xg_test = xgb.DMatrix(tfidf_Xtest,label=y_test)

    print 'training...'
    param = {'max_depth':4,
         'eta':0.3,
         'silent':1,
         'objective':'binary:logistic',
         'eval_metric':'auc'
         }
    num_round = 163
    watchlist = [(xg_train,'train'), (xg_test, 'eval')]
    results = dict()
    model = xgb.train(param, xg_train, num_round, watchlist, evals_result=results,verbose_eval=True)
    xg_train.save_binary('../Data/xg_train.buffer')
    model.save_model('../Data/xgb.model')
    # print "Results: ", results


if __name__ == '__main__':
    create_xgb('../Data/X.pickle','../Data/y.pickle')
