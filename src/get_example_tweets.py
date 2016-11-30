from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import cPickle as pickle
from string import punctuation
from nltk import word_tokenize
from nltk.stem import snowball
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from nltk.tokenize import PunktSentenceTokenizer
import time
from itertools import izip

stemmer = snowball.SnowballStemmer("english")

###############################################################################
#OHS tokenization code

def load_data(filename):
    '''
    Load data into a data frame for use in running model
    '''
    return pickle.load(open(filename, 'rb'))


def stem_tokens(tokens, stemmer):
    '''Stem the tokens.'''
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def OHStokenize(text):
    '''Tokenize & stem. Stems automatically for now.
    Leaving "stemmer" out of function call, so it works with TfidfVectorizer'''
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems



def get_example_tweets(cross_val_file,model_pickle,vectorizer_pickle, n_tweets):
    with open(model_pickle, 'r') as f:
        clf = pickle.load(f)
    with open(vectorizer_pickle, 'r') as f:
        vect = pickle.load(f)
    dfcv = pd.read_pickle(cross_val_file)
    X = dfcv.text
    y = dfcv.label
    text_counts = vect.transform(dfcv['text'].values)
    X = pd.DataFrame(X)
    X['probability'] = clf.predict_proba(text_counts)[:,1:]
    X['prediction'] = clf.predict(text_counts)
    X['actual'] = y
    X.columns = ['text', 'probability', 'prediction', 'actual']
    X_fp = X[(X.prediction == True) & (X.actual== False)].sort_values(by='probability',ascending=True)
    X_tp = X[(X.prediction == True) & (X.actual == True)].sort_values(by='probability',ascending=True)
    X_fn = X[(X.prediction == False) & (X.actual== True)].sort_values(by='probability',ascending=False)
    X_tn = X[(X.prediction == False) & (X.actual== False)].sort_values(by='probability',ascending=False)
    db_list = [X_fp,X_tp,X_fn,X_tn]
    name_list = ['False Positive', 'True Positive', 'False Negative', 'True Negative']
    for db, name in izip(db_list,name_list):
        print "The most uncertain {} tweets from the {} category:".format(n_tweets,name)
        print db.iloc[:n_tweets]['text'].values
        print "-"*28
    for db, name in izip(db_list,name_list):
        print "The most certain {} tweets from the {} category:".format(n_tweets,name)
        print db.iloc[-n_tweets:]['text'].values
        print "-"*28


if __name__ == '__main__':
    get_example_tweets('../Data/cv.p',\
                       '../Data/hate_modelNB.pickle','vect.p',5)
