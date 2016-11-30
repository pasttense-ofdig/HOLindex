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
import seaborn as sb

stemmer = snowball.SnowballStemmer("english")

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

def main():
    model = pickle.load(open('../Data/hate_modelNB.pickle','rb'))
    feature_probs = np.exp(model.feature_log_prob_[:1,:].flatten())
    top_features_idx = feature_probs.argsort()[-10:-1]
    top_feature_probs = feature_probs[top_features_idx]
    vect = pickle.load(open('vect.p','rb'))
    feature_names = np.array(vect.get_feature_names())
    
