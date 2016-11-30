import pandas as pd
import numpy as np
import pickle
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DAILY
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
from nltk.stem import snowball
from nltk import word_tokenize

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



def main():
    print 'loading data...'
    vect = pickle.load(open('vect.p', 'rb'))
    nf = pickle.load(open('../Data/labeledRedditComments2.p', 'rb'))
    print 'transforming vectors'
    X = nf.body
    y = nf.label
    tfidf_X = vect.transform(X)
    print 'creating NB model'
    clf = MultinomialNB()
    clf.fit(tfidf_X,y)
    print 'pickling model'
    with open('../Data/hate_modelNB.pickle','w') as f:
        pickle.dump(clf, f)


if __name__ == '__main__':
    main()
