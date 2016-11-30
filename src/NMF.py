from __future__ import print_function
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
from nltk import word_tokenize
from nltk.stem import snowball
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


def describe_nmf_results(document_term_mat, feature_words, W, H, X,n_top_words = 15,n_top_tweets = 5):
    with open('../Data/nmf_results2.txt','wb') as f:
        for topic_num, topic in enumerate(H):
            print("\nTopic %d:" % topic_num,file=f)
            print(" ".join([feature_words[i] \
                    for i in topic.argsort()[:-n_top_words - 1:-1]]),file=f)
            print("\nTopic {} tweets:".format(topic_num),file=f)
            print("\n".join([X.iloc[i] for i in \
                    W[:,topic_num].argsort()[:-n_top_tweets - 1:-1]]),file=f)
            print("-"*50,file=f)
        return


def get_topics(doc_path,vect_file,n_topics):
    vect = pickle.load(open(vect_file, 'rb'))
    df = pd.read_pickle('../Data/NB_predicted_tweets.p')
    df = df[df.is_hate_NB == 1]
    X = df.text
    document_term_mat = pickle.load(open('../Data/NB_tweet_matrix.p','rb'))
    # dense_document_term_mat = document_term_mat.todense()
    feature_words = vect.get_feature_names()
    nmf = NMF(n_components=n_topics)
    W = nmf.fit_transform(document_term_mat)
    H = nmf.components_
    describe_nmf_results(document_term_mat,feature_words,W,H,X)

def main():

    path = '../Data/labeledRedditComments2.p'
    get_topics(path,'vect2.p',25)

if __name__ == '__main__':
    main()
