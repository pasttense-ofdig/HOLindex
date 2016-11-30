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
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import datetime

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



def predict_timeline_tweets(model_pickle,vectorizer_pickle,reload_data=False,show=False):
    print "loading data and models..."
    with open(model_pickle, 'r') as f:
        clf = pickle.load(f)
    # with open(vectorizer_pickle, 'r') as f:
    #     vect = pickle.load(f)
    if reload_data:
        tweet_df = get_full_tweet_info('USTweets','timelinetweets')
        tweet_df['date'] = pd.to_datetime(tweet_df['date'])
        mask = tweet_df['date'] > '2016-1-1'
        tweet_df = tweet_df[mask]
        tweet_df.to_pickle('../Data/tweets.pkl')
    tweet_df = pd.read_pickle('../Data/tweets.pkl')
    # liberal_df = pd.read_pickle('../Data/liberal_predictions.p')
    # return tweet_df.groupby('date').count()
    # import pdb; pdb.set_trace()
    # text_counts = vect.transform(tweet_df['text'])
    text_counts = pickle.load(open('../Data/tweet_text.p','rb'))
    tweet_df['is_hate_NB'] = clf.predict(text_counts)
    tweet_df = tweet_df[tweet_df.date > '2016-01-01']
    tweet_df.to_pickle('../Data/NB_predicted_tweets.p')
    chart_df = tweet_df[['date', 'is_hate_NB']]
    hate_tweets = chart_df.groupby(chart_df['date'].dt.normalize()).sum()
    hate_tweets.columns = ['sum']
    total_tweets = chart_df.groupby(chart_df['date'].dt.normalize()).count()
    total_tweets.drop('is_hate_NB',inplace=True,axis=1)
    total_tweets.columns = ['count']
    combined_hate = hate_tweets.join(total_tweets)
    combined_hate['percent'] = combined_hate['sum'] / combined_hate['count']
    # liberal_hate_tweets = liberal_df.groupby(liberal_df['date'].dt.normalize()).sum()
    # liberal_hate_tweets.columns = ['sum']
    # liberal_combined = liberal_hate_tweets.join(total_tweets)
    # liberal_combined['percent'] = liberal_combined['sum'] / liberal_combined['count']
    import pdb; pdb.set_trace()
    return combined_hate #, liberal_combined

def plot_combined(combined_hate):
    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    yearsFmt = mdates.DateFormatter('%m-%Y')
    fig, ax = plt.subplots()
    ax.plot(combined_hate.index,combined_hate['percent'],'r')
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(yearsFmt)
    fig.autofmt_xdate()
    # plt.plot(liberal_combined.index,liberal_combined['percent'],'b')
    #
    # plt.locator_params(nbins=6)
    # loc = ax.xaxis.get_major_locator()
    # loc.maxticks[DAILY] = 8

    # if show:
    #     plt.show()
    plt.savefig('final_plot_NB.png')




def main():

    print 'loading data and model'
    model_path = '../Data/hate_modelNB.pickle'
    vector_path = '../Data/vect.p'
    print 'predicting tweets'
    combined = predict_timeline_tweets(model_path,vector_path)
    print 'making chart'
    plot_combined(combined)



if __name__ == '__main__':
    main()
