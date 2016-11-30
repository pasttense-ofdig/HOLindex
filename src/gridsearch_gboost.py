from __future__ import print_function
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
from sklearn.ensemble import GradientBoostingClassifier
from nltk.tokenize import PunktSentenceTokenizer
from sklearn.grid_search import GridSearchCV
import time
import multiprocessing


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

def grid_search():
    X = pickle.load(open('../Data/doc_matrix.pickle','rb'))
    df = pd.read_pickle('../Data/labeledRedditComments2.p')
    y = df.label
    n_cores = multiprocessing.cpu_count()

        # Initalize our model here
    est = GradientBoostingClassifier()

    # Here are the params we are tuning, ie,
    # if you look in the docs, all of these are 'nobs' within the GradientBoostingClassifier algo.
    param_grid = {'learning_rate': [0.001, 0.05, 0.02, 0.1],
                  'n_estimators': [100,300,500,1000],
                  'max_depth': [2, 3],
                  'min_samples_leaf': [3, 5],
                  }

    # Plug in our model, params dict, and the number of jobs, then .fit()
    gs_cv = GridSearchCV(est, param_grid, n_jobs=n_cores,scoring='roc_auc').fit(X, y)

    # return the best score and the best params
    with open('../Data/grid_search_results.txt','wb') as f:
        print('Best score: {} \n Best Params: {}'.format(gs_cv.best_score_, gs_cv.best_params_),file=f)

if __name__ == '__main__':
    grid_search()
