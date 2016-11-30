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
import time

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
    print "entering main..."

    path = '../Data/labeledRedditComments2.p'
    ppath = '../Data/cv.p'

    load_tstart = time.time()
    print 'loading data...'
    df = load_data(path)
    dfcv = pd.read_pickle(ppath)
    load_tstop = time.time()

    #take a subset of the data for testing this code
    # randNums = np.random.randint(low=0,high=len(df.index),size=(200,1))
    # rowList = [int(row) for row in randNums]
    # dfsmall = df.ix[rowList,:]

    nf = df

    #create training set and labels
    X = nf.body
    y = nf.label
    Xcv = dfcv['text'].values
    ycv = dfcv['label'].values

    # vect_tstart = time.time()
    # print "creating vectorizer..."
    # vect = TfidfVectorizer(stop_words='english', decode_error='ignore',
    #                        tokenizer=OHStokenize)
    #
    # print "vectorizing..."
    # # fit & transform comments matrix
    # tfidf_X = vect.fit_transform(X)
    #
    # print "pickling vectorizer..."
    # pickle.dump(vect, open('vect2.p', 'wb'))
    print 'vectorizing'
    # vect = pickle.load(open('vect2.p', 'rb'))
    # tfidf_X = vect.transform(X)
    # tfidf_Xcv = vect.transform(Xcv)
    # with open('../Data/doc_matrix.pickle','wb') as f:
    #     pickle.dump(tfidf_X,f)
    # vect_tstop = time.time()

    X = pickle.load(open('../Data/doc_matrix.pickle','rb'))
    print 'training model'
    bst = GradientBoostingClassifier(n_estimators = 150, learning_rate = 0.1)
    bst.fit_transform(X,y)
    print 'pickling'
    with open('../Data/boost4.model','wb') as f:
        pickle.dump(bst,f)









if __name__ == '__main__':
    main()
