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

def predict_gb():
    print "entering main..."
    modelpath = '../Data/boost3.model'

    print "loading model..."
    model = pickle.load(open(modelpath,'rb'))

    # print "getting feature importances..."
    # df = pd.DataFrame({'f1_score':model.get_fscore().values()},
    #                   index=model.get_fscore().keys())
    #
    # df.to_csv('f1_score_dataframe.csv')


    print "loading vectorizer..."
    vect = pickle.load(open('vect2.p', 'rb'))

    # cvpath = 'twitter_cross_val.csv'
    testpath = '../Data/cv.p'
    dfcv = pd.read_pickle(testpath)
    Xcv = dfcv['text'].values
    ycv = dfcv['label'].values
    # import pdb; pdb.set_trace()
    print "transforming cross val data..."
    tfidf_Xcv = vect.transform(Xcv)
    tfidf_Xcvd = tfidf_Xcv.todense()

    print "predicting..."

    proba = model.predict_proba(tfidf_Xcvd)[:,1:].flatten()
    dfcv['boost_predict'] = proba
    dfcv.to_pickle('../Data/twitter_test_gboost_results3.p')





def predict_xgb():
    print "entering main..."
    modelpath = 'xgbfinal3_35.model'

    print "loading model..."
    model = xgb.Booster(model_file=modelpath)

    # print "getting feature importances..."
    # df = pd.DataFrame({'f1_score':model.get_fscore().values()},
    #                   index=model.get_fscore().keys())
    #
    # df.to_csv('f1_score_dataframe.csv')


    print "loading vectorizer..."
    vect = pickle.load(open('vect2.p', 'rb'))

    # cvpath = 'twitter_cross_val.csv'
    testpath = '../Data/cv.p'
    dfcv = pd.read_pickle(testpath)
    Xcv = dfcv['text'].values
    ycv = dfcv['label'].values
    import pdb; pdb.set_trace()
    print "transforming cross val data..."
    tfidf_Xcv = vect.transform(Xcv)
    tfidf_Xcvd = tfidf_Xcv.todense()

    xg_cv = xgb.DMatrix(tfidf_Xcvd, label=ycv)

    print "predicting..."
    proba = model.predict(xg_cv)

    dfcv['xgboost_predict'] = proba
    dfcv.to_pickle('../Data/twitter_test_xgboost_results4.p')

def predict_NB():
    print "entering main..."

    print "loading model..."
    with open('../Data/hate_modelNB.pickle', 'r') as f:
        clf = pickle.load(f)

    # print "getting feature importances..."
    # df = pd.DataFrame({'f1_score':model.get_fscore().values()},
    #                   index=model.get_fscore().keys())
    #
    # df.to_csv('f1_score_dataframe.csv')


    print "loading vectorizer..."
    vect = pickle.load(open('../Data/vect.p', 'rb'))

    # cvpath = 'twitter_cross_val.csv'
    testpath = '../Data/cv.p'
    dfcv = pd.read_pickle(testpath)
    Xcv = dfcv['text'].values
    ycv = dfcv['label'].values

    print "transforming cross val data..."
    tfidf_Xcv = vect.transform(Xcv)


    print "predicting..."
    proba = clf.predict_proba(tfidf_Xcv)[:,1:]

    dfcv['nb_predict'] = proba
    dfcv.to_pickle('../Data/twitter_test_NB_results.p')

def main():
    # predict_xgb()
    # predict_NB()
    predict_gb()

if __name__ == '__main__':
    '''This script collects feature importances and predicted probablities'''
    main()
