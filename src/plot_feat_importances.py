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

def myfunc(x,vocab):
    return str(vocab[int(x[1:])])

def main():
    modelpath = '../Data/xgbfinal4.model'
    print "loading model..."
    model = xgb.Booster(model_file=modelpath)
    vect = pickle.load(open('vect.p','rb'))
    vocab = vect.get_feature_names()
    print "getting feature importances..."
    df = pd.DataFrame({'f1_score':model.get_fscore().values()},
                      index=model.get_fscore().keys())

    df.to_csv('../Data/f1_score_dataframe.csv')

    print 'making home-grown feature importance chart'
    fscorepath = '../Data/f1_score_dataframe.csv'
    dfscore = pd.read_csv(fscorepath)
    dfscore['token'] = dfscore['Unnamed: 0'].map(lambda x: myfunc(x,vocab))
    dfscore.index = dfscore['token']
    dfscore = dfscore.sort('f1_score',ascending=False)
    dfscore[:20].plot(kind='barh',legend=False,)
    plt.xlabel("F1 Score")
    plt.ylabel("Features")
    plt.title("TF-IDF Feature Importances")
    plt.gca().invert_yaxis()
    plt.savefig('Feature_Importance_mine.jpg')

    # print 'making XGBoost\'s feature importances'
    # xgb.plot_importance(model)
    # plt.savefig('Feature_importance_xgb.jpg')

    # print 'making other plots'
    # xgb.plot_tree(model, num_trees=2)
    # plt.savefig('/images/plot_tree.jpg')
    # xgb.to_graphviz(model,num_trees=2)
    # plt.savefig('/images/plot_graphviz.jpg')


if __name__ == '__main__':
    main()
