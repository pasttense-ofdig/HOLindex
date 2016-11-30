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

def main():
    # resultspath2 = '../../data/gridsearch_modelbase2mini_on_test.csv'
    # df2 = pd.read_csv(resultspath2)
    xg_resultspath = '../Data/twitter_test_gboost_results3.p'
    xg_results = pd.read_pickle(xg_resultspath)
    nb_resultspath = '../Data/twitter_test_NB_results.p'
    nb_results = pd.read_pickle(nb_resultspath)
    xg_labels = xg_results['label'].values
    nb_labels = nb_results['label'].values
    TPR_xgb = []
    FPR_xgb = []
    TPR_NB = []
    FPR_NB = []


    for i in xrange(101):
        threshold = i/100.0
        predict = xg_results['boost_predict'].values >= threshold

        TP = sum(predict+xg_labels==2)
        TN = sum(predict+xg_labels==0)
        FP = sum(predict-xg_labels==1)
        FN = sum(predict-xg_labels==-1)

    #     print "accuracy: {} | threshold {}".format((TP+TN)/float(len(labels)),threshold)

        TPR_xgb.append(TP/float(TP+FN))
        FPR_xgb.append(FP/float(FP+TN))


    for i in xrange(101):
        threshold = i/100.0
        predict = nb_results['nb_predict'].values >= threshold

        TP = sum(predict+nb_labels==2)
        TN = sum(predict+nb_labels==0)
        FP = sum(predict-nb_labels==1)
        FN = sum(predict-nb_labels==-1)

    #     print "accuracy: {} | threshold {}".format((TP+TN)/float(len(labels)),threshold)

        TPR_NB.append(TP/float(TP+FN))
        FPR_NB.append(FP/float(FP+TN))

    plt.figure()

    # yvals = df2['recall']
    # xvals = df2['FP']/(df2['FP'] + df2['TN'])
    # plt.plot(xvals,yvals,'r',linewidth=2,label='doc2vec')

    plt.plot(FPR_xgb,TPR_xgb,'b',linewidth=2,label='GBoost')
    plt.plot(FPR_NB,TPR_NB,'r',linewidth=2,label='NaiveBayes')

    plt.xlabel('False Postive Rate')
    plt.xlim([0,1])
    plt.ylabel('True Positive Rate')
    plt.ylim([0,1])
    plt.legend()

    # doc2vecAUC = np.round(np.trapz(yvals[::-1],x=xvals[::-1]),decimals=2)
    XGB_AUC = np.round(np.trapz(TPR_xgb[::-1],x=FPR_xgb[::-1]),decimals=2)
    NB_AUC = np.round(np.trapz(TPR_NB[::-1],x=FPR_NB[::-1]),decimals=2)
    titlestr = "GBoost AUC: {} | NBayes AUC: {}".format(XGB_AUC, NB_AUC)
    plt.title(titlestr)
    plt.savefig('testROC_2.jpg')


if __name__ == '__main__':
    main()
