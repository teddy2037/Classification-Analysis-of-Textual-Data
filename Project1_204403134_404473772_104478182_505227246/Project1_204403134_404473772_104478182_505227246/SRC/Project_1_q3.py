import Project_1_q1_q2
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from string import punctuation
from sklearn.utils.extmath import randomized_svd

def LSI_(X_train_tfidf, X_test_tfidf):

    svd = TruncatedSVD(n_components=50, random_state=0)

    U, Sigma, VT = randomized_svd(X_train_tfidf, 
                              n_components=50,
                              n_iter=5,
                              random_state=None)

    # print U.shape, Sigma.shape, VT.shape
    X_train_r = svd.fit_transform(X_train_tfidf)
    X_test_r = svd.transform(X_test_tfidf)
    Err_LSI = 0
    A = np.dot(np.dot(U,np.diag(Sigma)),VT)
    Err_LSI = np.sum(np.array(X_train_tfidf - A)**2)
    return X_train_r, X_test_r, Err_LSI

def NMF_(X_train_tfidf, X_test_tfidf):

    model = NMF(n_components=50, init='random', random_state=0)

    W_train_r = model.fit_transform(X_train_tfidf)
    W_test_r = model.transform(X_test_tfidf)

    H = model.components_
    Err_NMF = 0
    Err_NMF = np.sum(np.array(X_train_tfidf - W_train_r.dot(H))**2)
    return W_train_r, W_test_r, H, Err_NMF

if __name__ == "__main__":
    X_train_tfidf, X_test_tfidf, A, B = Project_1_q1_q2.question2()
    y1, y2, y3 = LSI_(X_train_tfidf, X_test_tfidf)
    z1, z2, z3, z4 = NMF_(X_train_tfidf, X_test_tfidf)
    print("Error in LSI (squared): ", y3)
    print("Error in NMF (squared): ", z4)


