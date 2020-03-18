from Project_1_q1_q2 import question2 as getData
from Project_1_q3 import LSI_
from Project_1_q4 import fit_predict_and_plot_roc1, plot_confusion_matrix
from Project_1_q5 import print_metrics

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, f1_score
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB

def Naive_Bae(X_train_r, y_train, X_test_r, y_test):
    clf = GaussianNB().fit(X_train_r, y_train)
    fit_predict_and_plot_roc1(clf, X_test_r, y_test)
    cnf_matrix = confusion_matrix(y_test, clf.predict(X_test_r))
    plt.figure()
    class_names = ['Computer Technology', 'Recreational Activity']
    plot_confusion_matrix(cnf_matrix, classes=class_names,  normalize=True, title='Bayes Classifier Confusion matrix')
    plt.show()
    print_metrics(y_test, clf.predict(X_test_r))


if __name__ == "__main__":
    # Using similar framework as Q5
    X_train_tfidf, X_test_tfidf, target_Train, target_Test = getData()
    y1, y2, _ = LSI_(X_train_tfidf, X_test_tfidf)

    print("Naive Bayesian Classifier:")
    Naive_Bae(y1, target_Train, y2, target_Test)
    print("Bae is done")
