import numpy as np
import random
import matplotlib.pyplot as plt
from Project_1_q3 import LSI_
from Project_1_q1_q2 import vectorize_data
from sklearn.svm import LinearSVC
from Project_1_q4 import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from Project_1_q5 import print_metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV


def question8_multi(opt):
    class_names = ["comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware","misc.forsale", "soc.religion.christian"]
    X_train_tfidf, X_test_tfidf, Target_Train, Target_Test = vectorize_data(class_names, binary = False)
    X_train_r, X_test_r, _ = LSI_(X_train_tfidf, X_test_tfidf)
    parameters = {'estimator__C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    if opt == 1:
        ovr = OneVsRestClassifier(LinearSVC(class_weight = "balanced"))
        clf = GridSearchCV(ovr, parameters, cv=5).fit(X_train_r, Target_Train)
        title = 'One v All Confusion matrix'

    if opt == 2:
        ovo = OneVsOneClassifier(LinearSVC())
        clf = GridSearchCV(ovo, parameters, cv=5).fit(X_train_r, Target_Train)
        title='One v One Confusion matrix'

    if opt == 3:
        clf = GaussianNB().fit(X_train_r, Target_Train)
        title = 'Bayes Classifier One_v_All Confusion matrix'
    
    cnf_matrix = confusion_matrix(Target_Test, clf.predict(X_test_r))
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,  normalize=True, title=title)
    plt.show()
    print_metrics(Target_Test, clf.predict(X_test_r), "macro")


if __name__ == "__main__":
    question8_multi(1)
    question8_multi(2)
    question8_multi(3)






