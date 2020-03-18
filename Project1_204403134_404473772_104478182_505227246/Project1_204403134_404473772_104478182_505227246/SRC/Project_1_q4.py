import Project_1_q1_q2
import Project_1_q3
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.svm import LinearSVC
from sklearn.decomposition import NMF
import itertools
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import LinearSVC

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


############################# Plot from DOC ############################
def plot_roc(fpr, tpr):
    fig, ax = plt.subplots()

    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, lw=2, label='area under curve = %0.4f' % roc_auc)

    ax.grid(color='0.7', linestyle='--', linewidth=1)

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=15)
    ax.set_ylabel('True Positive Rate', fontsize=15)

    ax.legend(loc="lower right")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(15)


############################################################################
def fit_predict_and_plot_roc1(sv_instance, test_data, test_label):
    # pipeline1.predict(twenty_test.data)

    if hasattr(sv_instance, 'decision_function'):
        prob_score = sv_instance.decision_function(test_data)
        fpr, tpr, _ = roc_curve(test_label, prob_score)
    else:
        prob_score = sv_instance.predict_proba(test_data)
        fpr, tpr, _ = roc_curve(test_label, prob_score[:, 1])

    # prob_score = sv_instance.decision_function(test_data)
    # fpr, tpr, _ = roc_curve(test_label, prob_score)

    plot_roc(fpr, tpr)


#     return pipe


#######################################  Confusion Matrix ##############################


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


##########################################################################################################


if __name__ == "__main__":
    ##########################   Export data from Project_1_q1_q2  ########################################

    X_train_tfidf, X_test_tfidf, Bin_Target_Train, Bin_Target_Test = Project_1_q1_q2.question2()

    X_train_r, X_test_r, Err_LSI = Project_1_q3.LSI_(X_train_tfidf, X_test_tfidf)
    #W_train_r, W_test_r, Err_NMF = Project_1_q3.LSI_(X_train_tfidf, X_test_tfidf)

    #########################  SVM #######################################################
    gamma1 = 1000
    gamma2 = 0.0001
    svc1 = LinearSVC(C=gamma1).fit(X_train_r, Bin_Target_Train)
    svc2 = LinearSVC(C=gamma2).fit(X_train_r, Bin_Target_Train)

    fit_predict_and_plot_roc1(svc1, X_test_r, Bin_Target_Test)
    plt.show()
    fit_predict_and_plot_roc1(svc2, X_test_r, Bin_Target_Test)
    plt.show()

    class_names = ['Computer Technology', 'Recreational Activity']
    Bin_Target_predict1 = svc1.predict(X_test_r)
    Bin_Target_predict2 = svc2.predict(X_test_r)



    ################ Compute confusion matrix for gamma = 1000 ##############
    cnf_matrix1 = confusion_matrix(Bin_Target_Test, Bin_Target_predict1)
    ################ Compute confusion matrix for gamma = 0.0001 ##############
    cnf_matrix2 = confusion_matrix(Bin_Target_Test, Bin_Target_predict2)

    np.set_printoptions(precision=2)

    print('-' * 35)

    print('SVM with Gamma = 0.0001')
    print('Accuracy: ', accuracy_score(Bin_Target_Test, Bin_Target_predict2))
    print('Precision: ', precision_score(Bin_Target_Test, Bin_Target_predict2))
    print('Recall: ', recall_score(Bin_Target_Test, Bin_Target_predict2))
    print('F1-score: ', f1_score(Bin_Target_Test, Bin_Target_predict2))




    print('SVM with Gamma = 1000')
    print('Accuracy: ', accuracy_score(Bin_Target_Test, Bin_Target_predict1))
    print('Precision: ', precision_score(Bin_Target_Test, Bin_Target_predict1))
    print('Recall: ', recall_score(Bin_Target_Test, Bin_Target_predict1))
    print('F1-score: ', f1_score(Bin_Target_Test, Bin_Target_predict1))




    print('-' * 35)
    # Plot non-normalized confusion matrix for gamma  = 1000
    plt.figure()
    plot_confusion_matrix(cnf_matrix1, classes=class_names,
                          title='Confusion matrix, without normalization for gamma = 1000')

    plt.figure()
    plot_confusion_matrix(cnf_matrix1, classes=class_names, normalize=True,
                          title='Normalized confusion matrix for gamma = 1000')

    # Plot non-normalized confusion matrix for gamma  = 0.001
    plt.figure()
    plot_confusion_matrix(cnf_matrix2, classes=class_names,
                          title='Confusion matrix, without normalization for gamma = 0.0001')

    plt.figure()
    plot_confusion_matrix(cnf_matrix2, classes=class_names, normalize=True,
                          title='Normalized confusion matrix for gamma = 0.0001')
    plt.show()

    ################################   Cross validation #######################################################

    print("Accuracy results")

    g = [0] * 8
    for i in range(8):
        g[i] = 10 ** (i - 4)
        print(g[i])

    for i in range(8):
        svc = LinearSVC(C=g[i]).fit(X_train_r, Bin_Target_Train)
        cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
        scores = cross_val_score(svc, X_train_r, Bin_Target_Train, cv=cv, scoring='accuracy')

        print(np.mean(scores))

    ################################   scores  ################################################################

    kfold = 5

    model1_0 = LinearSVC(C=0.0001)
    model_1 = LinearSVC(C=0.001)
    model_2 = LinearSVC(C=0.01)
    model_3 = LinearSVC(C=0.1)
    model_4 = LinearSVC(C=1)
    model_5 = LinearSVC(C=10)
    model_6 = LinearSVC(C=100)
    model_7 = LinearSVC(C=1000)

    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}
    results_m0 = {}
    results_m1 = {}
    results_m2 = {}
    results_m3 = {}
    results_m4 = {}
    results_m5 = {}
    results_m6 = {}
    results_m7 = {}

    for method in enumerate(scoring):
        results_m0[method[1]] = cross_val_score(estimator=model1_0,
                                                X=X_train_r,
                                                y=Bin_Target_Train,
                                                cv=cv,
                                                scoring=scoring[method[1]])

        results_m1[method[1]] = cross_val_score(estimator=model_1,
                                                X=X_train_r,
                                                y=Bin_Target_Train,
                                                cv=cv,
                                                scoring=scoring[method[1]])
        results_m2[method[1]] = cross_val_score(estimator=model_2,
                                                X=X_train_r,
                                                y=Bin_Target_Train,
                                                cv=cv,
                                                scoring=scoring[method[1]])
        results_m3[method[1]] = cross_val_score(estimator=model_3,
                                                X=X_train_r,
                                                y=Bin_Target_Train,
                                                cv=cv,
                                                scoring=scoring[method[1]])
        results_m4[method[1]] = cross_val_score(estimator=model_4,
                                                X=X_train_r,
                                                y=Bin_Target_Train,
                                                cv=cv,
                                                scoring=scoring[method[1]])
        results_m5[method[1]] = cross_val_score(estimator=model_5,
                                                X=X_train_r,
                                                y=Bin_Target_Train,
                                                cv=cv,
                                                scoring=scoring[method[1]])
        results_m6[method[1]] = cross_val_score(estimator=model_6,
                                                X=X_train_r,
                                                y=Bin_Target_Train,
                                                cv=cv,
                                                scoring=scoring[method[1]])
        results_m7[method[1]] = cross_val_score(estimator=model_7,
                                                X=X_train_r,
                                                y=Bin_Target_Train,
                                                cv=cv,
                                                scoring=scoring[method[1]])
    print('SVC with gamma= ', 0.0001, 'Results: ')
    print("results_accuracy:  ", np.mean(results_m0['accuracy']))
    print("results_precision: ", np.mean(results_m0['precision']))
    print("results_recall:    ", np.mean(results_m0['recall']))
    print("results_f1_score:  ", np.mean(results_m0['f1_score']))
    print('SVC with gamma= ', 0.001, 'Results: ')
    print("results_accuracy:  ", np.mean(results_m1['accuracy']))
    print("results_precision: ", np.mean(results_m1['precision']))
    print("results_recall:    ", np.mean(results_m1['recall']))
    print("results_f1_score:  ", np.mean(results_m1['f1_score']))
    print('SVC with gamma= ', 0.01, 'Results: ')
    print("results_accuracy:  ", np.mean(results_m2['accuracy']))
    print("results_precision: ", np.mean(results_m2['precision']))
    print("results_recall:    ", np.mean(results_m2['recall']))
    print("results_f1_score:  ", np.mean(results_m2['f1_score']))
    print('SVC with gamma= ', 0.1, 'Results: ')
    print("results_accuracy:  ", np.mean(results_m3['accuracy']))
    print("results_precision: ", np.mean(results_m3['precision']))
    print("results_recall:    ", np.mean(results_m3['recall']))
    print("results_f1_score:  ", np.mean(results_m3['f1_score']))
    print('SVC with gamma= ', 1, 'Results: ')
    print("results_accuracy:  ", np.mean(results_m4['accuracy']))
    print("results_precision: ", np.mean(results_m4['precision']))
    print("results_recall:    ", np.mean(results_m4['recall']))
    print("results_f1_score:  ", np.mean(results_m4['f1_score']))
    print('SVC with gamma= ', 10, 'Results: ')
    print("results_accuracy:  ", np.mean(results_m5['accuracy']))
    print("results_precision: ", np.mean(results_m5['precision']))
    print("results_recall:    ", np.mean(results_m5['recall']))
    print("results_f1_score:  ", np.mean(results_m5['f1_score']))
    print('SVC with gamma= ', 100, 'Results: ')
    print("results_accuracy:  ", np.mean(results_m6['accuracy']))
    print("results_precision: ", np.mean(results_m6['precision']))
    print("results_recall:    ", np.mean(results_m6['recall']))
    print("results_f1_score:  ", np.mean(results_m6['f1_score']))
    print('SVC with gamma= ', 1000, 'Results: ')
    print("results_accuracy:  ", np.mean(results_m7['accuracy']))
    print("results_precision: ", np.mean(results_m7['precision']))
    print("results_recall:    ", np.mean(results_m7['recall']))
    print("results_f1_score:  ", np.mean(results_m7['f1_score']))



    svc3 = LinearSVC(C=10).fit(X_train_r, Bin_Target_Train)

    fit_predict_and_plot_roc1(svc3, X_test_r, Bin_Target_Test)
    plt.show()

    Bin_Target_predict3 = svc3.predict(X_test_r)
    cnf_matrix3 = confusion_matrix(Bin_Target_Test, Bin_Target_predict3)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix for gamma  = 10
    plt.figure()
    plot_confusion_matrix(cnf_matrix3, classes=class_names,
                          title='Confusion matrix, without normalization for gamma = 10')

    plt.figure()
    plot_confusion_matrix(cnf_matrix3, classes=class_names, normalize=True,
                          title='Normalized confusion matrix for gamma = 10')

    plt.show()




