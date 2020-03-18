from Project_1_q1_q2 import question2 as getData
from Project_1_q3 import LSI_
from Project_1_q4 import fit_predict_and_plot_roc1, plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, f1_score
from sklearn.model_selection import cross_val_score


def print_metrics(y_true, y_predict, options= "binary" ):
    print('Accuracy: ',accuracy_score(y_true, y_predict))
    print('Precision: ',precision_score(y_true, y_predict,average = options))
    print('Recall: ',recall_score(y_true, y_predict,average = options))
    print('F1-score: ', f1_score(y_true, y_predict,average = options))



def logistic_classifier(X, y, X_test, y_test, reg = None):
    if (reg == None):
        c = 1e30 # Approximate no regularization with large C
        clf = LogisticRegression(C=c, solver='liblinear').fit(X,y)
        # Plot ROC and confusion matrix
        fit_predict_and_plot_roc1(clf, X_test, y_test)
        cnf_matrix = confusion_matrix(y_test, clf.predict(X_test))
        plt.figure()
        class_names = ['Computer Technology', 'Recreational Activity']
        plot_confusion_matrix(cnf_matrix, classes=class_names,  normalize=True,
                                  title='Logistic Regression Confusion matrix')
        plt.show()
    else:
        c_values = 10.0 ** np.array(range(-3,4)) # 10^-3 to 10^3
        clf = LogisticRegressionCV(Cs=c_values, penalty=reg, cv=5, solver='liblinear').fit(X, y)
        print('Best C: ', clf.C_)
        # Plot effect of C on coefficients
        plt.figure()
        coefs = clf.coefs_paths_[1]
        plt.semilogx(c_values,  np.mean(abs(coefs), axis=(0,2)), label=reg)
        plt.title('Average Coefficient Magnitude')
        plt.legend()
        plt.show()
    print_metrics(y_test, clf.predict(X_test))



if __name__ == "__main__":
    X_train_tfidf, X_test_tfidf, target_Train, target_Test = getData()
    y1, y2, _ = LSI_(X_train_tfidf, X_test_tfidf)
    print("No Regularization")
    logistic_classifier(y1, target_Train, y2, target_Test)
    print("\nL1 Regularization")
    logistic_classifier(y1, target_Train, y2, target_Test, 'l1')
    print("\nL2 Regularization")
    logistic_classifier(y1, target_Train, y2, target_Test, 'l2')










