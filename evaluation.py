from sklearn.metrics import balanced_accuracy_score,plot_confusion_matrix, accuracy_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""
This function Evaluates the performance of the model.
:param: svm_model represents the support vector machine model
:param: log_model represents the logistic regression model 
:param X_valid represents the validation X data
:param y_valid represents the validation target which will be used to access accuracy
:param figname represents the name of the figure.

return table_report represents the evaluation report.

"""


def model_evaluation (svm_model, log_model, knn, X_valid, y_valid,figname):
    # get the predicted values for each of the model
    y_pred_svm = svm_model.predict(X_valid)
    y_pred_log = log_model.predict(X_valid)
    y_pred_knn = knn.predict(X_valid)
    # get the balanced accuracy and accuracy score
    balanced_svm, accuracy_svm = balanced_accuracy_score(y_valid, y_pred_svm),  accuracy_score(y_valid, y_pred_svm)
    balanced_log, accuracy_log = balanced_accuracy_score(y_valid, y_pred_log), accuracy_score(y_valid, y_pred_log)
    balanced_knn, accuracy_knn = balanced_accuracy_score(y_valid, y_pred_knn), accuracy_score(y_valid, y_pred_knn)

    # get the f1 score
    f1_svm = f1_score(y_valid, y_pred_svm, average='micro')
    f1_log = f1_score(y_valid, y_pred_log, average='micro')
    f1_knn = f1_score(y_valid, y_pred_knn, average='micro')

    # store the result in a list
    svm_scores = [balanced_svm, accuracy_svm, f1_svm]
    log_scores = [balanced_log, accuracy_log, f1_log]
    knn_scores = [balanced_knn, accuracy_knn, f1_knn]
    table_report = pd.DataFrame(data=[svm_scores, log_scores, knn_scores],
                                columns=['balanced_accuracy', 'accuracy','f1_score'],
                                index=['svm_model', 'logistic_model', 'knn_model'])
    return table_report
