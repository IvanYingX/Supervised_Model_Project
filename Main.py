import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GUI_models import train_predict
from Models import *
from Models import AdaBoost, Decision_Tree
from Models import GradientBoost, KNN, Linear_Discriminant
from Models import Logistic_reg, Naive_Bayes, SVC
from Models import Random_Forest, SGD, Quadratic_Discriminant
from os.path import dirname, basename, isfile, join
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


fun_dict = {
    'AdaBoost': AdaBoost.train_AdaBoost,
    'Decision_Tree': Decision_Tree.train_Decision_Tree,
    'GradientBoost': GradientBoost.train_GradientBoost,
    'KNN': KNN.train_KNN,
    'Linear_Discriminant':
        Linear_Discriminant.train_Linear_Discriminant,
    'Logistic_reg': Logistic_reg.train_Logistic_reg,
    'Naive_Bayes': Naive_Bayes.train_Naive_Bayes,
    'SVC': SVC.train_SVC,
    'Random_Forest': Random_Forest.train_Random_Forest,
    'SGD': SGD.train_SGD,
    'Quadratic_Discriminant':
        Quadratic_Discriminant.train_Quadratic_Discriminant
}

# Load the classification models
action = train_predict.train_or_predict()
if action:
    df = pd.read_csv('Data_For_Model.csv')
    X = df.iloc[:, 1::].values
    y = df.iloc[:, 0].values
    # Split the data into train and validation. The training set will be
    # used later in k-cross validation, so it remains as X, and y
    X_train, X_validation, y_train, y_validation = \
        train_test_split(X, y, test_size=0.3,
                         random_state=42)
    classifiers = train_predict.train()
    clf_dict = {}
    for clf in classifiers:
        clf_dict[clf] = fun_dict[clf](X_train, y_train)

# rf_clf = train_Random_Forest(X_train, y_train)
# print(rf_clf.accuracy(X_validation, y_validation))
# sgd_classifier = SGDClassifier(random_state=42)
# sgd_classifier.fit(X, y)
# y_pred = sgd_classifier.predict(X_validation)
# n_correct = sum(y_pred == y_validation)
# print(n_correct / len(y_pred))
# sgd_classifier = SGDClassifier(random_state=42)

# skfolds = StratifiedKFold(n_splits=10)
# for train_index, test_index in skfolds.split(X, y):
#     clone_classifier = clone(sgd_classifier)
#     X_train = X[train_index]
#     y_train = y[train_index]
#     X_test = X[test_index]
#     y_test = y[test_index]
#     # Use the cloned classifier to see the score in each strata
#     clone_classifier.fit(X_train, y_train)
#     y_pred = clone_classifier.predict(X_test)
#     n_correct = sum(y_pred == y_test)
#     print(n_correct / len(y_pred))

# print(cross_val_score(sgd_classifier, X_train, y_train,
#       cv = 10, scoring = 'accuracy'))

# cols = ['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1Score']
# df_metrics = pd.DataFrame(columns=cols)
# acc_dict = {}
# prec_dict = {}
# recall_dict = {}
# f1_dict = {}

# Fit each model with k splits. Thus, we split the data
# into (n_splits) buckets, and use (n_splits - 1) buckets to
# train, and 1 bucket to validate the model. Previously we already
# split the data into training and validation
# for clf in classifiers:
#     for train_index, test_index in skfolds.split(X, y):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         name = clf.__class__.__name__
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)
#         accu = accuracy_score(y_test, y_pred)
#         prec = precision_score(y_test, y_pred, average='weighted')
#         recall = recall_score(y_test, y_pred, average='weighted')
#         f1 = f1_score(y_test, y_pred, average='weighted')

#         if name in acc_dict:
#             acc_dict[name].append(accu)
#             prec_dict[name].append(prec)
#             recall_dict[name].append(recall)
#             f1_dict[name].append(f1)
#         else:
#             acc_dict[name] = [accu]
#             prec_dict[name] = [prec]
#             recall_dict[name] = [recall]
#             f1_dict[name] = [f1]

# for classifier in acc_dict:
#     acc_dict[classifier] = np.mean(acc_dict[classifier])
#     prec_dict[classifier] = np.mean(prec_dict[classifier])
#     recall_dict[classifier] = np.mean(recall_dict[classifier])
#     f1_dict[classifier] = np.mean(f1_dict[classifier])
#     new_entry = pd.DataFrame([[classifier, acc_dict[classifier],
#                                prec_dict[classifier],
#                                recall_dict[classifier],
#                                f1_dict[classifier]]],
#                              columns=cols)
#     df_metrics = df_metrics.append(new_entry, ignore_index=True)

# barWidth = 0.2
# r1 = np.arange(len(df_metrics.Classifier))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
# r4 = [x + barWidth for x in r3]

# fig, ax = plt.subplots(figsize=(15, 10))
# plt.bar(r1, df_metrics.Accuracy, width=0.2)
# plt.bar(r2, df_metrics.Precision, width=0.2)
# plt.bar(r3, df_metrics.Recall, width=0.2)
# plt.bar(r4, df_metrics.F1Score, width=0.2)
# plt.grid()
