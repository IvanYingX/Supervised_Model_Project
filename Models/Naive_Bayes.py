from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from joblib import dump
import pandas as pd
import numpy as np
import os


def train_NB(X, y):

    crossvalidation = KFold(n_splits=5, shuffle=True, random_state=42)
    NB_clf = GaussianNB()
    search_grid = {'var_smoothing': np.logspace(0, -9, num=100)}

    clf = GridSearchCV(estimator=NB_clf,
                       param_grid=search_grid,
                       scoring='accuracy',
                       n_jobs=4,
                       cv=crossvalidation,
                       verbose=1)

    clf.fit(X, y)
    filename = ('Models/'
                + os.path.basename(
                    __file__).replace(
                    '.py', '_model.joblib')
                )
    dump(clf, filename)
    return clf
