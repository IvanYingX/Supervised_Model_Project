
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from joblib import dump
import pandas as pd
import numpy as np
import os


class Classifier:

    def __init__(self, model, search_grid, n_splits=5,
                 scoring='accuracy', shuffle=True,
                 random_state=42, verbose=3):

        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle,
                                  random_state=random_state)
        self.clf = GridSearchCV(estimator=model,
                                param_grid=search_grid,
                                scoring=scoring,
                                n_jobs=4,
                                cv=self.cv,
                                verbose=verbose)
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None

    def train(self, X, y):
        self.clf.fit(X, y)
        self.best_params_ = self.clf.best_params_
        self.best_score_ = self.clf.best_score_
        self.best_estimator_ = self.clf.best_estimator_

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        return sum(y_pred == y_true)/len(y_pred)
