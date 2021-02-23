from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from .classifier import Classifier
from joblib import dump
import pandas as pd
import numpy as np
import os


def train_Logistic_reg(X, y, save_model=True):

    model = LogisticRegression()
    search_grid = {'penalty': ['l1', 'l2'],
                   'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    clf = Classifier(model, search_grid)
    clf.train(X, y)
    if save_model:
        filename = ('Models/'
                    + os.path.basename(
                     __file__).replace(
                     '.py', '_model.joblib')
                    )
        dump(clf, filename)
    return clf


if __name__ == '__main__':

    df = pd.read_csv('Data_For_Model.csv')
    X = df.iloc[:, 1::].values
    y = df.iloc[:, 0].values
    train_Logistic_reg(X, y)
