from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from .classifier import Classifier
from joblib import dump
import pandas as pd
import numpy as np
import os


def train_Decision_Tree(X, y, save_model=True):

    model = DecisionTreeClassifier()
    search_grid = {'criterion': ['gini', 'entropy'],
                   'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20]}
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
    train_Decision_Tree(X, y)
