from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from .classifier import Classifier
from joblib import dump
import pandas as pd
import numpy as np
import os


def train_GradientBoost(X, y, save_model=True):

    model = GradientBoostingClassifier()
    search_grid = {'n_estimators': range(20, 101, 40),
                   'max_depth': range(5, 16, 5),
                   'min_samples_split': range(200, 1001, 400),
                   'learning_rate': [.001, 0.01, 0.1]}
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
    train_GradientBoost(X, y)
