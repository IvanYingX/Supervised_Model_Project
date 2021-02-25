from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from .classifier import Classifier
from joblib import dump
import pandas as pd
import numpy as np
import os


def train_Quadratic_Discriminant(X, y, save_model=True):

    model = QuadraticDiscriminantAnalysis()
    search_grid = {'reg_param': np.arange(0.1, 0.5, 0.05)}
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
    train_Quadratic_Discriminant(X, y)
