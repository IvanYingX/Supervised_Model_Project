from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from .classifier import Classifier
from joblib import dump
import pandas as pd
import numpy as np
import os


def train_Linear_Discriminant(X, y, save_model=True):

    model = LinearDiscriminantAnalysis()
    search_grid = {'solver': ['svd', 'lsqr', 'eigen'],
                   'shrinkage': np.arange(0, 1, 0.1)}
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
    train_Linear_Discriminant(X, y)
