from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from joblib import dump

import pandas as pd

df = pd.read_csv('Data_For_Model.csv')
X = df.iloc[:, 1::].values
y = df.iloc[:, 0].values
crossvalidation = KFold(n_splits=5, shuffle=True, random_state=1)
DTC = DecisionTreeClassifier()
search_grid = {'criterion': ['gini', 'entropy'],
               'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20]}

clf = GridSearchCV(estimator=DTC,
                   param_grid=search_grid,
                   scoring='accuracy',
                   n_jobs=4,
                   cv=crossvalidation,
                   verbose=3)

clf.fit(X, y)
dump(clf, 'Decision_Tree_model.joblib')