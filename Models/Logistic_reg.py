from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from joblib import dump
import pandas as pd

df = pd.read_csv('Data_For_Model.csv')
X = df.iloc[:, 1::].values
y = df.iloc[:, 0].values

crossvalidation = KFold(n_splits=5, shuffle=True, random_state=1)
LR_clf = LogisticRegression()
search_grid = {'penalty': ['l1', 'l2'],
               'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

clf = GridSearchCV(estimator=LR_clf,
                   param_grid=search_grid,
                   scoring='accuracy',
                   n_jobs=4,
                   cv=crossvalidation,
                   verbose=3)

clf.fit(X, y)
dump(clf, 'LR_model.joblib')
