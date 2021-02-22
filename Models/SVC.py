from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from joblib import dump
import pandas as pd

df = pd.read_csv('Data_For_Model.csv')
X = df.iloc[:, 1::].values
y = df.iloc[:, 0].values

crossvalidation = KFold(n_splits=5, shuffle=True, random_state=42)
SVC_clf = SVC()
search_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                'C': [1, 10, 100, 1000]},
               {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

clf = GridSearchCV(estimator=SVC_clf,
                   param_grid=search_grid,
                   scoring='accuracy',
                   n_jobs=4,
                   cv=crossvalidation,
                   verbose=3)

clf.fit(X, y)
dump(clf, 'SVC_model.joblib')
