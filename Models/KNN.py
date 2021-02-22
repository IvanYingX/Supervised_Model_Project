from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from joblib import dump
import pandas as pd

df = pd.read_csv('Data_For_Model.csv')
X = df.iloc[:, 1::].values
y = df.iloc[:, 0].values

crossvalidation = KFold(n_splits=5, shuffle=True, random_state=1)
KN_clf = KNeighborsClassifier()
search_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
clf = GridSearchCV(estimator=KN_clf,
                   param_grid=search_grid,
                   scoring='accuracy',
                   n_jobs=4,
                   cv=crossvalidation,
                   verbose=3)

clf.fit(X, y)
dump(clf, 'KN_model.joblib')
