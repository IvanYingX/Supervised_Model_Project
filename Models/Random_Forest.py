from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from joblib import dump
import pandas as pd

df = pd.read_csv('Data_For_Model.csv')
X = df.iloc[:, 1::].values
y = df.iloc[:, 0].values
crossvalidation = KFold(n_splits=5, shuffle=True, random_state=42)
RFC = RandomForestClassifier()
search_grid = {
    'n_estimators': [200, 300, 400, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}

clf = GridSearchCV(estimator=RFC,
                   param_grid=search_grid,
                   scoring='accuracy',
                   n_jobs=4,
                   cv=crossvalidation,
                   verbose=3)

clf.fit(X, y)
dump(clf, 'Random_Forest_model.joblib')
