from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from joblib import dump
import pandas as pd

df = pd.read_csv('Data_For_Model.csv')
X = df.iloc[:, 1::].values
y = df.iloc[:, 0].values
crossvalidation = KFold(n_splits=5, shuffle=True, random_state=42)
ada = AdaBoostClassifier()
search_grid = {'n_estimators': [1000, 1500, 2000],
               'learning_rate': [.001, 0.01, 0.1]}

clf = GridSearchCV(estimator=ada,
                   param_grid=search_grid,
                   scoring='accuracy',
                   n_jobs=4,
                   cv=crossvalidation,
                   verbose=3)

clf.fit(X, y)
dump(clf, 'Ada_Boost_model.joblib')
