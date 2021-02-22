from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from joblib import dump

import pandas as pd

df = pd.read_csv('Data_For_Model.csv')
X = df.iloc[:, 1::].values
y = df.iloc[:, 0].values
crossvalidation = KFold(n_splits=5, shuffle=True, random_state=1)
GBC = GradientBoostingClassifier()
search_grid = {'n_estimators': range(20, 101, 40),
               'max_depth': range(5, 16, 5),
               'min_samples_split': range(200, 1001, 400),
               'learning_rate': [.001, 0.01, 0.1]}

clf = GridSearchCV(estimator=GBC,
                   param_grid=search_grid,
                   scoring='accuracy',
                   n_jobs=4,
                   cv=crossvalidation,
                   verbose=3)

clf.fit(X, y)
dump(clf, 'Gradient_Boost_model.joblib')
