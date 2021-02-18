import sklearn
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators = 100),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]

cols = ['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1Score']
log = pd.DataFrame(columns = cols)
acc_dict = {}
prec_dict = {}
rec_dict = {}
f1_dict = {}

skfolds = StratifiedKFold(n_splits = 10)

for train_index, test_index in skfolds.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accu = accuracy_score(y_test,y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        if name in acc_dict:
            acc_dict[name] += accu
            prec_dict[name] += prec
            rec_dict[name] += rec
            f1_dict[name] += f1
        else:
            acc_dict[name] = accu
            prec_dict[name] = prec
            rec_dict[name] = rec
            f1_dict[name] = f1
            
for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf]/10
    prec_dict[clf] = prec_dict[clf]/10
    rec_dict[clf] = rec_dict[clf]/10
    f1_dict[clf] = f1_dict[clf]/10
    new_entry = pd.DataFrame([[clf, acc_dict[clf], prec_dict[clf], rec_dict[clf], f1_dict[clf]]], columns = cols)
    log = log.append(new_entry, ignore_index = True)
    
barWidth = 0.2
r1 = np.arange(len(log.Classifier))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

fig, ax = plt.subplots(figsize = (15,10))
plt.bar(r1, log.Accuracy, width = 0.2)
plt.bar(r2, log.Precision, width = 0.2)
plt.bar(r3, log.Recall, width = 0.2)
plt.bar(r4, log.F1Score, width = 0.2)
plt.grid()
# Load the data
