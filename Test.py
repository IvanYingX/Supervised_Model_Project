from Models.Naive_Bayes import train_NB
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
df = pd.read_csv('Data_For_Model.csv')
X = df.iloc[:, 1::].values
y = df.iloc[:, 0].values
# Split the data into train and validation. The training set will be
# used later in k-cross validation, so it remains as X, and y
X, X_validation, y, y_validation = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)
train_NB(X, y)
clf = load('Models/Naive_Bayes_model.joblib')
y_pred = clf.predict(X_validation)
print(sum(y_pred == y_validation) / len(y_pred))
