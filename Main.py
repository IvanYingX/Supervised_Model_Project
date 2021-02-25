import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Feature import *
from GUI_models import train_predict
from Models import *
from Models import AdaBoost, Decision_Tree
from Models import GradientBoost, KNN, Linear_Discriminant
from Models import Logistic_reg, Naive_Bayes, SVC
from Models import Random_Forest, SGD, Quadratic_Discriminant
from joblib import load
from os.path import dirname, basename, isfile, join
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import tkinter as tk
import pandas as pd
from pandastable import Table, TableModel


class DataFrameTable(tk.Frame):
    def __init__(self, parent=None, df=pd.DataFrame()):
        super().__init__()
        self.parent = parent
        self.pack(fill=tk.BOTH, expand=True)
        self.table = Table(
            self, dataframe=df,
            showtoolbar=False,
            showstatusbar=True,
            editable=False)
        self.table.show()


def _quit():
    root.quit()
    root.destroy()


def _stop():
    """Stop scanning by setting the global flag to False."""
    global running
    root.quit()
    root.destroy()
    running = False


fun_dict = {
    'AdaBoost': AdaBoost.train_AdaBoost,
    'Decision_Tree': Decision_Tree.train_Decision_Tree,
    'GradientBoost': GradientBoost.train_GradientBoost,
    'KNN': KNN.train_KNN,
    'Linear_Discriminant':
        Linear_Discriminant.train_Linear_Discriminant,
    'Logistic_reg': Logistic_reg.train_Logistic_reg,
    'Naive_Bayes': Naive_Bayes.train_Naive_Bayes,
    'SVC': SVC.train_SVC,
    'Random_Forest': Random_Forest.train_Random_Forest,
    'SGD': SGD.train_SGD,
    'Quadratic_Discriminant':
        Quadratic_Discriminant.train_Quadratic_Discriminant
}


df = pd.read_csv('Data_For_Model.csv')
X = df.iloc[:, 1::].values
y = df.iloc[:, 0].values
# Split the data into train and validation. The training set will be
# used later in k-cross validation, so it remains as X, and y
X_train, X_validation, y_train, y_validation = \
    train_test_split(X, y, test_size=0.3,
                     random_state=42)

running = True

while running:

    action = train_predict.train_predict_check()

    # If action is 1, that means train has been pressed
    if action == 1:
        # Load the classification models
        classifiers = train_predict.train()
        clf_dict = {}
        for clf in classifiers:
            clf_dict[clf] = fun_dict[clf](X_train, y_train)

        cols = ['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1Score']
        df_metrics = pd.DataFrame(columns=cols)
        clfs = list(fun_dict.keys())

        for clf in clfs:

            model = load(f'Models/{clf}_model.joblib')
            y_pred = model.predict(X_validation)
            accu = accuracy_score(y_validation, y_pred)
            prec = precision_score(y_validation, y_pred, average='weighted')
            recall = recall_score(y_validation, y_pred, average='weighted')
            f1 = f1_score(y_validation, y_pred, average='weighted')
            new_entry = pd.DataFrame([[clf, accu, prec, recall, f1]],
                                     columns=cols)
            df_metrics = df_metrics.append(new_entry, ignore_index=True)

        df_metrics.to_csv('Models/Metrics.csv', index=False)

    # If action is 2, that means predict has been pressed
    elif action == 2:
        # Get information for the next round
        # Ask what league the user wants to predict from:
        next_match_list, df_to_show = train_predict.get_next_matches()

        # Ask what model the user wants to use for prediction:
        clf = train_predict.choose_model()[0]
        clf = load(f'Models/{clf}_model.joblib')
        leagues = []
        last_season = []
        for match in next_match_list:
            # Take only the leagues and years of the desired predictions
            leagues.append(match['League'].unique()[0])
            last_season.append(match['Season'].unique()[0])
        season_league = zip(last_season, leagues)
        df_sta_whole = pd.read_csv('Standings_Cleaned.csv')
        df_sta_list = []

        for season, league in season_league:
            df_sta_list.append(
                df_sta_whole[
                    (df_sta_whole['Season'] == season)
                    & (df_sta_whole['League'] == league)])
        df_sta = pd.concat(df_sta_list)

        # Let's concat the match list in a single Dataframe
        df_pred = pd.concat(next_match_list)
        df_pred[['Home_Goals', 'Away_Goals', 'Label']] = None
        create_features(df_pred, df_sta, predict=True)
        df_to_predict = pd.read_csv('Data_to_Predict.csv')
        predictions = clf.predict(df_to_predict)
        


    # If action is 3, that means check has been pressed
    elif action == 3:

        df_metrics = pd.read_csv('Models/Metrics.csv')
        root = tk.Tk()
        table = DataFrameTable(root, df_metrics)
        button_back = tk.Button(root, text='Go Back', command=_quit)
        button_back.pack(side=tk.BOTTOM)
        button = tk.Button(master=root, text="Quit", command=_stop)
        button.pack(side=tk.BOTTOM)
        root.mainloop()
