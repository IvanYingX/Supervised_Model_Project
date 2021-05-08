import pandas as pd
from collections import defaultdict
import glob
import itertools
import threading
import time
import sys
from Streak import process_streaks_numeric
from Streak import process_streaks_one_hot
from Day_Time import get_daytime

def norm_and_select(df):
    '''
    Normalize the following features:
        Position Home Team
        Position Away Team
        Goals For
        Goals Away
    And select the features to be used in the prediction
    Parameters
    ----------
    df : pandas Dataframe
        Dataframe with the info of the matches without normalizing
    Returns
    -------
    df_final : pandas Dataframe
        Updated Dataframe with normalized and selected features
    '''
    list_init_position = list(df.filter(regex='Position').columns)
    list_init_goals = list(df.filter(regex='Goal').columns)[2:]
    list_init_WDL = list(df.filter(regex=('Wins|Draw|Lose')))
    list_init_scores = list(df.filter(regex=('Score')).columns)

    list_final_position = [x + '_Norm' for x in list_init_position]
    list_final_goals = [x + '_Norm' for x in list_init_goals]
    list_final_WDL = [x + '_Norm' for x in list_init_WDL]
    list_final_scores = [x + '_Norm' for x in list_init_scores]

    list_to_drop = list_init_position + list_init_goals + \
        list_init_WDL + ['Home_Team', 'Away_Team', 'Result',
                         'Goals_For_Home', 'Goals_For_Away',
                         'Link', 'Number_Teams', 'Total_Rounds',
                         'Round', 'Points_Home', 'Points_Away']

    df[list_final_position] = df[list_init_position].divide(
                                            df['Number_Teams'], axis=0)
    df[list_final_goals] = df[list_init_goals].divide(
                                            df['Round'], axis=0)
    df[list_final_WDL] = df[list_init_WDL].divide(
                                            df['Round'], axis=0)
    df['Round_Norm'] = df['Round'].divide(
                                df['Total_Rounds'], axis=0)

    df = df.drop(list_to_drop, axis=1)

    return df


def animate():
    animation = [
                "[        ]",
                "[=       ]",
                "[===     ]",
                "[====    ]",
                "[=====   ]",
                "[======  ]",
                "[======= ]",
                "[========]",
                "[ =======]",
                "[  ======]",
                "[   =====]",
                "[    ====]",
                "[     ===]",
                "[      ==]",
                "[       =]",
                "[        ]",
                "[        ]"
                ]
    for c in itertools.cycle(animation):
        if done:
            break
        sys.stdout.write('\rloading ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rDone!                 ')


def create_features(results_dir, predict=True):
    '''
    Takes the clean dataset and creates features by applying
    feature engineering
    These new features include:
        Weekend: Whether the match took place during the weekend
        Daytime: Whether the match took place in the morning, during the
                 afternoon or during the evening
        Position_Home and Position_Away: Position of the respective team in the
                corresponding round
        Goals_For_Home and Goals_For_Away: Cumulated sum of the goals for
                of the respective team in the corresponding round
        Goals_Against_Home and Goals_Against_Away: Cumulated sum of the goals
                against of the respective team in the corresponding round
    Parameters
    ----------
    df_results : pandas Dataframe
        Dataframe with the info of the matches
    predict: bool
        If True, the code creates small dataframes with the upcoming matches so
        they are used for prediction
        if False, the code creates datasets with all the provided data
        with the added features, so it can be used for training
    '''
    # Check where to store the output
    if predict:
        output_file_predict = 'Data_to_Predict.csv'
        print('\rCreating data for prediction')
    else:
        output_file_predict = 'Data_For_Model.csv'
        print('\rCreating data for training')
    t = threading.Thread(target=animate)
    t.start()
    df_list = []
    for data_file in sorted(glob.glob(f'{results_dir}/*/*')):
        df_partial = pd.read_csv(data_file)
        df_list.append(df_partial)
    df_results = pd.concat(df_list, axis=0, ignore_index=True)
    # df_results = process_streaks_one_hot(df_results)
    df_results = process_streaks_numeric(df_results)
    df_selected = norm_and_select(df_results)

    # Save the dataset as a csv
    df_selected.to_csv(output_file_predict, index=False)


if __name__ == '__main__':
    results_dir = 'Data/Results_Cleaned'
    done = False
    create_features(results_dir, predict=False)
    done = True
