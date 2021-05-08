import pandas as pd
from collections import defaultdict
import glob
import itertools
import threading
import time
import sys
from sklearn.preprocessing import OneHotEncoder


def get_daytime(x):
    '''
    Returns an integer which represents
    the time of the day the match took place
    0: Morning
    1: Afternoon
    2: Evening
    The function eventually will take the sunrise
    and sunset time to change the outcome.
    So far it assumes that the sunset is at 18:00 and the
    sunrise is at 12:00

    Parameters
    ----------
    x : str
        The time of the match in 24h format (17:00)

    Returns
    -------
    int
        An integer representing the time of the day
    '''
    hour = int(x.split(':')[0])
    if (hour >= 18) or (hour == 0):
        return 2
    if hour >= 12:
        return 1
    else:
        return 0


def process_streaks_one_hot(df):
    '''
    Creates new columns corresponding to a One Hot
    Encoding of the last matches of each team
    Parameters
    ----------
    df : pandas DataFrame
        Dataframe with the results and performance for 
        each team
    Returns
    -------
    df_ohe: pandas DataFrame
        Dataframe updated with the new columns corresponding
        to the One Hot Encoding
    '''
    streak_list = list(df.filter(regex='Streak').columns)
    df[streak_list] = df[streak_list].fillna('N')
    df[streak_list] =  df[streak_list].applymap(lambda x: x[-3:])
    enc = OneHotEncoder(drop='first')
    df_streakless = df[df.columns.drop(list(df.filter(regex='Streak')))]
    df_streaks = df.filter(regex='Streak')
    # Convert the streaks to OHE array
    streaks_ohe_array = enc.fit_transform(df_streaks).toarray()
    # Give a prefix to each new columns according to the previous name
    new_columns = enc.get_feature_names(
        list(df.filter(regex='Streak').columns))
    df_streaks_ohe = pd.DataFrame(streaks_ohe_array,
                                  columns=new_columns)
    df_ohe = pd.concat([df_streakless, df_streaks_ohe], axis=1)
    return df_ohe

# Create a dictionary with the corresponding value to each
# character
streak_dict = {'N': 0, 'W': 1, 'D': -0.3, 'L': -1}
# Create a list with the corresponding weight of each match
# e.g. last match has a weight of 1, the previous one has a
# weight of 0.5
streak_list = [1, 0.5, 0.3]

def streak_2_num(x):
    '''
    Returns a numeric value for each passed row
    It reads the last three values, and transforms
    each character into a number according to
    streak_dict and streak_list
    Parameters
    ----------
    x : str
        Last three characters of the streak column
    Returns
    -------
    float
        Numeric value corresponding to the 
        streak
    '''
    val = 0
    chars = x[::-1][:len(x)]
    for char, match in zip(chars, streak_list[:len(x)]):
        val += streak_dict[char] * match
    return val

def process_streaks_numeric(df):
    '''
    Changes the streak values for numeric values to be
    processed by the model
    Parameters
    ----------
    df : pandas DataFrame
        Dataframe with the results and performance for 
        each team
    Returns
    -------
    df: pandas DataFrame
        Dataframe updated with the streak column as a numeric
        value
    '''
    streak_list = list(df.filter(regex='Streak').columns)
    df[streak_list] = df[streak_list].fillna('N')
    df[streak_list] = df[streak_list].replace(to_replace='0',
                                              value='N')
    df[streak_list] =  df[streak_list].applymap(lambda x: x[-3:])
    df[streak_list] =  df[streak_list].applymap(streak_2_num)
    return df


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
