import pandas as pd
import glob
import threading
import itertools
import time
import sys
from collections import defaultdict
from Streak import process_streaks_numeric
from Streak import process_streaks_one_hot
from Day_Time import get_daytime
from Normalize import norm_and_select

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
    

    # Check where to store the output
    if predict:
        output_file_predict = 'Data_to_Predict.csv'
        print('\rCreating data for prediction')
    else:
        output_file_predict = 'Data_For_Model.csv'
        print('\rCreating data for training')
    done = False
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
    done = True


if __name__ == '__main__':
    results_dir = 'Data/Results_Cleaned'
    create_features(results_dir, predict=False)
