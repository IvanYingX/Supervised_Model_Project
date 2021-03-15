import pandas as pd
from Data.load_df import load_leagues
import numpy as np
import pickle


def translate_streak(df_results, n_match=3):
    '''
    Creates new columns based on the streak of each team.
    It creates n_match columns, where each column correponds
    to the result of the team in the previous matches
    Parameters
    ----------
    df_results : pandas Dataframe
        Dataframe with the results of the matches
    n_match: int
        Number of matches to take into account from the streak
    Returns
    -------
    df_results: pandas Dataframe
        Dataframe with the results of the matches
        with new columns for each team and its streak
    '''
    def get_nmatch(x):
        if (x is np.nan) or (x == '0'):
            string_streak = 'N' * n_match
        elif len(x) >= n_match:
            string_streak = x[-n_match:]
        else:
            string_streak = 'N' * (n_match - len(x)) + x

        return string_streak

    streak_columns = ['Total_Streak_Home', 'Total_Streak_Away',
                      'Streak_When_Home', 'Streak_When_Away']
    for streak in streak_columns:
        new_cols = df_results[streak].map(get_nmatch)
        for match in range(n_match):
            col_name = streak + '-' + str(n_match - match)
            match_col = new_cols.map(lambda x: x[match])
            new_streaks = pd.get_dummies(match_col,
                                         drop_first=True,
                                         prefix=col_name)
            df_results = pd.concat([df_results, new_streaks], axis=1)
    return df_results


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


def get_hour(x):
    '''
    Takes the second part of the date column, which corresponds to the hour
    If not available, it returns a standard hour, 17:00

    Parameters
    ----------
    x: str
        The result of the match
    Returns
    -------
    str:
        The hour of the match, if not available, returns 17:00
    '''
    if len(x.split(',')) > 2:
        return x.split(',')[2]
    else:
        return '17:00'


def standard_results(df_results):
    '''
    Standarize the following features:
        Position Home Team
        Position Away Team
        Goals For (Home, and Away)
        Goals Away (Home, and Away)
        Points_Home
        Points_Away
        Round
    Parameters
    ----------
    df : pandas Dataframe
        Dataframe with the info of the matches without normalizing
    Returns
    -------
    df_final : pandas Dataframe
        Updated Dataframe with standarized features
    '''
    list_position = ['Position_Home', 'Position_Away']
    list_goals = ['Total_Goals_For_Home_Team', 'Total_Goals_Against_Home_Team',
                  'Total_Goals_For_Away_Team', 'Total_Goals_Against_Away_Team',
                  'Goals_For_When_Home', 'Goals_Against_When_Home',
                  'Goals_For_When_Away', 'Goals_Against_When_Away']
    list_points = ['Points_Home', 'Points_Away']
    list_win_draw_lose = ['Total_Wins_Home', 'Total_Draw_Home',
                          'Total_Lose_Home', 'Total_Wins_Away',
                          'Total_Draw_Away', 'Total_Lose_Away',
                          'Wins_When_Home', 'Draw_When_Home',
                          'Lose_When_Home', 'Wins_When_Away',
                          'Draw_When_Away', 'Lose_When_Away']

    columns_to_encode = ['Daytime', 'Month']
    df_results[list_position] = df_results[list_position].divide(
                                            df_results['Number_Teams'], axis=0)
    df_results[list_points] = df_results[list_points].divide(
                                            df_results['Round'], axis=0)
    df_results[list_win_draw_lose] = df_results[list_win_draw_lose].divide(
                                            df_results['Round'], axis=0)
    df_results[list_goals] = df_results[list_goals].divide(
                                            df_results['Round'], axis=0)
    df_results['Round'] = df_results['Round'].divide(
                                            df_results['Total_Rounds'], axis=0)

    return df_results


def clean_data(res_dir='Data/Results_Cleaned/*'):
    '''
    Loads the datasets of all the available leagues, concatenates them,
    en cleans the data

    Parameters
    ----------
    res_dir: str
        Directory with the CSVs of the results
    Returns
    -------
    df_results: pandas DataFrame
        Dataframe with the result data of all the leagues
        concatenated and cleaned
    '''
    df_results = load_leagues(res_dir)
    filename = 'Data/Dictionaries/dict_match.pkl'
    with open(filename, 'rb') as f:
        dict_match = pickle.load(f)
    df_team = pd.read_csv('./Data/Team_Info.csv')
    # We start by cleaning the results dataframe. We also add some
    # features from the match dataframe
    df_team['Capacity'] = df_team['Capacity'].map(
        lambda x: int(x.replace(',', '')))

    # Clean the pitch column by changing the strings for categories
    pitch_list = ['Natural', 'grass', 'Césped', 'cesped natural', 'Grass',
                  'cesped real', 'NATURAL', 'Césped natural', 'natural grass',
                  'Natural grass', 'Césped Natural', 'Cesped natural',
                  'natural', 'Césped Artificial', 'AirFibr ', 'Artificial']
    pitch_dict = {x: 0 for x in pitch_list[:-3]}
    pitch_dict.update({x: 1 for x in pitch_list[-3:]})
    df_team['Pitch'] = df_team['Pitch'].map(pitch_dict)
    values_c = df_team['Pitch'].value_counts(normalize=True)
    missing = df_team['Pitch'].isnull()
    df_team.loc[missing, 'Pitch'] = np.random.choice(
                    values_c.index, size=len(df_team[missing]),
                    p=values_c.values)
    df_team['Pitch'] = df_team['Pitch'].astype('int')

    # Clean the date. The match dataframe has the right date so
    # we can merge them first, and substitute the data from the
    # match dataframe
    df_results['Time'] = df_results['Link'].map(
        dict_match).map(
            lambda x: x[0]).map(
                get_hour)

    # Then we can determine whether the match took place during
    # the morning, afternoon or evening
    df_results['Daytime'] = df_results['Time'].map(get_daytime)

    # We can see the date to see if the mo
    df_results['Date'] = pd.to_datetime(df_results['Link'].map(
                dict_match).map(lambda x: x[0]).map(
                        lambda x: x.split(',')[1]))
    df_results['Weekend'] = df_results['Date'].dt.dayofweek.map(
                        lambda x: 0 if x < 4 else 1)
    df_results['Month'] = df_results['Date'].dt.month
    df_results = df_results.merge(df_team, left_on='Home_Team',
                                  right_on='Team')

    # We don't want the first round of each season, because there
    # are no previous data for those matches
    df_results = df_results[df_results['Round'] != 1]

    # Let's get dummies for the month and the daytime columns
    month_columns = pd.get_dummies(df_results['Month'],
                                   prefix='Month',
                                   drop_first=True)
    df_results = pd.concat([df_results, month_columns], axis=1)
    daytime_columns = pd.get_dummies(df_results['Daytime'],
                                     prefix='Daytime',
                                     drop_first=True)
    df_results = pd.concat([df_results, daytime_columns], axis=1)
    # We need to somehow translate the streaks of each team, so we create a
    # set of dummies columns
    df_results = translate_streak(df_results)
    # Also, some numbers such as the current position, the number of points,
    # or the current round can be standarize for each season
    df_results = standard_results(df_results)
    columns_to_drop = ['Home_Team', 'Away_Team', 'Result',
                       'Link', 'Season', 'Goals_For_Home',
                       'Goals_For_Away', 'League', 'Team',
                       'City', 'Country', 'Stadium',
                       'Total_Rounds', 'Number_Teams',
                       'Total_Streak_Home', 'Total_Streak_Away',
                       'Streak_When_Home', 'Streak_When_Away',
                       'Month', 'Daytime', 'Date', 'Time']
    df_results = df_results.drop(columns_to_drop, axis=1)
    return df_results


if __name__ == '__main__':

    df_results = clean_data()
    # Export it as a csv for using in the Feature.py script
    df_results.to_csv('Results_Cleaned.csv', index=False)
