import pandas as pd


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


def generate_keys(df_res, df_sta):
    '''
    Returns the dataframes with added common keys, so they can be
    merged together to extract more features
    Parameters
    ----------
    df_res : pandas Dataframe
        Dataframe with the results and info of the matches
    df_sta : pandas Dataframe
        Dataframe with the standings positions
    Returns
    -------
    df_res : pandas Dataframe
        Updated Dataframe with new keys
    df_sta : pandas Dataframe
        Updated Dataframe with new keys
        An integer representing the time of the day
    '''

    # The Round + 1 is to take into account that the standings dataframe is
    # constructed at the end of each round
    df_sta['Key'] = df_sta.apply(
                            lambda x: f"{x['Team']}-{x['Round'] + 1}"
                            + "-{x['Season']}-{x['League']}",
                            axis=1)
    df_sta['Key_Season_League'] = df_sta.apply(
                            lambda x: f"{x['Season']}-{x['League']}",
                            axis=1)

    df_res['Key_Home'] = df_res.apply(
                            lambda x: f"{x['Home_Team']}-{x['Round']}"
                            + f"-{x['Season']}-{x['League']}",
                            axis=1)
    df_res['Key_Away'] = df_res.apply(
                            lambda x: f"{x['Away_Team']}-{x['Round']}"
                            + f"-{x['Season']}-{x['League']}",
                            axis=1)
    df_res['Key_Season_League'] = df_res.apply(
                            lambda x: f"{x['Season']}-{x['League']}",
                            axis=1)

    matches = df_res.loc[
                :, ['Key_Season_League', 'Round']
                ].groupby(by=['Key_Season_League']).max()
    teams = df_res.loc[
                :, ['Key_Season_League', 'Home_Team']
                ].groupby(by=['Key_Season_League']).nunique()

    # In the sites where webscraping was done, some seasons are incomplet or
    # disorganised. Thus, some seasons in the site only included 1 round. We
    # can get rid of those seasons checking if the number of matches is equal
    # to the number of teams times 2 minus 2.
    mask = matches.values == (2*(teams-1)).values
    accepted_keys = matches[mask].index
    df_res = df_res[df_res['Key_Season_League'].isin(accepted_keys)]
    df_sta = df_sta[df_sta['Key_Season_League'].isin(accepted_keys)]

    # We can add a new variable that contains the number of matches in a
    # certain season and league
    dict_matches = matches.to_dict()['Round']
    dict_teams = teams.to_dict()['Home_Team']
    df_res['Number_Rounds'] = df_res['Key_Season_League'].map(dict_matches)
    df_res['Number_Teams'] = df_res['Key_Season_League'].map(dict_teams)

    return df_res, df_sta


def generate_streaks(df_res):
    '''
    '''
    dict_label = {0: 1, 1: 0, 2: -1}
    df_res['Win_Home'] = df_res['Label'].map(dict_label)
    df_res['Win_Away'] = df_res['Label'].map(dict_label) * (-1)
    df_res['Home_Streak_Total'] = None
    df_res['Away_Streak_Total'] = None
    df_res['Home_Streak'] = None
    df_res['Away_Streak'] = None
    for team in set(df_res['Home_Team']):
        df_res_home = df_res[(df_res['Home_Team'] == team)]
        df_res_away = df_res[(df_res['Away_Team'] == team)]
        df_res_team = df_res[(df_res['Home_Team'] == team)
                             | (df_res['Away_Team'] == team)]
        for league in df_res_team['League'].unique():
            for year in df_res_team['Season'].unique():
                subset = df_res_team[
                            (df_res_team['Season'] == year)
                            & (df_res_team['League'] == league)
                            ].sort_values(by=['Round'])[
                            ['Home_Team', 'Away_Team',
                             'Win_Home', 'Win_Away']]
                home_subset = df_res_home[
                            (df_res_home['Season'] == year)
                            & (df_res_home['League'] == league)
                            ].sort_values(by=['Round'])['Win_Home']
                away_subset = df_res_away[
                            (df_res_away['Season'] == year)
                            & (df_res_away['League'] == league)
                            ].sort_values(by=['Round'])['Win_Away']

                streak = (
                    (subset["Home_Team"] == team)
                    .multiply(subset["Win_Home"])
                    + (subset["Away_Team"] == team)
                    .multiply(subset["Win_Away"])
                    )
                streak_value = (
                    streak
                    + 0.5 * streak.shift(1, fill_value=0)
                    + 0.2 * streak.shift(2, fill_value=0)
                    )
                streak_home_value = (
                    home_subset
                    + 0.5 * home_subset.shift(1, fill_value=0)
                    + 0.2 * home_subset.shift(2, fill_value=0)
                    )
                streak_away_value = (
                    away_subset
                    + 0.5 * away_subset.shift(1, fill_value=0)
                    + 0.2 * away_subset.shift(2, fill_value=0)
                    )

                home_streak_total = pd.Series(
                    streak_value[subset['Home_Team'] == team],
                    name='Home_Streak_Total')
                away_streak_total = pd.Series(
                    streak_value[subset['Away_Team'] == team],
                    name='Away_Streak_Total')
                home_streak_series = pd.Series(
                    streak_home_value,
                    name='Home_Streak')
                away_streak_series = pd.Series(
                    streak_away_value,
                    name='Away_Streak')

                df_res.update(home_streak_total)
                df_res.update(away_streak_total)
                df_res.update(home_streak_series)
                df_res.update(away_streak_series)

                df_res.sort_index()

    return df_res


# Load the cleaned data

df_results = pd.read_csv('Results_Cleaned.csv')
df_standings = pd.read_csv('Standings_Cleaned.csv')

# Create a new feature to see if the weekday was a weekend or not
df_results['Weekend'] = pd.to_datetime(
                    df_results['Day']
                    ).dt.dayofweek.map(
                    lambda x: 0 if x < 4 else 1)

# Create a new feature to see if the match took place during the
# morning, afternoon, or evening.
df_results['Daytime'] = df_results['Time'].map(get_daytime)

# Generate the keys to merge centain values to the results dataframe
df_results, df_standings = generate_keys(df_results, df_standings)
column_list = ['Position', 'Goals_For', 'Goals_Against']
dict_key = df_standings[['Key'] + column_list].set_index('Key').to_dict()

# Merge the Position, Goals for, and Goals againts from the standings to the
# Results dataframe
for column in column_list:
    df_results[column + '_Home'] = df_results['Key_Home'].map(dict_key[column])
    df_results[column + '_Away'] = df_results['Key_Away'].map(dict_key[column])
    df_results.dropna(
        subset=[column + '_Home',
                column + '_Away'],
        inplace=True)
    df_results[[column + '_Home', column + '_Away']] = df_results[
                            [column + '_Home', column + '_Away']
                            ].astype('int64')
