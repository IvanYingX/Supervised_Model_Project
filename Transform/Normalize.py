import pandas as pd

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
