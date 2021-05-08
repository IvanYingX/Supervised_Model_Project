import pandas as pd
from sklearn.preprocessing import OneHotEncoder


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