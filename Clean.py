import pandas as pd
from load_df import load_leagues
import numpy as np


def get_hour(x):
    if len(x.split(',')) > 2:
        return x.split(',')[2]
    else:
        return '17:00'
    
def create_mask(x):
    if (len(x) == 1) | (len(x) == 5):
        return False
    else:
        return True


res_dir = './Data/Results'
sta_dir = './Data/Standings'
df_results = load_leagues(res_dir)
df_standings = load_leagues(sta_dir)
df_match = pd.read_csv('./Data/Dictionaries/Match_Info.csv')
df_team = pd.read_csv('./Data/Dictionaries/Team_Info.csv')

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
df_team.loc[missing,'Pitch'] = np.random.choice(
                values_c.index, size=len(df_team[missing]),
                p=values_c.values)

# Clean the date. The match dataframe has the right date so
# we can merge them first, and substitute the data from the 
# match dataframe

df_results = df_results.merge(df_match, on='Link')
df_results = df_results.drop_duplicates('Link', ignore_index=True)
df_results = df_results[['Home_Team', 'Away_Team', 'Result', 'Year', 'Round', 'League', 'Date_New']]
df_results = df_results.rename(columns={'Date_New': 'Date', 'Year': 'Season'})
df_results['Time'] = df_results['Date'].map(get_hour)
df_results['Day'] = pd.to_datetime(df_results['Date'].map(lambda x: x.split(',')[1]))
df_results['Weekend'] = df_results['Day'].dt.dayofweek.map(lambda x: 0 if x < 4 else 1)

# Divide the results in two columns, and also obtain the label

mask = df_results['Result'].map(create_mask)
df_results = df_results[mask]
df_results['Home_Goals'] = df_results['Result'].map(lambda x: x.split('-')[0])
df_results['Away_Goals'] = df_results['Result'].map(lambda x: x.split('-')[1])
home_goals_list = ['(0)0', '(1)2', '(0)1', '(3)3', '0', '1', '2', '3', '4',
                   '5', '6', '7', '8', '9', '10', '12', '01', '02', '03', '04',
                   '05', '06', '07', '08']
home_goals_trans = ['0', '2', '1', '3', '0', '1', '2', '3', '4', '5', '6', '7', '8',
                    '9', '10', '12', '1', '2', '3', '4', '5', '6', '7', '8']
home_goals_dict = dict(zip(home_goals_list, home_goals_trans))
away_goals_list = ['0(0)', '2(99)', '1(0)', '0(99)', 'Jan', 'Apr', 'Feb', 'Mar',
                   'May', 'Aug', 'Jul', 'Jun', '2(2)', '1(1)', '0', '1', '2','3',
                   '4', '5', '6', '7', '8', '9', '13']
away_goals_trans = ['0', '2', '1', '0', '1', '4', '2', '3', '5', '8', '7', '6', '2', '1',
                     '0', '1', '2','3', '4', '5', '6', '7', '8', '9', '13']
away_goals_dict = dict(zip(away_goals_list, away_goals_trans))
df_results['Home_Goals'] = df_results['Home_Goals'].map(home_goals_dict).astype('int')
df_results['Away_Goals'] = df_results['Away_Goals'].map(away_goals_dict).astype('int')
df_results['Label'] = ((df_results['Home_Goals'] < df_results['Away_Goals']) * 3 \
                      + (df_results['Home_Goals'] == df_results['Away_Goals']) * 2 \
                      + (df_results['Home_Goals'] > df_results['Away_Goals']) * 1) - 1
df_results = df_results.drop(['Result', 'Date'], axis=1)

