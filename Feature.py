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
    
df_results = pd.read_csv('Data_Cleaned.csv')
# Create a new feature to see if the weekday was a weekend or not
df_results['Weekend'] = pd.to_datetime(
                    df_results['Day']
                    ).dt.dayofweek.map(
                    lambda x: 0 if x < 4 else 1)

# Create a new feature to see if the match took place during the 
# morning, afternoon, or evening.
df_results['Daytime'] =  df_results['Time'].map(get_daytime)
