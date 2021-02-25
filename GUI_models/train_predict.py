import glob
import sys
import itertools
import re
import glob
import pandas as pd
import numpy as np
import tkinter as tk
from urllib.request import urlopen
from bs4 import BeautifulSoup
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from os.path import dirname, basename, isfile, join
from Data import load_df
import Clean
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from pandastable import Table, TableModel

button_ttc = None


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


def accept_cookies(year, league, round=None):
    '''
    Starts the driver which returns the html code of the webpage
    of a given year, league, and round to extract the data afterwards.

    Parameters
    ----------
    year: int
        Year of the match
    league: str
        League of the match
    round: int
        Number of the round from which the code starts the search.
        If None, the driver will start from the last round of that year,

    Returns
    -------
    driver: webdriver
        The webdriver object that can extract the HTML code to look for the
        data in the wanted year, league and round
    '''

    ROOT_DIR = "https://www.besoccer.com/"
    driver_dir = './chrome_driver/chromedriver.exe'
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("start-maximized")
    options.add_argument("disable-infobars")
    options.add_argument("--disable-extensions")
    driver = webdriver.Chrome(driver_dir, chrome_options=options)
    if round:
        driver.get(ROOT_DIR + league + str(year) +
                   "/group1/round" + str(round))
    else:
        driver.get(ROOT_DIR + league + str(year))

    cookies_button = driver.find_elements_by_xpath(
                    '//button[@class="sc-ifAKCX hYNOwJ"]')

    try:
        for button in cookies_button:
            if button.text == "AGREE":
                relevant_button = button
                relevant_button.click()
    except AttributeError:
        pass
    finally:
        return driver


def extract_results(driver):
    '''
    Returns the results from the matches for a given year, league, and round

    Parameters
    ----------
    driver: webdriver
        The webdriver object that can extract the HTML code to look for the
        data in the wanted year, league and round

    Returns
    -------
    results: list
        Returns a nested list with:
            Home Team: home_team
            Away Team: away_team
            Time: time of the match
            Link: link of the URL containing info about the match
        If one of the list couldn't be extracted, the function
        return a list of null values
    '''
    page = driver.page_source
    soup = BeautifulSoup(page, 'html.parser')
    regex = re.compile('nonplayingnow')
    soup_table = soup.find("table", {"id": 'tablemarcador'})
    if soup_table:
        results_table = soup_table.find('tbody').find_all(
            "tr", {"class": regex})
    else:
        return None

    num_matches = len(results_table)
    home_team = [results_table[i].find('td', {'class': 'team-home'}).find(
        'span').find('a').text for i in range(num_matches)]
    away_team = [results_table[i].find('td', {'class': 'team-away'}).find(
        'span').find('a').text for i in range(num_matches)]
    link = []
    time = []
    for i in range(num_matches):
        try:
            link.append(results_table[i].find_all('td')[2].find('a')['href'])
        except AttributeError:
            link.append(np.nan)

        try:
            time.append(results_table[i].find(
                'div', {'class': 'clase'}).text)
        except AttributeError:
            time.append(np.nan)

    results = [home_team, away_team, time, link]

    if len(set([len(i) for i in results])) == 1:
        return results
    else:
        return [None] * len(results)


def train_predict_check():
    root = tk.Tk()

    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            root.destroy()
            sys.exit('Quitting...')

    def OnButtonClick(button_id):
        global button_ttc
        button_ttc = button_id
        root.destroy()

    root.title("Train, predict, or check")
    root.protocol("WM_DELETE_WINDOW", on_closing)
    tk.Label(root, text="Do you want to train a new model,\n"
             + "use an existing one to predict results,\n"
             + "or check the performance of the existing ones?",
             justify=tk.CENTER,
             padx=20,
             pady=20).pack(fill=BOTH, expand=True)
    button1 = tk.Button(root, text="Train",
                        command=lambda *args: OnButtonClick(1))
    button1.pack(fill=BOTH, expand=True)
    button2 = tk.Button(root, text="Predict",
                        command=lambda *args: OnButtonClick(2))
    button2.pack(fill=BOTH, expand=True)
    button3 = tk.Button(root, text="Check",
                        command=lambda *args: OnButtonClick(3))
    button3.pack(fill=BOTH, expand=True)
    # tk.Radiobutton(root, text="Train", indicatoron=0,
    #                width=30, padx=20, variable=tr, value=1,
    #                command=root.destroy).grid(row=2, column=0)
    # tk.Radiobutton(root, text="Predict", indicatoron=0,
    #                width=30, padx=20, variable=te, value=1,
    #                command=root.destroy).grid(row=3, column=0)
    # tk.Radiobutton(root, text="Check", indicatoron=0,
    #                width=30, padx=20, variable=ch, value=1,
    #                command=root.destroy).grid(row=4, column=0)
    root.mainloop()
    return button_ttc


def get_models():
    modules = glob.glob(join(dirname(__file__), "../Models/*.py"))
    classifiers = ([basename(f)[:-3] for f in modules
                    if isfile(f) and not
                    (f.endswith("__init__.py") or f.endswith("classifier.py"))
                    ])
    box_var = []
    boxes = []
    box_num = 0
    root = tk.Tk()

    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            root.destroy()
            sys.exit('Quitting...')

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.geometry("450x300+120+120")
    tk.Label(
        root, text="Select the classifiers you want to tune and train",
        justify=tk.LEFT, font=("Arial", 14), padx=5, pady=10
        ).grid(row=0, column=0, columnspan=2)
    r = 0
    for clf in classifiers[:int(len(classifiers)/2) + 1]:
        box_var.append(tk.IntVar())
        boxes.append(
            tk.Checkbutton(
                root, text=clf,
                variable=box_var[box_num]
                )
            )
        box_var[box_num].set(1)
        boxes[box_num].grid(row=r + 1, column=0)
        box_num += 1
        r += 1
    confirm_row = r + 1
    r = 0
    for clf in classifiers[int(len(classifiers)/2) + 1:]:
        box_var.append(tk.IntVar())
        boxes.append(
            tk.Checkbutton(
                root, text=clf,
                variable=box_var[box_num]
                )
            )
        box_var[box_num].set(1)
        boxes[box_num].grid(row=r + 1, column=1)
        box_num += 1
        r += 1

    tk.Button(root, text="Confirm", width=10, relief=tk.RAISED,
              command=root.destroy, justify=tk.CENTER
              ).grid(row=confirm_row, column=0, pady=10, columnspan=2)
    root.mainloop()
    mask = [val.get() for val in box_var]
    return list(itertools.compress(classifiers, mask))


def train():

    classifiers = get_models()
    if classifiers:
        modules = glob.glob(join(dirname(__file__), "../Models/*.joblib"))
        models = ([basename(f).replace('_model.joblib', '')
                   for f in modules if isfile(f)])
        common_clf = list(set(classifiers).intersection(models))
        diff_clf = list(set(classifiers).difference(models))
        box_var = []
        boxes = []
        box_num = 0
        root = tk.Tk()

        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                root.destroy()
                sys.exit('Quitting...')

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.geometry("450x300+120+120")
        tk.Label(
            root, text=("The following classifiers have already"
                        + "been trained.\n"
                        + "Do you want to overwrite any of them?"),
            justify=tk.LEFT, font=("Arial", 14), padx=5, pady=10
                ).grid(row=0, column=0, columnspan=2)
        r = 0
        for clf in common_clf[:int(len(common_clf)/2) + 1]:
            box_var.append(tk.IntVar())
            boxes.append(
                tk.Checkbutton(
                    root, text=clf,
                    variable=box_var[box_num]
                    )
                )
            box_var[box_num].set(0)
            boxes[box_num].grid(row=r + 1, column=0)
            box_num += 1
            r += 1
        confirm_row = r + 1
        r = 0
        for clf in common_clf[int(len(common_clf)/2) + 1:]:
            box_var.append(tk.IntVar())
            boxes.append(
                tk.Checkbutton(
                    root, text=clf,
                    variable=box_var[box_num]
                    )
                )
            box_var[box_num].set(0)
            boxes[box_num].grid(row=r + 1, column=1)
            box_num += 1
            r += 1

        tk.Button(root, text="Confirm", width=10, relief=tk.RAISED,
                  command=root.destroy, justify=tk.CENTER
                  ).grid(row=confirm_row, column=0, pady=10, columnspan=2)
        root.mainloop()
        mask = [val.get() for val in box_var]
        return list(diff_clf) + list(itertools.compress(common_clf, mask))
    else:

        def countdown(count):
            # change text in label
            label['text'] = f"No model selected, exiting program in {count}"
            if count > 0:
                # call countdown again after 1000ms (1s)
                root.after(1000, countdown, count-1)
            else:
                close()

        def close():
            root.destroy()

        root = tk.Tk()
        root.geometry("450x300+120+120")
        count = 5
        label = tk.Label(root,
                         text=f"No model selected, exiting program in {count}")
        label.place(x=35, y=15)

        # call countdown first time
        countdown(count)
        root.mainloop()


def get_leagues():

    leagues = [
            'premier_league', 'primera_division',
            'serie_a', 'ligue_1', 'bundesliga',
            'eredivisie', 'primeira_liga'
            ]
    leagues_names = [
                'Premier League', 'Primera Division',
                'Serie A', 'Ligue 1', 'Bundesliga',
                'Eredivisie', 'Primeira Liga'
                ]
    leagues_2 = [
            'championship', 'segunda_division',
            'serie_b', 'ligue_2', '2_liga',
            'eerste_divisie', 'segunda_liga'
            ]
    leagues_names_2 = [
                'Championship', 'Segunda Division',
                'Serie B', 'Ligue 2', '2. Bundesliga',
                'Eerste Divisie', 'Segunda Liga'
                ]
    box_var = []
    boxes = []
    box_num = 0
    root = tk.Tk()

    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            root.destroy()
            sys.exit('Quitting...')

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.geometry("450x300+120+120")
    tk.Label(
        root, text="""Select the leagues you want to get predictions from""",
        justify=tk.CENTER, padx=5, pady=10
        ).grid(row=0, column=0, columnspan=2)
    r = 0
    for league in leagues_names:
        box_var.append(tk.IntVar())
        boxes.append(
            tk.Checkbutton(
                root, text=league,
                variable=box_var[box_num]
                )
            )
        box_var[box_num].set(0)
        boxes[box_num].grid(row=r + 1, column=0)
        box_num += 1
        r += 1

    r = 0
    for league in leagues_names_2:
        box_var.append(tk.IntVar())
        boxes.append(
            tk.Checkbutton(
                root, text=league,
                variable=box_var[box_num]
                )
            )
        box_var[box_num].set(0)
        boxes[box_num].grid(row=r + 1, column=1)
        box_num += 1
        r += 1

    tk.Button(root, text="Confirm", width=10, relief=tk.RAISED,
              command=root.destroy, justify=tk.CENTER
              ).grid(row=8, column=0, pady=10, columnspan=2)
    root.mainloop()
    mask = [val.get() for val in box_var]
    return list(itertools.compress(leagues + leagues_2, mask))


def get_date(link):
    '''
    Return the date of the match. The main page includes
    the date, but not the year, so it takes the links
    of the dataframe to see the full date
    Parameters
    ----------
    link: str
        A string which includes the link corresponding
        to each match

    Returns
    -------
    date: str
        The date in which the match takes place
    '''

    ROOT = 'https://www.besoccer.com'

    print(f'Working on it {link}')
    date = None
    URL = ROOT + link
    match_url = urlopen(URL)
    match_bs = BeautifulSoup(match_url.read(), 'html.parser')
    match_table = match_bs.find('div', {'id': 'marcador'})

    if match_table:
        if match_table.find('div', {'class': 'marcador-header'}):
            date = match_table.find(
                'div', {'class': 'marcador-header'}).find(
                'span', {'class': 'jor-date'}).text
            return date.split(',')[1]

    return None


def get_last_round_season(df, league):
    '''
    Return the matchs left in the last available round in df
    If there are no matches left in that round, it returns
    the matches from the next round
    IF there are no rounds left in that season, it returns a
    message saying that there are no matches available for that league
    Parameters
    ----------
    df: pandas DataFrame
        A string which includes the link corresponding
        to each match

    Returns
    -------
    date: str
        The date in which the match takes place
    '''
    # Get the last year of the specified league
    last_year = df[df['League'] == league].Year.max()
    last_round = df[(df['League'] == league)
                    & (df['Year'] == last_year)].Round.max()
    list_results = ['Home_Team', 'Away_Team', 'Time', 'Link',
                    'Season', 'Round', 'League']
    dict_results = {x: [] for x in list_results}
    driver = accept_cookies(last_year, league, last_round)
    results = extract_results(driver)
    driver.quit()

    if results is None:
        return 0, last_round, None

    for i, key in enumerate(list_results[:-3]):
        dict_results[key].extend(results[i])

    dict_results['Season'].extend([last_year] * len(results[0]))
    dict_results['Round'].extend([last_round] * len(results[0]))
    dict_results['League'].extend([league] * len(results[0]))

    df_predict = pd.DataFrame(dict_results)
    mask = df_predict['Time'].map(lambda x: ':' in x,
                                  na_action=None)
    df_predict = df_predict[mask]

    if df_predict.shape[0] == 0:
        dict_results = {x: [] for x in list_results}
        driver = accept_cookies(last_year, league, last_round + 1)
        results = extract_results(driver)
        driver.quit()
        if results is None:
            return 1, last_round + 1, None
        for i, key in enumerate(list_results[:-3]):
            dict_results[key].extend(results[i])

        dict_results['Season'].extend([last_year] * len(results[0]))
        dict_results['Round'].extend([last_round + 1] * len(results[0]))
        dict_results['League'].extend([league] * len(results[0]))
        df_predict = pd.DataFrame(dict_results)
        mask = df_predict['Time'].map(lambda x: ':' in x,
                                      na_action=None)
        df_predict = df_predict[mask]
        if df_predict.shape[0] == 0:
            return 2, None, None
        else:
            df_predict['Day'] = pd.to_datetime(
                        df_predict['Link'].map(get_date))
            return 3, last_round + 1, df_predict

    else:
        df_predict['Day'] = pd.to_datetime(df_predict['Link'].map(get_date))
        return 4, last_round, df_predict


def get_next_matches():
    '''
    Return the upcoming matches that hasn't been played
    These matches are the immediate next to the current ones
    Returns
    -------
    league_round_list: list
        List of pandas Dataframes
    df_to_show: pandas Dataframe
        Dataframe with the name of each match in each league
    '''
    leagues = get_leagues()
    df = load_df.load_leagues('Data/Results')
    league_round_list = []
    matches = {}
    for league in leagues:
        print(f'Getting information about {league}')
        msg, last_round, df_predict = \
            get_last_round_season(df, league)
        if msg == 0:
            print(f'No available matches in league {league}'
                  + f'for round {last_round}')
        elif msg == 1:
            print(f'Round {last_round - 1} was already played.\n'
                  + f'When searching for round {last_round}'
                  + f'in league {league} \n'
                  + f'no matches were found')
        elif msg == 2:
            print(f'Too many missing rounds for {league}\n'
                  + 'Please, update the database')
        elif msg == 3:
            print(f'Round {last_round - 1} was already played.\n'
                  + f'Returning matches from round {last_round}'
                  + f' in league {league} \n')
            league_round_list.append(df_predict)
        else:
            print(f'Returning matches from round {last_round}'
                  + f' in league {league} \n')
            league_round_list.append(df_predict)

    for league in league_round_list:
        matches[league['League'][0]] = []
        for index, row in league.iterrows():
            matches[row['League']].append(f'{row["Home_Team"]} vs.'
                                          + f'{row["Away_Team"]}')

    df_to_show = pd.DataFrame.from_dict(matches)
    root = tk.Tk()
    table = DataFrameTable(root, df_to_show)
    button = tk.Button(master=root, text="Confirm", command=root.destroy)
    button.pack(side=tk.BOTTOM)
    root.mainloop()
    return league_round_list
    # root = Tk()

    # def on_closing():
    #     if messagebox.askokcancel("Quit", "Do you want to quit?"):
    #         root.destroy()
    #         sys.exit('Quitting...')

    # root.protocol("WM_DELETE_WINDOW", on_closing)
    # root.geometry("450x300+120+120")

    # root.mainloop()
    # print(root.filename)


def choose_model():
    modules = glob.glob(join(dirname(__file__), "../Models/*.py"))
    classifiers = ([basename(f)[:-3] for f in modules
                    if isfile(f) and not
                    (f.endswith("__init__.py") or f.endswith("classifier.py"))
                    ])
    box_var = []
    boxes = []
    box_num = 0
    root = tk.Tk()

    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            root.destroy()
            sys.exit('Quitting...')

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.geometry("450x300+120+120")
    tk.Label(
        root, text="Select the classifiers use for the prediction",
        justify=tk.LEFT, font=("Arial", 14), padx=5, pady=10
        ).grid(row=0, column=0, columnspan=2)
    r = 0
    for clf in classifiers[:int(len(classifiers)/2) + 1]:
        box_var.append(tk.IntVar())
        boxes.append(
            tk.Checkbutton(
                root, text=clf,
                variable=box_var[box_num]
                )
            )
        box_var[box_num].set(0)
        boxes[box_num].grid(row=r + 1, column=0)
        box_num += 1
        r += 1
    confirm_row = r + 1
    r = 0
    for clf in classifiers[int(len(classifiers)/2) + 1:]:
        box_var.append(tk.IntVar())
        boxes.append(
            tk.Checkbutton(
                root, text=clf,
                variable=box_var[box_num]
                )
            )
        box_var[box_num].set(0)
        boxes[box_num].grid(row=r + 1, column=1)
        box_num += 1
        r += 1

    tk.Button(root, text="Confirm", width=10, relief=tk.RAISED,
              command=root.destroy, justify=tk.CENTER
              ).grid(row=confirm_row, column=0, pady=10, columnspan=2)
    root.mainloop()
    mask = [val.get() for val in box_var]
    return list(itertools.compress(classifiers, mask))
