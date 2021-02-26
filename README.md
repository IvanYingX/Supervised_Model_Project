# Supervised_Model_Project

Predict the result of an upcoming football match <br>

Second Project of AI Core. In this project we look at different classification models and see which one performs better. Additionally, we look for potential stakeholders that can benefit from our idea <br>
The scope of the project falls in line with the one in the Data Pipeline project, where the collected data is meant to predict the result of a football match (Win, Lose, or Draw). The data collection has not been finished yet, so to this day (17/02/2021), the classification models will have less features than the final model. More features and function will be added as the data collection progresses. <br>
In addition to the models, this project looks at the scope of the data, so it sees if there should be a model for each country, or just one model for all countries. The former idea might decrease bias, but leads to fewer samples. <br>

The code follows these steps:
- Cleaning and Feature creation
- Training the model(s)
- Predicting the result of upcoming matches
Additionally, the code allows the user to tune many models, save them, and check their performance

## Data cleaning and Feature Engineering

Before using the data as input, it has to be cleaned, and for improving the accuracy, some feature engineering can be applied. Thus, the first part of this project consists on cleaning the data and extracting new features that can potentially increase the accuracy of the model. <br>
The new features include:

* **Weekend**: Whether the match took place during the weekend
* **Daytime**: Whether the match took place in the morning, during the afternoon or during the evening
* **Position_Home and Position_Away**: Position of the respective team in the corresponding round
* **Goals_For_Home and Goals_For_Away**: Cumulated sum of the goals for of the respective team in the corresponding round
* **Goals_Against_Home and Goals_Against_Away**: Cumulated sum of the goals against of the respective team in the corresponding round

Before running the main code, which is included in Main.py and Main.ipynb, one should generate the cleaned and transformed data by running Clean.py, which generates two csv's, Results_Cleaned.csv, and Standings_Cleaned.csv. These two csv's are used in the Feature.py script to generate the features and select the desired features. And store them in the <br>

The current repo already has those files, so there is no need for rerunning the mentioned scripts.

## Training

Before training, we need to split our data into training, testing, and validation. In this case, we use cross validation, so we just need to split the data into training and validation, and later on, the training set will be used in the cross validation.<br>
We need to check which classifier performs best with our data, so we train different models.<br>
We can choose which one we want to train with in a GUI as shown in the following picture:




