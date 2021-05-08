# Supervised_Model_Project

Predict the result of an upcoming football match <br>

Second Project of AI Core. In this project we look at different classification models and see which one performs better. Additionally, we look for potential stakeholders that can benefit from our idea <br>
The scope of the project falls in line with the one in the Data Pipeline project, where the collected data is meant to predict the result of a football match (Win, Lose, or Draw). The data collection has not been finished yet, so to this day (17/02/2021), the classification models will have less features than the final model. More features and function will be added as the data collection progresses. <br>
In addition to the models, this project looks at the scope of the data, so it sees if there should be a model for each country, or just one model for all countries. The former idea might decrease bias, but leads to fewer samples. <br>

The code follows these steps:
- Cleaning and Feature creation
- Training the model(s), additionally, the code allows the user to tune many models, save them, and check their performance
- Interpreting the results
- Predicting the result of upcoming matches

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
When running Main.py, or Main.ipynb, a window prompts, and we can choose which one we want to train with in a GUI as shown in the following picture:

![Main GUI](/images/Main_GUI.png)

Then, we can press train to select which models we want to use:

![Train GUI](/images/Train_GUI.png)

Once trained, the code will store the trained models in the Models folder. We can check the performance of these models pressing the Check button in the Main GUI:

![Model Preformance GUI](/images/Check_Models_GUI.png)

We can also see these results graphically in the Main notebook

![Model Preformance GUI](/images/Models_performances.png)

It can be seen that the model with best performance is the Gradient Boost, so let's take a look at the metrics of this model

## Interpreting the results

In the previous graph we can see that the accuracy of the gradient boost is around 59%. That's not very high, but considering that there are three classes to predict, it is not that bad. Also, there are many features that hasn't been included yet, such as the weather, the players info, or the city information, so eventually, this metrics will improve. <br>

We can see some other metrics of the Gradient Boost such as the confusion matrix:

![Confusion Matrix](/images/Gradient_Boost_cm.png)

In the confusion matrix above, we can see that the model especially struggles to get the draws right. This might be changed if we tweak the threshold.<br>
A good threshold can be calculated with the ROC curve

![ROC curve](/images/Gradient_Boost_ROC.png)

Each point of the ROC curve is plotted with a different threshold, so we want to get a threshold that balances the TPR and the FPR, and this can be determined by optimizing the G-Mean `g-mean = sqrt(tpr * (1-fpr))`. That threshold corresponds to 0.253:

![ROC curve with threshold](/images/Gradient_Boost_threshold.png)

Another thig to consider in the model is the feature importance, which defines how useful a feature is when classifying a sample<br>
In the gradient boost model, we have the following feature importance array (plotted):


![Feature Importance](/images/Gradient_Boost_features.png)

This means that the current position in the standing table of the home team and the away team are the most relevant features to classify the result of the match. On the other hand, whether it was played during the weekend or during the morning, afternoon or evening is not relevant for this classification.

## Prediction

Finally, we have our model chosen, so we can use it for predicting the outcome of an upcoming match. <br>
In the main GUI, the Predict button prompts the following window:

![Leagues to predict from](/images/Predict_Leagues_GUI.png)

We can choose which league(s) we want our predictions from, and after choosing the league(s) and the model, the code will webscrape to look for the next match after the results we have. So, for example if out last data is from round 20, the code will check if there are remaining matches in round 20, if not, it will go to round 21. However, if round 21 is already played, that means our dataset is out of date, and we need to update it.<br>

_Note: if this happens, you can go to the Data Pipeline repo, run the update_database script, and copy the result and standing folders in this repo_ <br>

After scraping for the next matches, this will be prompted

![Matches to predict from](/images/Predict_Matches.png)

The data is cleaned and transformed, and the model we chose (gradient boost in this case) will predict and show the outcome:

![Predictions](/images/Predictions_GUI.png)

# Final notes

I will try to update this repo until everythin is fully automated, in the meantime, there might be some bugs due to the lack of data. If that happens, just DM me and I will try to fix it ASAP
