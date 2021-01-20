# Kaggle - Car Insurance Data Analysis and Models



## Contents
1. Explatory Data Analysis
- Dataframe shape, head, 
- Null/Outlier check 
- Visualisation/plots
- Feature engineering
2. Baseline Model
3. XGBoost Model
4. Setup the environment
5. Automated Tests


**1. Explatory Data Analysis (EDA)**
EDA consists of couple of steps such as checking raw dataframe, handling null (NaN) or outliers, retrieving correlation values between columns and CarInsurance, investigating plots for more explanations, and feature engineering. 

- DataFrame/columns: 


- Outlier check:

According to Balance and PrevAttempts boxplots below, we can see that values are distributed homogenously, but one particular data point is far from other data points. So, that data point (maximum number in columns) is dropped from the dataset not to create any noise. As can be seen, there is no outlier in CallDuration and DaysPassed. 

<p align="center">
    <img src="/visuals/balance_boxplot.png" width=400>
    <img src="/visuals/prev_attempts_boxplot.png" width=400>
</p>

<p align="center">
    <img src="/visuals/call_duration_boxplot.png" width=400>
    <img src="/visuals/days_passed_boxplot.png" width=400>
</p>

- Missing data:

- Visualisation/plots:





**2. Baseline Model**


**3. XGBoost Model**

**4. Neural Network Model**


## Setup the environment

Python requirement: > Python3.7.5
pip install -r requirements.txt

**4 Automated Tests**


Filepath to run model tests: kaggle-car-insurance/
$ python -m unittest discover -s tests -p test_models.py -t ..
