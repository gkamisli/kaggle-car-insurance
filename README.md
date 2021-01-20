# Kaggle - Car Insurance Data Analysis and Models



## Contents
1. Explatory Data Analysis
- Dataframe: shape, columns, description, dtypes 
- Null/Outlier check 
- Data visualisation/plots
- Feature engineering
2. Model Selection
- Baseline Model
- XGBoost Model
- Neural Network Model
3. Setup the environment
4. Automated Tests
##

### **1. Explatory Data Analysis (EDA)**

EDA consists of couple of steps such as checking raw dataframe, handling null (NaN) or outliers, retrieving correlation values between columns and CarInsurance, investigating plots for more explanations, and feature engineering. 

**DataFrame:**

Exploration of the dataset, including a snapshot of the dataset, datatypes of each column, statistics behind each column is needed before going deeper into the data processing and feature engineering steps. 

Our dataset has 18 columns of which 8 columns are *object* datatype (i.e. string or mixed), and the rest are *int64* datatype. 

<p align="center">
    <img src="visuals/dataframe_head.png">
</p>

<p align="center">
    <img src="/visuals/dtypes_per_column.png" width=200>
</p>


**Outlier check**:

According to Balance and PrevAttempts boxplots below, we can see that values are distributed homogenously, but one particular data point is far from other data points. So, that data point (maximum number in columns) is dropped from the dataset not to create any noise. As can be seen, there is no obvious outlier in CallDuration and DaysPassed columns. Note: CallDuration is calculated by the difference between CallEnd and CallStart and it's added at this stage in order to investigate its correlation with CarInsurance. 

<p align="center">
    <img src="/visuals/balance_boxplot.png" width=400>
    <img src="/visuals/prev_attempts_boxplot.png" width=400>
</p>

<p align="center">
    <img src="/visuals/call_duration_boxplot.png" width=400>
    <img src="/visuals/days_passed_boxplot.png" width=400>
</p>

**Missing data:**

As one of biggest problem inherent in datasets, tackling with missing data is important task for the predictive model. Even though some models such as XGBoost classifiers can handle missing data, others like RandomForest classifier or Neural Networks are sensitive to data with missing values. Hence, we need to impute them before building the models. 

In our dataset, Job, Education, Communication, and Outcome columns have missing values. While Job and Education have considerable amount of missing data (0.4% and 4.3%, respectively), there are significant number of missing value in Communication and Outcome (22.5% and 76%, respectively).

Steps followed:
1. Assign "No Campaign" in *Outcome* to the rows with *DaysPassed = -1*
2. Fill remaining NA in *Outcome* with None
3. Fill NA in *Communication* with None
4. Fill NA in *Education* and *Job* with the most frequent value 

<p align="center">
    <img src="/visuals/missing_data_per_column.png" width=200>
</p>

**Correlation:**

We seek a dependency relationship between columns with CarInsurance sales. Here in the correlation table sorted in decreasing order and correlation heatmap, it's obvious that CarInsurance is correlated with CallDuration as well as a positive correlation with PrevAttempts and DaysPassed. CarInsurance success is negatively correlated with HHInsurance, NoOfContacts, CarLoan, and LastContactDay. However, there is no strong signal among other variables. 

<p align="center">
    <img src="/visuals/correlations.png" width=300>
    <img src="/visuals/correlation_matrix_heatmap.png" width=550>
</p>

**Data visualisation/plots:**

Important information from the plots:
- CallDuration: Longer calls are likely to make people purchase a car insurance.
- Age: Older people are more likely to buy a car insurance. 
- DaysPassed: People are more likely to buy a car insurance if they're contacted after longer period of time. 
- HHInsurance: People having HHInsurace are less likely to buy a car insurance.

<p align="center">
    <img src="/visuals/pairwise_plots.png" width=800>
</p>

The barplots of categorical columns like Communication, Education, Marital, Job, and LastContactMonth show that single people and highly educated (tertiary) are more inclined to purchase car insurances.

<p align="center">
    <img src="/visuals/Communication_carinsurance_plot.png" width=400>
    <img src="/visuals/Education_carinsurance_plot.png" width=400>
    <img src="/visuals/Marital_carinsurance_plot.png" width=400>
</p>

<p align="center">
    <img src="/visuals/Job_carinsurance_plot.png" width=400>
    <img src="/visuals/LastContactMonth_carinsurance_plot.png" width=400>
</p>

**Feature engineering:**

Feature engineering is useful to create new features from the existing ones like we did for CallDuration as well as to scale or convert the existing columns. It compromises of processes for categorical and numerical features separately. 

Numerical features: 
- Implemented group based category on Age, Balance, CallDuration intervals as below:
```
df['Age'] = pd.qcut(df['Age'], 5, labels = [1,2,3,4,5])
df['Balance'] = pd.qcut(df['Balance'], 5, labels = [1,2,3,4,5])
df['CallDuration'] = pd.qcut(df['CallDuration'], 5, labels = [1,2,3,4,5])
```
- However, it creates more noise to the models and decreased the accuracy/F1 score. So, keeping those features as it is results better. 

Categorical features:
- Predictive models need numerical encodings for string data (e.g. one-hot encoding, word vectors with embeddings, ordinal representation); hence, categorical columns are converted into one-hot encoding representation. 

##
### **2. Model Selection**

The models used for this classification are RandomForest as the baseline model, XGBoost classifier and LSTM based neural network to compare results with it. 

**Frameworks:** pandas, numpy, seaborn, sklearn, xgboost, tensorflow, unittest

**Model cross-validation**: 

Because test data isn't labelled, training data is split into train and validation data for model comparisons. 
- Baseline (RandomForest) and XGBoost: 5-fold cross-validation is used along with Grid Search to find the best parameters and model.
- Neural Network: Validation data is obtained from 25% of training data. Then, it is used against the training data not to overfit while training the model at every epoch. 

**Baseline Model**

RandomForest classifier is used to investigate how a basic set of decision trees would perform on our data. Because not all features contribute as equally significant as others, *max_features* parameter has been used. *n_estimators* and *max_depth* are to check whether the number and depth of trees will influence on the majority of vote; hence, the prediction results. *min_samples_split* is increased for minimum number of samples required per node. 

The parameter grid used for the grid search is:

```
param_grid = {
    'n_estimators': [100, 200],
    'max_features': [15, 16],
    'max_depth': [8, 9],
    'criterion': ['gini'],
    'min_samples_split': [7, 8]
}
```

Best parameters: 
```
{'criterion': 'gini', 'max_depth': 9, 'max_features': 16, 'min_samples_split': 8, 'n_estimators': 100}
```

**XGBoost Model**

XGBoost classifier has been implemented to investigate how boosting model training in succession, with each model being fine-tuned to correct the errors made by the previous models. *n_estimators* and *max_depth* are applied for the effect of number and depth of the trees. *n_jobs* is for parallel threads to run xgboost. *subsample* and *colsample_bytree* have been implemented to set a fraction of observations and columns to randomly assign per each tree.

The parameter grid used for the grid search is:

```
param_grid = {
    'n_estimators': [800, 900],
    'learning_rate': [0.01],
    'max_depth': [3,4],
    'min_child_weight': [1],
    'subsample': [0.8],
    'n_jobs': [15],
    'colsample_bytree':[0.3],
    'objective': ['binary:logistic'],
    'eval_metric': ['error'],
}
```

Best parameters:
```
{'colsample_bytree': 0.3, 'eval_metric': 'error', 'learning_rate': 0.01, 'max_depth': 4, 'min_child_weight': 1, 'n_estimators': 900, 'n_jobs': 15, 'objective': 'binary:logistic', 'subsample': 0.8}
```

**Neural Network Model**

Neural Network is built to compare the performance of a basic deep learning model with statistical predictive models. ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau are used along with the fit function to monitor validation accuracy, save the model whenever the validation accuracy improves, halt the training or reduce the learning rate if no further improvement. 

Model parameters and hyperparameters:
```
{'LSTM_neurons': 64,
    'LSTM_dropout': 0.1, 
    'LSTM_kernel_regularizer': l2(0.00001),
    'optimizer': Adam,
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 5,
}

**Models Comparison**

## 
### **3. Setup the environment**

Python requirement: > Python3.7.5
<br>$ pip install -r requirements.txt

##
### **4 Automated Tests**

The aim of the automated test is to check the processes of training, saving, and loading the models, evaluating their performances on the same validation data, and   
Filepath to run model tests: kaggle-car-insurance/
<br>$ python -m unittest discover -s tests -p test_models.py -t ..
