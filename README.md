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

We seek a dependency relationship between columns with CarInsurance sales. Here in the correlation table sorted in decreasing order and correlation heatmap, it's obvious that CarInsurance is correlated with CallDuration as well as a positive correlation with PrevAttempts and DaysPassed. However, there is no strong signal among other variables. 

<p align="center">
    <img src="/visuals/correlations.png" width=300>
    <img src="/visuals/correlation_matrix_heatmap.png" width=500>
</p>

**Data visualisation/plots:**

Important information from the plots:
- CallDuration: Longer calls are likely to make people purchase car insurance.
- Age: Older people are more likely to buy car insurance. 
- DaysPassed: People are more likely to buy car insurance if they're contacted after longer period of time. 

<p align="center">
    <img src="/visuals/pairwise_plots.png" width=700>
</p>

<p align="center">
    <img src="/visuals/Communication_carinsurance_plot.png" width=300>
    <img src="/visuals/Education_carinsurance_plot.png" width=300>
    <img src="/visuals/Marital_carinsurance_plot.png" width=300>
</p>

<p align="center">
    <img src="/visuals/Job_carinsurance_plot.png" width=300>
    <img src="/visuals/LastContactMonth_carinsurance_plot.png" width=300>
</p>






### **2. Model Selection**

**Baseline Model**


**XGBoost Model**

**Neural Network Model**


## 
### **3. Setup the environment**

Python requirement: > Python3.7.5
<br>$ pip install -r requirements.txt

### **4 Automated Tests**


Filepath to run model tests: kaggle-car-insurance/
<br>$ python -m unittest discover -s tests -p test_models.py -t ..
