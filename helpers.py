import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import os
import datetime


class Data(object):

    def __init__(self):
        self.data = {}
        self.setup_dataset()
        
    def setup_dataset(self):
    
        # Load dataset
        self.df_train_data, self.df_test_data = self._load_dataset()

        self._eda_analysis()
        self._outlier_check()
        self._handle_missing_data()
        self._corr_check()
        self._get_plots()
        self._feature_engineering()

    def _eda_analysis(self):

        # Display top 5 rows 
        print(self.df_train_data.head())

        # Display columns in dataframe
        print(self.df_train_data.columns)

        # Shape of dataframe
        print(self.df_train_data.shape)

        # Description of all data and statistics
        print(self.df_train_data.describe())
        
        # Data types
        print("Dtypes: \n", self.df_train_data.dtypes)

        print("Explatory Data Analysis completed. \n")
    
    def _outlier_check(self):

        # Concatenate train and test data
        self.df = pd.concat([self.df_train_data, self.df_test_data], keys=('train','test'))

        # Call duration as a new column, convert TimeDelta format to total seconds
        self.df['CallDuration'] = pd.to_datetime(self.df['CallEnd']) - pd.to_datetime(self.df['CallStart'])
        self.df['CallDuration'] = [t.total_seconds() for t in self.df['CallDuration']]

        # Change column orders
        cols = self.df.columns.tolist()
        col_orders = cols[:-2] + ['CallDuration', 'CarInsurance']
        self.df = self.df[col_orders]

        # Outliers check for numerical columns 
        ## Balance column
        sns.boxplot(x='Balance', data=self.df, palette='flare')

        if not os.path.exists("plots"):
            os.mkdir("plots")

        plt.savefig("plots/balance_boxplot.png")

        ## CallDuration column
        plt.clf()
        sns.boxplot(x='CallDuration', data=self.df, palette='flare')
        plt.savefig("plots/call_duration_boxplot.png")

        ## DaysPassed column
        plt.clf()
        sns.boxplot(x='DaysPassed', data=self.df, palette='flare')
        plt.savefig("plots/days_passed_boxplot.png")

        ## PrevAttempts column
        plt.clf()
        sns.boxplot(x='PrevAttempts', data=self.df, palette='flare')
        plt.savefig("plots/prev_attempts_boxplot.png")

        # Remove outliers from Balance and PrevAttemtps column
        self.df.drop(self.df.loc[self.df.Balance == self.df.Balance.max()].index, inplace=True)
        self.df.drop(self.df.loc[self.df.PrevAttempts == self.df.PrevAttempts.max()].index, inplace=True)

        print("Outlier checks completed. \n")

    def _handle_missing_data(self):

        # Handle missing values
        print("Missing values: \n", self.df.isnull().sum())

        perc_with_missing_values = {self.df.columns[i]: mv/len(self.df) for i, mv in enumerate(self.df.isnull().sum()) if mv > 0}
        print("Percentage of missing values: \n", perc_with_missing_values, '\n')

        # Fill any corresponding rows of Outcome with -1 from DaysPassed with "No Campaign"
        self.df.loc[self.df['DaysPassed'] == -1, 'Outcome'] = 'no_campaign'

        # Remove any column if there are more than 50% of missing values
        self.df['Outcome'].fillna('None', inplace=True)

        # Fill Communication column with None, and Job and Education columns with the most frequent value
        self.df['Communication'].fillna('None', inplace=True)
        self.df['Education'].fillna(self.df.mode().iloc[0]['Education'], inplace=True)
        self.df['Job'].fillna(self.df.mode().iloc[0]['Job'], inplace=True)

        # Check for duplicate rows and drop them
        duplicate_values = self.df[self.df.duplicated()]
        print("Number of duplicate rows: {}".format(self.df[self.df.duplicated()]))
        if len(duplicate_values)>0: self.df.drop([i for i in duplicate_values])

        print("Missing data handled. \n")

    def _corr_check(self):

        # Correlation 
        df_corr = self.df.corr()
        df_corr_car_insurance = df_corr['CarInsurance'][:-1]
        df_corr_car_insurance = df_corr_car_insurance.sort_values(ascending=False)

        print("Correlated columns and their correlations: \n", df_corr_car_insurance)

        # Correlation plot
        plt.clf()
        mask = np.zeros_like(df_corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)]=True
        with sns.axes_style('white'):
            plt.figure(figsize = (15,10))
            sns.color_palette('pastel')
            sns.heatmap(data=df_corr, annot=True, mask=mask, square=True, linewidths=.5, vmax=.5)
            plt.savefig("plots/correlation_matrix_heatmap.png")

        print("Correlations checked and correlation matrix saved. \n")

    def _get_plots(self):

        # Categorical column plots
        categorical_cols = self.df.select_dtypes(include=['object']).columns 
        for col in categorical_cols:
            if col != 'CallEnd' and col != 'CallStart':
                plt.clf()
                sns.barplot(col, 'CarInsurance', data=self.df, palette='flare')
                plt.savefig('plots/{}_carinsurance_plot.png'.format(col))

        # Pairwise plots across important features selected based on correlation values
        pair_features = ['CarInsurance', 'CallDuration', 'PrevAttempts', 'DaysPassed', 'Balance', 'Age', 'CarLoan', 'NoOfContacts', 'HHInsurance']
        plt.clf()
        sns.pairplot(data=self.df[pair_features], hue='CarInsurance', palette='flare', vars=pair_features, height=2.0)
        plt.savefig('plots/pairwise_plots.png')

        print("Pairwise plot saved. \n")

    def _feature_engineering(self):

        # Months from string to numerical values
        self.df['CallStartHour'] = pd.to_datetime(self.df['CallStart']).dt.hour
        self.df['LastContactMonth'] = pd.to_datetime(self.df['LastContactMonth'], format="%b").dt.month
        self.df.drop(columns=['CallStart', 'CallEnd'], inplace=True)

        # Numerical columns
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        df_numerical = self.df[numerical_cols]
        print("Numerical columns: \n", numerical_cols)

        # Convert categorical columns into one-hot encoding
        categorical_cols = self.df.select_dtypes(include=['object']).columns

        print("Categorical columns: \n", categorical_cols)

        df_categorical = self.df[categorical_cols]
        df_categorical = pd.get_dummies(df_categorical, dummy_na=True)

        # Merge categorical and numerical dataframes
        self.df_all = pd.concat([df_categorical, df_numerical], axis=1)

        # Split the dataset
        idx = pd.IndexSlice
        df_train_data = self.df_all.loc[idx[['train',],:]]
        df_test_data = self.df_all.loc[idx[['test',],:]]

        print("Columns: \n", df_train_data.columns)
        
        # Convert them into numpy arrays
        self.data['train_labels'] = df_train_data['CarInsurance'].values
        self.data['train_data'] = df_train_data.loc[:, df_train_data.columns != 'CarInsurance'].values
        self.data['test_labels'] = df_test_data['CarInsurance'].values
        self.data['test_data'] = df_test_data.loc[:, df_test_data.columns != 'CarInsurance'].values
        
        print("Number of features: ", self.data['train_data'].shape[1])

        print("Train and test data split. \n")
        
    def _load_dataset(self):

        # Load the train and test data
        df_train_data = pd.read_csv("data/carInsurance_train.csv", index_col = 'Id')
        df_test_data = pd.read_csv("data/carInsurance_test.csv", index_col = 'Id')

        print("Data loaded. \n")

        return df_train_data, df_test_data
