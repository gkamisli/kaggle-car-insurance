import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import sys
import time

from helpers import Data

python_version = (sys.version_info[0], sys.version_info[1], sys.version_info[2])
assert sys.version_info > (3,7,5), f"Your Python version is: {python_version[0]}.{python_version[1]}.{python_version[2]}. Upgrade it to at least version 3.7.5"

class BaselineModel(object):

    def __init__(self):

        self.clf = None

    def train_model(self, X_train, y_train):

        st = time.time()

        # GridSearch and 5-Cross Validation
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_features': [15, 16, 17, 18, 19],
            'max_depth': [7,8,9,10],
            'criterion': ['gini'],
            'min_samples_split': [9, 10, 11]
        }

        # Instantiate RandomForestClassifier and train
        clf = RandomForestClassifier(random_state=42)
        cv_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
        cv_clf.fit(X_train, y_train)

        print("Best parameters: \n", cv_clf.best_params_)

        # Train the model with best parameters
        best_params = cv_clf.best_params_
        self.clf = RandomForestClassifier(
                                        n_estimators = best_params['n_estimators'],
                                        max_features = best_params['max_features'],
                                        max_depth = best_params['max_depth'],
                                        criterion = best_params['criterion']
                                        )
        self.clf.fit(X_train, y_train)

        print("Time: {}".format(time.time()-st))

    def classify_insurance(self, person_details):
        pass
    

class XGBoostModel(object):
    
    def __init__(self):

        self.clf = None

    def train_model(self, X_train, y_train):

        st = time.time()

        # GridSearch and 5-Cross Validation
        param_grid = {
            'n_estimators': [1000, 1100, 1200],
            'learning_rate': [0.01, 0.001],
            'max_depth': [3,4,5,6,7],
            'min_child_weight': [1,2,3],
            'subsample': [0.8, 0.9],
            'n_jobs': [10],
            'colsample_bytree':[0.3,0.4,0.5],
            'objective': ['binary:logistic'],
            'eval_metric': ['error'],
        }

        # Instantiate RandomForestClassifier and train
        clf = XGBClassifier(use_label_encoder=False, random_state=42)
        cv_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
        cv_clf.fit(X_train, y_train)

        print("Best parameters: \n", cv_clf.best_params_)

        # Train the model with best parameters
        best_params = cv_clf.best_params_

        self.clf = XGBClassifier(
                                use_label_encoder=False, n_estimators=best_params['n_estimators'],
                                learning_rate=best_params['learning_rate'], max_depth=best_params['max_depth'],
                                min_child_weight=best_params['min_child_weight'], subsample=best_params['subsample'],
                                n_jobs=best_params['n_jobs'], objective=best_params['objective'], 
                                colsample_bytree=best_params['colsample_bytree'], eval_metric=best_params['eval_metric']
                                )
        self.clf.fit(X_train, y_train)

        print("Time: {}".format(time.time()-st))     

    def classify_insurance(self, person_details):
        pass

class ModelPipes(object):

    def __init__(self, baseline=True, xgboost=True):
        
        self.baseline = BaselineModel()
        self.xgboost = XGBoostModel()
        self.data = Data().data
                
        # Split train/validation data
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.data['train_data'], 
                                                                              self.data['train_labels'], 
                                                                              stratify = self.data['train_labels'], 
                                                                              test_size = 0.2, 
                                                                              random_state = 42)

    def train_pipes(self):

        self.baseline.train_model(self.X_train, self.y_train)
        self.xgboost.train_model(self.X_train, self.y_train)

    def eval_pipes(self):

        if not self.baseline.clf:
            self.baseline.train_model(self.X_train, self.y_train)
        if not self.xgboost.clf:
            self.xgboost.train_model(self.X_train, self.y_train)

        # Accuracy, F1 score on validation data
        for model in [self.baseline, self.xgboost]:
            if model == self.baseline: 
                print("Baseline Model")
            else: 
                print("XGBoost Classifier")

            preds = model.clf.predict(self.X_val)
            acc = model.clf.score(self.X_val, self.y_val)
            f1 = f1_score(self.y_val, preds)
            print('''Accuracy: {}\n,
                    F1 Score: {}'''.format(acc, f1)) 



if __name__ == "__main__":
    models = ModelPipes()
    models = models.eval_pipes()

    

        