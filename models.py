import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from helpers import Data

class BaselineModel(object):

    def __init__(self):

        self.clf = None

    def train_model(self, X_train, y_train):

        # Instantiate RandomForestClassifier and train
        self.clf = RandomForestClassifier(random_state=42)
        self.clf.fit(X_train, y_train)

    def classify_insurance(self, person_details):
        pass
    


class XGBoostModel(object):
    
    def __init__(self):

        self.clf = None

    def train_model(self, X_train, y_train):

        self.clf = XGBClassifier(use_label_encoder=False, objective="binary:logistic", eval_metric='error')
        self.clf.fit(X_train, y_train)     

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

        # Accuracy on validation data
        for model in [self.baseline, self.xgboost]:
            if model == self.baseline: 
                print("Baseline Model")
            else: 
                print("XGBoost Classifier")

            preds = model.clf.predict(self.X_val)
            acc = accuracy_score(self.y_val, preds)
            val_score = model.clf.score(self.X_val, self.y_val)
            f1 = f1_score(self.y_val, preds)
            print('''Score: {},
                    Accuracy: {}, 
                    F1 Score: {}'''.format(val_score, acc, f1)) 



if __name__ == "__main__":
    models = ModelPipes()
    models = models.eval_pipes()

    

        