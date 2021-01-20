import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, TimeDistributed, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Nadam, Adam, SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import os, sys
import time
import pickle

from helpers import Data

python_version = (sys.version_info[0], sys.version_info[1], sys.version_info[2])
assert sys.version_info > (3,7,5), f"Your Python version is: {python_version[0]}.{python_version[1]}.{python_version[2]}. Upgrade it to at least version 3.7.5"


class BaselineModel(object):

    def __init__(self, filepath):

        self.clf = None
        self.filepath = filepath

    def train_model(self, X_train, y_train):

        st = time.time()

        print("..RandomForest classifier..")

        # GridSearch and 5-Cross Validation
        '''
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_features': [15, 16],
            'max_depth': [9, 10, 11],
            'criterion': ['gini'],
            'min_samples_split': [7, 8, 9]
        }
        '''

        param_grid = {
            'n_estimators': [100],
            'max_features': [15],
            'max_depth': [9],
            'criterion': ['gini'],
            'min_samples_split': [7]
        }

        # Instantiate RandomForestClassifier and train
        clf = RandomForestClassifier(random_state=42)
        cv_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
        cv_clf.fit(X_train, y_train)

        # Get the best parameters
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

        with open(os.path.join(self.filepath, 'randomforestclassifier.pickle'), 'wb') as f:
            pickle.dump(self.clf, f, pickle.HIGHEST_PROTOCOL)
        print("RandomForest classifier saved.")

        print("Time: {}".format(time.time()-st), '\n')

    def classify_insurance(self, person_details):
        pass
    

class XGBoostModel(object):
    
    def __init__(self, filepath):

        self.clf = None
        self.filepath = filepath

    def train_model(self, X_train, y_train):

        st = time.time()

        print("..XGBoost classifier..")

        # GridSearch and 5-Cross Validation
        '''
        param_grid = {
            'n_estimators': [900, 1000, 1100, 1200],
            'learning_rate': [0.01],
            'max_depth': [3,4,5],
            'min_child_weight': [1],
            'subsample': [0.8],
            'n_jobs': [15],
            'colsample_bytree':[0.3,0.4,0.5],
            #'objective': ['binary:logistic'],
            #'eval_metric': ['error', 'logloss'],
        }
        '''

        param_grid = {
            'n_estimators': [900],
            'learning_rate': [0.01],
            'max_depth': [3],
            'min_child_weight': [1],
            'subsample': [0.8],
            'n_jobs': [15],
            'colsample_bytree':[0.3],
            'objective': ['binary:logistic'],
            'eval_metric': ['error', 'logloss'],
        }

        # Instantiate RandomForestClassifier and train
        clf = XGBClassifier(use_label_encoder=False, random_state=42)
        cv_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
        cv_clf.fit(X_train, y_train)

        # Get the best parameters
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

        with open(os.path.join(self.filepath, 'xgboostclassifier.pickle'), 'wb') as f:
            pickle.dump(self.clf, f, pickle.HIGHEST_PROTOCOL)
        print("XGBoost classifier saved.")

        print("Time: {}".format(time.time()-st), '\n')     

    def classify_insurance(self, person_details):
        pass


class NeuralNetworkModel(object):

    def __init__(self, filepath):

        self.clf = None
        self.filepath = filepath

    def _build_model(self):

        self.clf = Sequential()
        self.clf.add(LSTM(32, dropout=0.1, kernel_regularizer=l2(0.00001), return_sequences=True))
        self.clf.add(Dense(4))
        self.clf.add(Dense(2))
        self.clf.add(Dense(1, activation='sigmoid', name='output_layer'))

        self.clf.compile(
                        loss='binary_crossentropy', 
                        optimizer='adam', 
                        metrics=['accuracy']
                        )

    def train_model(self, X_train, y_train):
        
        print("..Neural Network..")

        self._build_model()

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.25, shuffle=True, random_state=42)
        self.clf.fit(np.expand_dims(X_train, 1), y_train, verbose=1, epochs=1, batch_size=5, validation_data=(np.expand_dims(X_val,1), y_val), callbacks=self.callback())

    def callback(self):
        checkpoint = ModelCheckpoint(
                                    filepath=os.path.join(self.filepath,"nn_model.h5"),
                                    monitor='val_accuracy',
                                    mode='max', 
                                    verbose=1,
                                    save_best_only=True,
                                    )

        earlystopping = EarlyStopping(
                                    monitor = 'val_accuracy',
                                    mode='max', 
                                    verbose=1,
                                    patience=20,
                                    restore_best_weights=True
                                    )

        learning_rate_reducer = ReduceLROnPlateau(
                                                monitor = 'val_accuracy',
                                                factor=0.9,
                                                cooldown=0,
                                                patience=30,
                                                verbose=1,
                                                mode='max',
                                                min_lr=0.5e-6
                                                )
        callbacks = [checkpoint, learning_rate_reducer]
        return callbacks

    def classify_insurance(self, person_details):
        pass
        

class ModelPipes(object):

    def __init__(self, filepath):

        self.filepath = filepath
        self.data = Data().data

        # Models
        self.baseline = BaselineModel(filepath)
        self.xgboost = XGBoostModel(filepath)
        self.nn = NeuralNetworkModel(filepath)
                
        # Split train/validation data
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.data['train_data'].astype(float), 
                                                                              self.data['train_labels'].astype(float), 
                                                                              stratify = self.data['train_labels'], 
                                                                              test_size = 0.2, 
                                                                              random_state = 42)


    def train_pipes(self):

        if not os.path.exists(self.filepath):
            os.mkdir(self.filepath)

        self.baseline.train_model(self.X_train, self.y_train)
        self.xgboost.train_model(self.X_train, self.y_train)
        self.nn.train_model(self.X_train, self.y_train)


    def load_pipes(self):

        if not os.path.exists(self.filepath):
            self.train_pipes()
            
        with open(os.path.join(self.filepath, 'randomforestclassifier.pickle'), 'rb') as f:
            self.baseline.clf = pickle.load(f)
        print("RandomForest classifier loaded.")
        
        with open(os.path.join(self.filepath, 'xgboostclassifier.pickle'), 'rb') as f:
            self.xgboost.clf = pickle.load(f)
        print("XGBoost classifier loaded.")

        self.nn.clf = load_model(os.path.join(self.filepath, 'nn_model.h5'))
        print("Neural network loaded.")


    def eval_pipes(self):

        self.load_pipes()

        # Accuracy, F1 score on validation data
        for model in [self.baseline, self.xgboost, self.nn]:

            if model == self.nn:
                print("Neural Network")
                self.X_val = np.expand_dims(self.X_val,1)
                preds = model.clf.predict(self.X_val).round()
                preds = np.squeeze(preds)
                acc = model.clf.evaluate(self.X_val, self.y_val)[1]
            else:
                if model == self.baseline: print("Baseline Model")
                else: print("XGBoost Classifier")

                preds = model.clf.predict(self.X_val)
                acc = model.clf.score(self.X_val, self.y_val)
            
            f1 = f1_score(self.y_val, preds)
            print('''Accuracy: {},
                    F1 Score: {}'''.format(acc, f1))  


if __name__ == "__main__":
    models = ModelPipes('models')
    models = models.eval_pipes()

    

        