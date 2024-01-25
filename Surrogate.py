# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:48:33 2022

@author: IRMAS
"""

import random
import pickle
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor


#This class implements different surrogates to speed up the evolutionary algorithm

class Surrogate:

    def predict(self, test_data):
        model = load('gbr.pkl')
        prediction = model.predict(test_data)
        return prediction
    def test(self,d,l):
        print(d)
        print(l)
    def random_forest(self,train_data,train_label):
        pass
    def elm(self,train_data,train_label):
        pass
    def xg_boost(self,train_data,train_label):
        pass
    def svm(self,train_data,train_label):
        pass
    def bo(self,train_data,train_label):
        pass
    def gbm_regressor(self, train_data, train_label):
        GBR = GradientBoostingRegressor()
        parameters = {'learning_rate': [0.01, 0.02, 0.03, 0.04],
                      'subsample': [0.9, 0.5, 0.2, 0.1],
                      'n_estimators': [100, 500, 1000, 1500],
                      'max_depth': [4, 6, 8, 10]
                      }
        grid_GBR = GridSearchCV(estimator=GBR, param_grid=parameters, cv=2, n_jobs=-1)
        grid_GBR.fit(train_data, train_label)
        dump(grid_GBR, 'gbr.pkl')
