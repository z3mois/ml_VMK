from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer

from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin
from hyperopt.pyll import scope
from hyperopt.plotting import main_plot_history

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from time import time

from functools import partial
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import plotly.express as px
import json
from numpy import ndarray

class Transformer:
    dict_key_words = {}
    genres = None
    direct = None
    film_loc = None
    bound = 100
    def fit(self, X, num_bound):
        y = X['awards']
        X = X.drop(columns='awards')
        
        lst = [b for i in list(X['keywords']) for b in i]
        dict_ = dict(zip(lst,[lst.count(i) for i in lst]))
        self.dict_key_words = dict(sorted(dict_.items(), key=lambda item: item[1])[::-1])
        self.bound = num_bound
        
        self.genres = list(X['genres'])
        self.direct = list(X['directors'])
        self.film_loc = list(X['filming_locations'])
        
        return X, y
    
    def transform(self, X):
        X[[b for i in self.genres for b in i]] = 0
        X[[b for i in self.direct for b in i]] = 0
        X[[b for i in self.film_loc for b in i]] = 0
        
        for key, value in self.dict_key_words.items():
            if value < self.bound:
                break
            X[key] = 0
        
        for val in {b for i in list(X['keywords']) for b in i}:
            if val in self.dict_key_words.keys() and self.dict_key_words[val] >= self.bound: 
                X.loc[[val in row['keywords'] for i, row in X.iterrows()], val] = 1
    
        list_ = ['genres', 'directors', 'filming_locations']
        for k in list_:
            X[[b for i in list(X[k]) for b in i]] = 0
    
            for val in {b for i in list(X[k]) for b in i}:
                X.loc[[val in row[k] for i, row in X.iterrows()], val] = 1
            
            X = X.drop(columns=k)
    
        X['actor_0_gender'] = X['actor_0_gender'].astype('category') 
        X['actor_1_gender'] = X['actor_1_gender'].astype('category') 
        X['actor_2_gender'] = X['actor_2_gender'].astype('category') 
    
        X = X.drop(columns=['u', 'n', 'k', 'w', 'o'])
        X = X.drop(columns='keywords')
        return X
    
    def fit_transform(self, X, num_bound):
        X, y = self.fit(X, num_bound)
        return self.transform(X), y

X_train = pd.read_json('public_tests/01_boosting_movies_input/train/train.jsonl', lines=True)
X_test = pd.read_json('public_tests/01_boosting_movies_input/test/test.jsonl', lines=True)
y_test = pd.read_json('public_tests/01_boosting_movies_gt/target.json')
tr = Transformer()
X_train, y_train = tr.fit_transform(X_train, 58)
X_test = tr.transform(X_test)
catg = ['actor_0_gender', 'actor_1_gender', 'actor_2_gender']
import warnings
warnings.filterwarnings("ignore")

trials = Trials()

def quality(params, X_train, y_train, type_boost):
    
    if type_boost == 1:
        pipeline=CatBoostRegressor(**params, cat_features=catg, verbose=False)
        X_1, X_2, y_1, y_2 = train_test_split(X_train, y_train, shuffle=True, random_state=42)
        pipeline.fit(X_1, y_1)
        Z=pipeline.predict(X_2)
        return   {'loss': mean_absolute_error(y_2, Z), 'params': params, 'status': STATUS_OK}
    
    #pipeline.set_params(**params)
    pipeline = LGBMRegressor(**params)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

    score = cross_val_score(estimator=pipeline, X=X_train, y=y_train, 
                            scoring='neg_mean_absolute_error', cv=skf, n_jobs=-1,
                            fit_params={'categorical_feature' : 'auto'})
    return   {'loss': -score.mean(), 'params': params, 'status': STATUS_OK}

grid = {
        'n_estimators' : scope.int(hp.quniform(label='n_estimators', 
                        low=200, 
                        high=700, 
                        q=1)),
        'max_depth' : scope.int(hp.quniform(label='max_depth', 
                        low=5, 
                        high=16, 
                        q=1)),
        'learning_rate' : hp.loguniform(label='learning_rate', 
                        low=-3*np.log(10), 
                        high=np.log(1))
                }

best = fmin(fn=partial(quality, #pipeline=CatBoostRegressor(), 
                       X_train=X_train, y_train=y_train, type_boost=1),
                space=grid,
                algo=tpe.suggest,
                max_evals=1,
                trials=trials,
                verbose= 1,
                rstate=np.random.default_rng(1),
                show_progressbar=True)
best_params = best.copy()
best_params['max_depth'] = int(best_params['max_depth'])
best_params['n_estimators'] = int(best_params['n_estimators'])
model = LGBMRegressor(**best_params)
model.fit(X_train, y_train, categorical_feature='auto')
print(mean_absolute_error(y_test, model.predict(X_test)))
catg = ['actor_0_gender', 'actor_1_gender', 'actor_2_gender']
model = CatBoostRegressor(**best_params, cat_features=catg, verbose=False)
model.fit(X_train, y_train)
print(mean_absolute_error(y_test, model.predict(X_test)))
print(best_params)
