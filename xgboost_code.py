# load base packages
import os
import pandas as pd
import numpy as np
import random
random.seed(540)

# load packages for XGBoost
import xgboost as xgb
from sklearn import pipeline, metrics, grid_search
from sklearn.metrics import mean_squared_error

# set working directory
path = "/Users/jordanbaker/Documents/School/University of Virginia/Capstone Project/Data"
os.chdir(path)



### XGBOOST GRID SEARCH SETUP ###
# source: https://www.kaggle.com/danielsack/liberty-mutual-group-property-inspection-prediction/let-s-try-to-run-gs-on-xgb-regressor/code#L53

def gini(y_true, y_pred):
    """ Simple implementation of the (normalized) gini score in numpy. 
        Fully vectorized, no python loops, zips, etc. Significantly
        (>30x) faster than previous implementions
        
        Credit: https://www.kaggle.com/jpopham91/
    """

    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true
    
    
def normalized_gini(y_true, y_pred):
    ng = gini(y_true, y_pred)/gini(y_true, y_true)
    return ng
    
    
    
def fit(train, target):
    
    # set up pipeline
    est = pipeline.Pipeline([
            ('xgb', xgb.XGBRegressor(silent=True)),
        ])
        
    # create param grid for grid search
    params = {
        'xgb__learning_rate': [0.05, 0.1, 0.3, ],
        'xgb__min_child_weight': [1, 2, ],
        'xgb__subsample': [1, ],
        'xgb__colsample_bytree': [1, ],
        'xgb__max_depth': [15, 20, ],
        'xgb__n_estimators': [1000, ],
        }

    # set up scoring mechanism
    gini_scorer = metrics.make_scorer(normalized_gini, greater_is_better=True)
    
    # initialize gridsearch
    gridsearch = grid_search.RandomizedSearchCV(
        estimator=est,
        param_distributions=params,
        scoring=gini_scorer,
        verbose=10,
        n_jobs=-1,
        cv=5,
        n_iter=3,
        )
        
    # fit gridsearch
    gridsearch.fit(train, target)
    print('Best score: %.3f' % gridsearch.best_score_)
    print('Best params:')
    for k, v in sorted(gridsearch.best_params_.items()):
        print("\t%s: %r" % (k, v))
        
    # get best estimator
    return gridsearch.best_estimator_
################


### XGBOOST ###
# documentation: https://xgboost.readthedocs.io/en/latest/#
# run grid search
est = fit(training.ix[:,2:-1], training.ix[:,-1])

# generate RMSE on validation data
# use optimal parameters identified from grid search

num_trees = 1000
learn_rate = 0.1
col_rate = 1
obs_rate = 0.9
leaf_samples = 2
depth = 10

xg = xgb.XGBRegressor(n_estimators = num_trees, nthread = -1, colsample_bytree = col_rate,
                      learning_rate = learn_rate, max_depth = depth, min_child_weight = leaf_samples,
                      subsample = obs_rate)
xg = xg.fit(training.ix[:,2:-1], training.ix[:,-1])
xg_output = xg.predict(validation.ix[:,2:-1])
xg_mse = mean_squared_error(validation.ix[:,-1], xg_output)
xg_rmse = np.sqrt(xg_mse)

# generate RMSE on testing data
xg = xg.fit(training.ix[:,2:-1], training.ix[:,-1])
xg_final_output = xg.predict(testing.ix[:,2:-1])
xg_final_mse = mean_squared_error(testing.ix[:,-1], xg_final_output)
xg_final_rmse = np.sqrt(xg_final_mse) #2712
################



### XGBOOST 2015 AND 2016 FORECASTING ###
val_test_2015 = val_test[val_test.ForecastYear == 2015]
val_test_2016 = val_test[val_test.ForecastYear == 2016]
xg_final_output_2015 = xg.predict(val_test_2015.ix[:,2:-1]) #100,174,770
xg_final_output_2016 = xg.predict(val_test_2016.ix[:,2:-1]) #100,751,320
################



### XGBOOST FINAL MODEL CREATION ###
# change service provider names to cat codes
agg_forecast['ShortName'] = agg_forecast['ShortName'].cat.codes
agg_final['ShortName'] = agg_final['ShortName'].cat.codes

# match columns
keep = agg_final.columns
agg_forecast = agg_forecast.ix[:,keep]
agg_forecast = agg_forecast.dropna(axis=1, how='any')

# train final model
xg = xg.fit(agg_final.ix[:,2:-1], agg_final.ix[:,-1])
################



### XGBOOST FINAL FORECAST ###
agg_forecast_2017 = agg_forecast[agg_forecast.ForecastYear == 2017]
agg_forecast_2018 = agg_forecast[agg_forecast.ForecastYear == 2018]
agg_forecast_2019 = agg_forecast[agg_forecast.ForecastYear == 2019]
agg_forecast_2020 = agg_forecast[agg_forecast.ForecastYear == 2020]
agg_forecast_2021 = agg_forecast[agg_forecast.ForecastYear == 2021]

output_2017 = xg.predict(agg_forecast_2017.ix[:,2:]) #98,112,672
output_2018 = xg.predict(agg_forecast_2018.ix[:,2:]) #98,365,592
output_2019 = xg.predict(agg_forecast_2019.ix[:,2:]) #98,637,984
output_2020 = xg.predict(agg_forecast_2020.ix[:,2:]) #99,521,552
output_2021 = xg.predict(agg_forecast_2021.ix[:,2:]) #100,508,710
################
