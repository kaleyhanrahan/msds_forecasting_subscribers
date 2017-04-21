# load base packages
import os
import pandas as pd
import numpy as np
import random
random.seed(540)

# load paackages for random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# set working directory
path = "/Users/jordanbaker/Documents/School/University of Virginia/Capstone Project/Data"
os.chdir(path)



### RANDOM FOREST EXPERIMENTATION ###
# documentation: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
num_trees = 1000
leaf_samples = 1

forest = RandomForestRegressor(n_estimators=num_trees, min_samples_leaf=leaf_samples, n_jobs=-1, oob_score=True)
forest = forest.fit(training.ix[:,2:-1], training.ix[:,-1])
rf_output = forest.predict(validation.ix[:,2:-1])
rf_mse = mean_squared_error(validation.ix[:,-1], rf_output) #6,950,420
################



### RANDOM FOREST RMSE CALCULATION ###
# generate RMSE of testing set using optimal parameters (noted above)
forest = RandomForestRegressor(n_estimators=num_trees, min_samples_leaf=leaf_samples, n_jobs=-1, oob_score=True)
forest = forest.fit(training.ix[:,2:-1], training.ix[:,-1])
rf_final_output = forest.predict(testing.ix[:,2:-1])
rf_final_mse = mean_squared_error(testing.ix[:,-1], rf_final_output)
rf_final_rmse = np.sqrt(rf_final_mse)
################



### RANDOM FOREST 2015 AND 2016 FORECASTING ###
val_test['ShortName'] = val_test['ShortName'].cat.codes
val_test_2015 = val_test[val_test.ForecastYear == 2015]
val_test_2016 = val_test[val_test.ForecastYear == 2016]

rf_final_output_2015 = forest.predict(val_test_2015.ix[:,2:-1])
rf_final_output_2016 = forest.predict(val_test_2016.ix[:,2:-1])
################
