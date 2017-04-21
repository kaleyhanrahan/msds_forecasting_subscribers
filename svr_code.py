import os
import pandas as pd
import numpy as np

import csv
from sklearn import svm
import sklearn

path = "C:/Users/Andrew Pomykalski/Desktop/Capstone/SVR_data_code"
os.chdir(path)

## Load in data ##
training = pd.read_csv('C:/Users/Andrew Pomykalski/Desktop/Capstone/SVR_data_code/training.csv')
validation = pd.read_csv('C:/Users/Andrew Pomykalski/Desktop/Capstone/SVR_data_code/validation.csv')
testing = pd.read_csv('C:/Users/Andrew Pomykalski/Desktop/Capstone/SVR_data_code/testing.csv')
################



## Make company names into cat codes ##
training['ShortName'] = training['ShortName'].astype('category')
training['ShortName'] = training['ShortName'].cat.codes
validation['ShortName'] = validation['ShortName'].astype('category')
validation['ShortName'] = validation['ShortName'].cat.codes
testing['ShortName'] = testing['ShortName'].astype('category')
testing['ShortName'] = testing['ShortName'].cat.codes

## Scaling of data for Grid Search and SVR ##
## Used a scaling factor of mean = 0 and standard deviation = 1 ##
## Need to scale training x values and validation x values. ##
## Leave all response variables unscaled to output unscaled values ##

from sklearn import preprocessing as pre
X_train = training.iloc[:,0:-1] ## training data, took off the response column
X_validate = validation.iloc[:,0:-1] ## validation data, took off the response column
y_train = training.iloc[:,-1] ## response column for training data

scaler = pre.StandardScaler().fit(X_train) ## Set the scaling on the x variables. DO NOT PUT THE RESPONSE COLUMN IN THIS ##
X_train_scaled = scaler.transform(X_train) ## Scaling the training data 
X_validate_scaled = scaler.transform(X_validate) ## Scaling the validation data
################



## Grid Search algorithm ##
# documentation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
from sklearn.model_selection import GridSearchCV
from time import time

## First, set up a generic SVR model ##
Cost = 100000000
gam = 0.000001
epsilon = 0.01

SVR_model = svm.SVR(kernel = 'rbf', C = Cost, gamma = gam, epsilon = epsilon).fit(X_train_scaled, y_train)

## For the grid search algorithm, it allows the user to input values for C, epsilon, and kernel 
## Just put in the values, seperated by commas in the brackets for C and epsilon 
## The three options for kernel include 'rbf', 'linear', and 'poly'; all should be entered in single quotes
## The more values placed into the parameters, the longer the algorithm will take to run 
## I would recommend starting with values of 1,10,100,1000 for C
## epsilon should be something around 0.1,0.01,0.001

## Set parameter testing values ##
param_grid = {'C': [100000000, 1000000000], 'epsilon':[0.01, 0.001, 0.0001], 'kernel': ['rbf']}
              
grid_search = GridSearchCV(SVR_model, param_grid=param_grid) ## Input test SVR model and input the parameter function defined above 
start = time() ## Start a time to judge how long it takes
grid_search.fit(X_train_scaled, y_train) ## Fitting the grid search with the training data 

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
print(grid_search.cv_results_)
################



## If you decide to use a kernel 'rbf', you will need to add a gamma parameter. 
## I would recommend setting up another grid search with the added parameter 'gamma'
## Ex: param_grid = {'C': [10], 'gamma':[0.1,0.01,0.001], 'epsilon':[0.01], 'kernel': ['rbf']}
## And run rest of code above this comment to get the best gamma


## SVR Model ##
## Input parameters that were returned from grid search ##
## The parameters in the function now are the final ones that I found to be the best
SVR_model = svm.SVR(kernel = 'rbf', C=Cost, gamma = gam, epsilon = epsilon).fit(X_train_scaled, y_train)

## This is the predicted array of subscriber counts ##
svr_final_output = SVR_model.predict(X_validate_scaled)

## Taking a look at error
from sklearn.metrics import mean_squared_error

## The accuracy or error messure we chose was Root Mean Squared Error
## This is the calculation for RMSE for the validation set and the predicted subscriber counts
svr_final_mse = mean_squared_error(validation.iloc[:,-1], predict_y_array)
svr_final_rmse = np.sqrt(mse)

## Put predictions in a DataFrame
predict = pd.DataFrame(predict_y_array)
################
