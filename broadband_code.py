import os
import numpy as np
import pandas as pd
from sklearn import linear_model

from sklearn.feature_selection import RFECV          # Feature Selection - Recursive Feature Elimination
from sklearn.model_selection import KFold            # Feature Selection - Cross Validation
from sklearn import metrics                          # Feature Selection - Variable Importance
from sklearn.ensemble import ExtraTreesClassifier    # Feature Selection - Variable Importance

from sklearn.model_selection import TimeSeriesSplit  # time series cross validation
from sklearn.model_selection import cross_val_score  # cross validation

import matplotlib.pyplot as plt                      # visualizations
import seaborn as sns                                # visualizations
sns.set(style="darkgrid", color_codes=True)          # viz style
import matplotlib.patches as mpatches                # plot legend

import random
random.seed(7)

#######################
### Read in and Clean Data
#######################
file = "SP-US-Multichannel-Master.xls"
path = "/Users/kaleyhanrahan/UVaMSDS/Capstone/Data/"
os.chdir(path)
master = pd.read_excel(file)

master = master[6:128]
master = master.transpose()
master = master.set_index(6)
master.columns = master.iloc[0]
master = master.drop(master.index[[0,46]])
master = master.drop(master.columns[[0,7,16,25,33,41,50,55,65,75,81,85,92,99,102,107,112,117]], axis=1)

# Pull necessary years
master = master.ix['1998Y ':'2016Y ',:]

# Create current year column
master['year'] = np.arange(1998, 2017)
# Create forecast year column
master['forecastYear'] = np.arange(2003, 2022)

master['deltaHSD'] = master['High Speed Data Subscribers'].pct_change()
master['deltaMultiChannel'] = master['Multichannel Subscribers'].pct_change()
master['deltaBasicCable'] = master['Basic Cable Subscribers'].pct_change()
master['HSDpenetration'] = master['High Speed Data Subscribers']/master['Occupied Households']

# Drop 1998
master = master.drop('1998Y ')

# Drop columns with NA values to perform regression
master = master.dropna(1)

# Create lagged Broadband for forecast
master['forecastSubs'] = master['High Speed Data Subscribers']
master['forecastSubs'] = master['forecastSubs'].shift(-5)

# Save current broadband values for plot later
hsd_save = master['High Speed Data Subscribers'].values

# Subset features of interest
j = master.columns.isin(['forecastYear','forecastSubs','year',
                         'deltaHSD', 'deltaMultiChannel', 
                         'deltaBasicCable', 'HSDpenetration'])
                         
master = master.ix[:,j]

master = master.apply(pd.to_numeric)

master.shape #(18, 7)


#######################
### Subset dataframes
#######################

X_full = master['1999Y':'2012Y'].drop('forecastSubs', 1)
Y_full = master.ix['1999Y':'2012Y', 'forecastSubs']

X_train = master['1999Y':'2010Y'].drop('forecastSubs', 1)
Y_train = master.ix['1999Y':'2010Y', 'forecastSubs']

X_test = master['2010Y':'2012Y'].drop('forecastSubs', 1)
Y_test = master.ix['2010Y':'2012Y', 'forecastSubs']

X_forecast = master['2012Y':'2017Y'].drop('forecastSubs', 1)


# Cross-validation splits - time series
tscv = TimeSeriesSplit(n_splits=5)
print(tscv)  
for train, test in tscv.split(X_full):
    print("%s %s" % (train, test))



#########################
### Feature Selection ###
#########################

var_names = X_full.columns


####### Recursive Feature Elimination #######

# Base model for feature selection
model = linear_model.LinearRegression()
# Fit Recursive Feature Elimination with 5-Fold Cross Validation
rfecv = RFECV(estimator=model, step=1, cv=KFold(5))
rfecv.fit(X_full, Y_full)
print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

print("Features sorted by their rank:")
print(sorted(zip(map(lambda x: round(x, 4), rfecv.ranking_), var_names)))
# HSDpenetration  1
# deltaBasicCable  1
# deltaHSD  1
# deltaMultiChannel  1
# year  1
# forecastYear  2
    
    
####### Variable Importance #######

# fit an Extra Trees model to the data
feat_importance = ExtraTreesClassifier(n_estimators=300,
                              random_state=9)
feat_importance.fit(X_full, Y_full)

varImportances = pd.DataFrame(data = {'Feature':var_names, 'Importance':feat_importance.feature_importances_}, 
                         index=np.arange(len(X_full.columns)))
                         
varImportances.sort_values(['Importance'], inplace=True, ascending=False)

print(varImportances)
#             Feature  Importance
#0               year    0.173889
#5     HSDpenetration    0.170000
#1       forecastYear    0.166667
#4    deltaBasicCable    0.166389
#2           deltaHSD    0.165000
#3  deltaMultiChannel    0.158056

#########################



#########################
### LINEAR REGRESSION ###
#########################

# Create linear regression object
lm = linear_model.LinearRegression()
# Train the model
lm.fit(X_train, Y_train)

# Make predictions
predY = lm.predict(X_test)

# Fitted Values
fitY = lm.predict(X_train)

# Coefficients
coeffs = pd.DataFrame(data = {'Feature':var_names, 'Coefficients':lm.coef_}, 
                         index=np.arange(len(X_full.columns)))
print(coeffs)
print('Intercept: \n', lm.intercept_)

# Test RMSE
print("Root mean squared error: %.0f"
      % np.sqrt(np.mean((predY - Y_test) ** 2)),
      "\nMean squared error: %.0f"
      % (np.mean((predY - Y_test) ** 2)))
    # RMSE = 260183

# Plot Performance on 2015-2016 Testing data
plt.plot(X_full['forecastYear'], Y_full, color='blue',
         linewidth=2)
plt.plot(X_test['forecastYear'], predY,  color='red', ls='dashed', 
            linewidth=5)
plt.plot(X_train['forecastYear'], fitY,  color='green', ls='dashed', 
            linewidth=5)
plt.show()

# Time series cross-validation
model = linear_model.LinearRegression()
scores = cross_val_score(model, X_full, Y_full,  scoring = 'mean_squared_error', cv=tscv)
np.mean(scores)
    # -3547915341565
np.sqrt(-np.mean(scores))
    # 1883591

#########################



########################
### 5-year Forecasts ###
########################
## Train on all data, predict 5 years forward

# Create linear regression object
lm = linear_model.LinearRegression()
# Train the model
lm.fit(X_full, Y_full)
# Make predictions
predY = lm.predict(X_forecast)

# Fitted Values
fitY = lm.predict(X_full)
fitY = pd.Series(fitY, index = X_full['forecastYear'])

# Create list of years and predictions
bb = pd.Series(hsd_save, index = master['year'])
bb_forecast = pd.Series(predY, index = X_forecast['forecastYear'])
bb = bb.append(bb_forecast)

# Plot
hist = plt.plot(bb[:18], linewidth =5)
fore = plt.plot(bb[18:], 'r--', linewidth =7)
fit = plt.plot(fitY, 'g--', linewidth =7)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Broadband Subscribers', fontsize=15)
axis_font = {'size':'16'}
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

red_patch = mpatches.Patch(color='r', label='Forecasted')
blue_patch = mpatches.Patch(color='b', label='Historical', ls='dashed')
plt.legend(handles=[blue_patch, red_patch], loc=4, fontsize=13)

plt.show()

print(bb[2021]) # 79998436




