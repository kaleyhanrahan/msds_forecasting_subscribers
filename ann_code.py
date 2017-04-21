import numpy as np
import pandas as pd

import os
import time

# Scikit learn Preprocessing documentation - http://scikit-learn.org/stable/modules/preprocessing.html
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing as pre
from sklearn.metrics import mean_squared_error

# Keras documentation - https://keras.io/
# ANNs built using Keras with a TensorFlow backend
# Installation instructions: https://keras.io/#installation
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam



######################
#### Load Dataset ####
######################
path = "/Users/kaleyhanrahan/UVaMSDS/Capstone/Data/"
os.chdir(path)

file_FINALtest = "testing.csv"
file_train = "training.csv"
file_test = "validation.csv"

df_test = pd.read_csv(file_test)
df_train = pd.read_csv(file_train)
df_FINALtest = pd.read_csv(file_FINALtest)
################



#########################################
#### Split into Testing and Training ####
#########################################
df_train = df_train.drop('ShortName',1)
df_train = df_train.drop('fips_x',1)

df_test = df_test.drop('ShortName',1)
df_test = df_test.drop('fips_x',1)

df_FINALtest = df_FINALtest.drop('ShortName',1)
df_FINALtest = df_FINALtest.drop('fips_x',1)

df_test.shape # = (13250, 593)
df_train.shape # = (39542, 593)
df_FINALtest.shape # = (13286, 593)
################



######################
#### Scale Inputs ####
######################
trainX = df_train.drop('SubscribersVideoEstimate',1).values 
scaler = pre.StandardScaler().fit(trainX) # Train scaler on training set inputs
trainX_scaled = scaler.transform(trainX) # Scale training set inputs
trainY = df_train['SubscribersVideoEstimate'].values

testX = df_test.drop('SubscribersVideoEstimate',1).values
testX_scaled = scaler.transform(testX) # Scale testing set inputs
testY = df_test['SubscribersVideoEstimate'].values

FINALtestX = df_FINALtest.drop('SubscribersVideoEstimate',1).values
FINALtestX_scaled = scaler.transform(FINALtestX) # Scale final testing set inputs
FINALtestY = df_FINALtest['SubscribersVideoEstimate'].values
################



###############################################
#### Build Single 2-layer Feed-Forward ANN ####
###############################################

# Information for sequential ANN structure: https://keras.io/getting-started/sequential-model-guide/
# Additional configurations to consider, though not included here are:
#       Drop out: model.add(Dropout(0.5))

# Initialize sequential model
model = Sequential()

# Add densely connected layers
model.add(Dense(20, input_dim=592, init='normal', activation='relu'))
## 20 densely connected nodes
## Input dimensions = number of predictors
## Init = initialize node weights from normal gaussian
## Activation = activation function, rectified linear unit

model.add(Dense(25, init='normal', activation='relu'))
## 25 densely connected nodes

model.add(Dense(1, init='normal'))
## Output layer with 1 node, no activation function

# Compile model (configure learning process) with loss function and optimizer
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

# Fit model on scaled training data
model.fit(trainX_scaled, trainY, nb_epoch=50, batch_size=5, verbose=0)

# evaluate the model
scores = model.evaluate(trainX_scaled, trainY)
################



######################################
#### Feed-Forward ANN GRID SEARCH ####
######################################
# Grid search of ANN parameters
# Below are three loops - for 1 layer ANNs, 2 Layer, and 3 layer
# For each loop, set the values to test for each parameter
# Train each model configuration on training set, evaluate using validation set 'test'
# After optimal parameters are identified, will conduct a final test using 'FINALtest'

#### GRID SEARCH - 1 layer #############################
# Set parameter values to test
batchSize = [3, 5, 10] # n observations in each forward/backward pass
epochs = [25, 50, 75] # n times each observation is used in a forward/backward pass
lrs = [0.01, 0.001] # learning rate for optimization function
layer1 = [5, 10, 15, 20, 25] # number of nodes in layer 1

# Number of combinations to test
total = len(batchSize) * len(epochs) * len(lrs) * len(layer1)


# Set up loop
count = 1 # Set counter
results = pd.DataFrame([]) # Initialize dataframe to hold results from each model

# Timer for each run of loop
start_time = time.time()
prevIter_time = time.time()

# Loop through all combinations of parameter values
for xBatch in batchSize:
    for xEpoch in epochs:
        for xLr in lrs:
            for xnode1 in layer1:

                # Build sequential, feed-forward ANN
                model = Sequential()
                # Add dense layer
                model.add(Dense(xnode1, input_dim=592, init='normal', activation='relu'))
                # Final layer                
                model.add(Dense(1, init='normal'))
                
                # Compile model learning process
                model.compile(loss='mean_squared_error', optimizer=Adam(lr=xLr))
                
                # Fit model on training set
                model.fit(trainX_scaled, trainY, nb_epoch=xEpoch, batch_size=xBatch, verbose=0)
                
                # Predict on testing set
                predictions = model.predict(testX_scaled)
                
                # Mean squared error
                xmse = mean_squared_error(testY, predictions)
                
                # Append MSE and model details to results dataframe
                results = results.append(pd.DataFrame({'batchSize':xBatch,'Epochs':xEpoch, 'learningRate':xLr,
                'node1':xnode1, 'mse':xmse}, index=[0]), ignore_index=True)
                
                # Print information
                iter_time = time.time() - prevIter_time # Calculate time for this iteration of loop
                elapsed_time = time.time() - start_time # Calculate total elapsed time over loop 
                print(count, "out of ",total,
                "\n batch:",xBatch," node1:",xnode1,
                "\n Total time:",elapsed_time,"\n Indiv iter time:",iter_time)
                
                # Add counter and reset time
                count += 1
                prevIter_time = time.time()

print(results)
################



#### GRID SEARCH - 2 layer #############################
# Set parameter search values
batchSize = [3, 5, 10] # n observations in each forward/backward pass
epochs = [25, 50, 75] # n times each observation is used in a forward/backward pass
lrs = [0.01, 0.001] # learning rate for optimization function
layer1 = [5, 10, 15, 20, 25] # number of nodes in layer 1
layer2 = [5, 10, 15, 20, 25] # number of nodes in layer 2

# Number of combinations to test
total = len(batchSize) * len(epochs) * len(lrs) * len(layer1) * len(layer2)


# Set up loop
count = 1 # Set counter
results = pd.DataFrame([]) # Initialize dataframe to hold results from each model

# Timer for each run of loop
start_time = time.time()
prevIter_time = time.time()

# Loop through all combinations of parameter values
for xBatch in batchSize:
    for xEpoch in epochs:
        for xLr in lrs:
            for xnode1 in layer1:
                for xnode2 in layer2:

                    # Build sequential, feed-forward ANN
                    model = Sequential()
                    # Add dense layer 1
                    model.add(Dense(xnode1, input_dim=592, init='normal', activation='relu'))
                    # Add dense layer 2                     
                    model.add(Dense(xnode2, init='normal', activation='relu'))
                    # Final layer                     
                    model.add(Dense(1, init='normal'))
            
                    # Compile model learning process
                    model.compile(loss='mean_squared_error', optimizer=Adam(lr=xLr))
            
                    # Fit model on training set
                    model.fit(trainX_scaled, trainY, nb_epoch=xEpoch, batch_size=xBatch, verbose=0)
          
                    # Predict on testing set
                    predictions = model.predict(testX_scaled)
                    
                    # Mean squared error
                    xmse = mean_squared_error(testY, predictions)

                    # Append MSE and model details to results dataframe                    
                    results = results.append(pd.DataFrame({'batchSize':xBatch,'Epochs':xEpoch, 'learningRate':xLr,
                    'node1':xnode1, 'node2':xnode2, 'mse':xmse}, index=[0]), ignore_index=True)
                    
                    # Print information
                    iter_time = time.time() - prevIter_time
                    elapsed_time = time.time() - start_time        
                    print(count, "out of ",total,
                    "\n batch:",xBatch," node1:",xnode1," node2:",xnode2,
                    "\n Total time:",elapsed_time,"\n Indiv iter time:",iter_time)
                    
                    # Add counter and reset time
                    count += 1
                    prevIter_time = time.time()

print(results)
################



############################
#### Optimal Parameters ####
############################
xBatch_optimal = 5
xEpoch_optimal = 50
xLr_optimal = 0.001
xnode1_optimal = 20
xnode2_optimal = 25
################



#########################################
#### Final Test MSE with Optimal ANN ####
#########################################
# Retrain ANN with optimal parameters identified above (on training data)
# Final evaluation metric produced by testing on second held out set, 'FINALtest'
model = Sequential()
model.add(Dense(xnode1_optimal, input_dim=592, init='normal', activation='relu'))
model.add(Dense(xnode2_optimal, init='normal', activation='relu'))
model.add(Dense(1, init='normal'))

model.compile(loss='mean_squared_error', optimizer=Adam(lr=xLr_optimal))
model.fit(trainX_scaled, trainY, nb_epoch=xEpoch_optimal, batch_size=xBatch_optimal, verbose=0)

# evaluate the model
scores = model.evaluate(trainX_scaled, trainY)

# Make predictions
ann_output = model.predict(FINALtestX_scaled)
# Final MSE on held-out testing set
ann_mse = mean_squared_error(FINALtestY, predictions)
################



################################################################################
################################################################################
############ 5-year forecasts with ANNs ########################################
################################################################################
################################################################################

######################
#### Load Dataset ####
######################
path = "/Users/kaleyhanrahan/UVaMSDS/Capstone/Data/"
os.chdir(path)

file_train = "agg_final.csv"
file_test = "agg_forecast.csv"

df_test = pd.read_csv(file_test)
df_train = pd.read_csv(file_train)
################



#########################################
#### Split into Testing and Training ####
#########################################
df_train['ShortName'] = df_train['ShortName'].astype('category')
df_train['ShortName'] = df_train['ShortName'].cat.codes
df_train = df_train.drop('Unnamed: 0',1)
df_train = df_train.drop('fips_x',1)

df_test['ShortName'] = df_test['ShortName'].astype('category')
df_test['ShortName'] = df_test['ShortName'].cat.codes
df_test = df_test.drop('Unnamed: 0',1)
df_test = df_test.drop('fips_x',1)

df_test.shape # = (66134, 584)
df_train.shape # = (64625, 585)
################



###############################################
#### Scale Inputs and Split Forecast Years ####
###############################################
trainX = df_train.drop('SubscribersVideoEstimate',1).values
scaler = pre.StandardScaler().fit(trainX) # Train scaler on training set inputs
trainX_scaled = scaler.transform(trainX) # Scale training set inputs
trainY = df_train['SubscribersVideoEstimate'].values

df_17 = df_test[df_test.ForecastYear==2017]
df_17 = df_17.values
testX_17 = scaler.transform(df_17) # Scale prediction inputs

df_18 = df_test[df_test.ForecastYear==2018]
df_18 = df_18.values
testX_18 = scaler.transform(df_18) # Scale prediction inputs

df_19 = df_test[df_test.ForecastYear==2019]
df_19 = df_19.values
testX_19 = scaler.transform(df_19) # Scale prediction inputs

df_20 = df_test[df_test.ForecastYear==2020]
df_20 = df_20.values
testX_20 = scaler.transform(df_20) # Scale prediction inputs

df_21 = df_test[df_test.ForecastYear==2021]
df_21 = df_21.values
testX_21 = scaler.transform(df_21) # Scale prediction inputs
################



##########################
#### Feed-Forward ANN ####
##########################
# Train ANN with full set of training data - 2012-2016
# Note the change in input dimensions
model = Sequential()
model.add(Dense(xnode1_optimal, input_dim=594, init='normal', activation='relu'))
model.add(Dense(xnode2_optimal, init='normal', activation='relu'))
model.add(Dense(1, init='normal'))

model.compile(loss='mean_squared_error', optimizer=Adam(lr=xLr_optimal))
model.fit(trainX_scaled, trainY, nb_epoch=xEpoch_optimal, batch_size=xBatch_optimal, verbose=0)

# evaluate the model
scores = model.evaluate(trainX_scaled, trainY)
################



#####################
#### Predictions ####
#####################
# Using model trained above, make predictions for the years 2017-2021
# Sum tract-level predictions for each year to calculate aggregate national-level predictions
predictions17 = model.predict(testX_17)
sum(predictions17) 

predictions18 = model.predict(testX_18)
sum(predictions18) 

predictions19 = model.predict(testX_19)
sum(predictions19) 

predictions20 = model.predict(testX_20)
sum(predictions20) 

predictions21 = model.predict(testX_21)
sum(predictions21) 
################
