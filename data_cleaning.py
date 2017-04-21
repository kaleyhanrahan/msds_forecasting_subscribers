# load base packages
import os
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import random
random.seed(540)

# set working directory
path = "/Users/jordanbaker/Documents/School/University of Virginia/Capstone Project/Data"
os.chdir(path)



#########################
###### DATA READ IN #####
#########################

### MULTICHANNERL MASTER ###
master = pd.read_excel("SP-US-Multichannel-Master.xls")
master = master[6:128]
master = master.transpose()
master = master.set_index(6)
master.columns = master.iloc[0]
master = master.drop(master.index[[0,46]])
master = master.drop(master.columns[[0,7,16,25,33,41,50,55,65,75,81,85,92,99,102,107,112,117]], axis=1)
################


### DEMOGRAPHICS ###
demographics = pd.read_csv("CountyDemographics.csv")
demographics.fips = demographics.fips.apply(lambda x: x[1:-1]) # remove quotation marks
###############


### OPERATOR SUBS BY COUNTY ###
opsub = pd.read_table("Operator Subs by Block Group.rpt", sep = '|')
opsub = opsub[1:] # remove the NA row

opsub['Year'] = opsub.DateEndedStandard.str[0:4] # create a year column
opsub['quarter'] = opsub.DateEndedStandard.astype(str) # create a quarter column
opsub.quarter = opsub.quarter.apply(lambda x: x[5:]) # clean the quarter column
opsub = opsub[opsub.quarter == '3/31'] # subset by Q1
# Q1 was selected randomly as the 2016 data did not contain all quarters

opsub.BlockGroupFIPS = opsub.BlockGroupFIPS.astype(str) # set CBG FIPS codes to strings
opsub['fips'] = opsub.BlockGroupFIPS.astype(str) # create a FIPS column for county

# add on leading 0 as necessary (identified by length of FIPS code)
# returns 5 digit FIPS codes for counties
def fips_update(fips):
    if (len(fips) == 13):
        return '0' + fips[0:-9]
    elif (len(fips) == 14):
        return fips[0:-9]

# apply that function to the newly created FIPS column
opsub.fips = opsub.fips.map(fips_update)

# group by FIPS, ShortName, and Year
opsub_sum = opsub.groupby(['fips', '﻿ShortName', 'Year']).sum().reset_index()
################





#########################
### DATA MANIPULATION ###
#########################

### BACKFILLING DATA ###
backfill = master[['Multichannel Subscribers']] # row 128 from the spreadsheet
backfill_dates = list(range(1980,2025,1)) # grab dates from 1980 to 2024 (span of file)
backfill.index = backfill_dates # set those dates as the file index
backfill = backfill[(backfill.index > 2006) & (backfill.index < 2013)] # subset the file to the backfill dates we need
                    
dist = opsub_sum[opsub_sum.Year == '2013'] # use 2013 as the distribution year
dist = dist[['fips', '﻿ShortName', 'SubscribersVideoEstimate']] # only keep necessary columns

# calculate the percentage (market share) for each observation
denom = dist.SubscribersVideoEstimate.sum()
dist['percent'] = dist.SubscribersVideoEstimate/denom  

# create empty dataframe for imputed data
subs_impute = pd.DataFrame()               
                    
# use distributions and national subscriber totals to backfill 2007 - 2012
for i in range(2007,2013):
    temp_data = dist[['fips', '﻿ShortName', 'percent']]
    temp_subs = backfill[backfill.index == i]
    temp_subs = float(temp_subs.iloc[0])
    temp_data[str(i)] =  temp_data.percent * temp_subs
    subs_impute = pd.concat([subs_impute, temp_data], axis=1)

# remove the few NA observations
subs_impute = subs_impute.loc[:,~subs_impute.columns.duplicated()]
################


### JOINING BACKFILL AND OPSUB_SUM ###
agg = opsub_sum
years = list(range(2007,2013,1))

# set the number of years to be backfilled
backfill_years = 5

# subs_impute data is stored differently (in columns) than the non-backfill years
# this loop goes through and merges all data together
for i in range(0,backfill_years+1):
    temp_data = pd.DataFrame(subs_impute[['fips','﻿ShortName', str(years[i])]])
    temp_data['Year'] = str(years[i])
    temp_data.columns = ['fips','﻿ShortName', 'SubscribersVideoEstimate', 'Year']
    temp_data = temp_data[['fips','﻿ShortName', 'Year', 'SubscribersVideoEstimate']]
    temp_data = temp_data.reset_index(drop=True)
    agg = agg.append(temp_data)
    
agg = agg.reset_index(drop=True)

# remove the space from this column name
agg.columns.values[1] = 'ShortName'
################


### FEATURE ENGINEERING ###
s_fips = []
s_year = []
s_provider = []
s_shares = []

# go through line by line of the aggregate file and compute the subs for that given observation divided by the subs for that county
# this gives us the market share for each service provider
for i in range(0,len(agg)):
    temp_fips = agg.fips[i]
    temp_year = agg.Year[i]
    temp_provider = agg.ShortName[i]
    temp_subs = agg.SubscribersVideoEstimate[i]
    temp_sum = np.sum(agg.SubscribersVideoEstimate[(agg.fips == temp_fips) & (agg.Year == temp_year)])
    temp_share = temp_subs/temp_sum
    
    s_fips.append(temp_fips)
    s_year.append(temp_year)
    s_provider.append(temp_provider)
    s_shares.append(temp_share)

# create the shares dataframe
shares = pd.DataFrame({'fips': s_fips, 'Year': s_year, 'ShortName':s_provider, 'Share':s_shares})
################





#########################
##### FILE CREATION #####
#########################

# agg_final will represent the 5-year file used to train on
# agg_forecast will represent the 5-year file used to forecast future values on

### CREATING MODELING FILES ###
agg_final = agg
agg_final.Year = pd.to_numeric(agg_final.Year)

# subset subs from 2012-2016
# subtract 5 from each year to easily match with lagged demographics values
agg_final = agg_final[(agg_final.Year > 2011) & (agg_final.Year < 2017)]
agg_final.Year = agg_final.Year - 5
agg_final.Year = agg_final.Year.apply(str)

# subset shares to be from 2007-2011
final_shares = shares[shares.Year < '2012']

# create a unique key for each file type
# merge subscribers with share values
agg_final["key"] = agg_final.fips + agg_final.ShortName + agg_final.Year
final_shares["key"] = final_shares.fips + final_shares.ShortName + final_shares.Year
agg_final = agg_final.merge(final_shares, how='left', on='key')

# drop necessary columns
# rename columns
drops = ['key', 'ShortName_y', 'Year_y', 'fips_y']
agg_final = agg_final.drop(drops, 1)
names = ['fips', 'ShortName', 'Year', 'SubscribersVideoEstimate', 'Share']
agg_final.columns = names

# subset the demographics from 2007-2011
agg_demo = demographics[(demographics.year < 2012) & (demographics.year > 2006)]
agg_demo.year = agg_demo.year.astype(str)

# create a unique key for each file type
# merge subscribers with lagged demographic values
agg_final["key"] = agg_final.fips + agg_final.Year
agg_demo["key"] = agg_demo.fips + agg_demo.year
agg_final = agg_demo.merge(agg_final, how='left', on='key')
################


### CLEANING AGG_FINAL ###
# repurpose demograhicsasof to a column that represents the forecast year (5 years in future)
agg_final = agg_final.rename(columns={'demographicasof':'ForecastYear'})
agg_final.ForecastYear = agg_final.ForecastYear + 5

# drop unnecessary columns
drops = ['UpdOperation', 'UpdDate', 'UpdDateApproved', 'CurrencyBOPDate', 
         'CurrencyEOPDate', 'KeyGeoEntity', 'year',
         'Year', 'key', 'fips_y', 'KeyDemographicGeo']
agg_final = agg_final.drop(drops, 1)

# turn service provider names into category codes
agg_final['ShortName'] = agg_final['ShortName'].astype('category')
agg_final['ShortName'] = agg_final['ShortName'].cat.codes

# reorder columns
colnames = agg_final.columns.tolist()
colnames = colnames[0:2] + colnames[-1:] + colnames[2:-1]
agg_final = agg_final[colnames]

# remove missing observations (only a few)
agg_final = agg_final[~np.isnan(agg_final.Share)]
agg_final = agg_final[~np.isnan(agg_final.SubscribersVideoEstimate)]
agg_final = agg_final.dropna(axis=1, how='any')
################


### CREATING FORECASTING FILES ###
agg_forecast = agg
agg_forecast.Year = pd.to_numeric(agg_forecast.Year)
agg_forecast = agg_forecast[(agg_forecast.Year > 2011) & (agg_forecast.Year < 2017)]
agg_forecast.Year = agg_forecast.Year.apply(str)

# subset shares to be from 2012-2016
forecast_shares = shares[shares.Year > '2011']

# subset the demographics from 2012-2016
forecast_demo = demographics[(demographics.year > 2011) & (demographics.year < 2017)]
forecast_demo.year = forecast_demo.year.astype(str)

# create a unique key for each file type
# merge shares with lagged demographic values
forecast_shares["key"] = forecast_shares.fips + forecast_shares.Year
forecast_demo["key"] = forecast_demo.fips + forecast_demo.year
agg_forecast = forecast_demo.merge(forecast_shares, how='left', on='key')
################


### CLEANING AGG_FORECAST ###
# repurpose demograhicsasof to a column that represents the forecast year (5 years in future)
agg_forecast = agg_forecast.rename(columns={'demographicasof':'ForecastYear'})
agg_forecast.ForecastYear = agg_forecast.ForecastYear + 5

# drop unnecessary columns
drops = ['UpdOperation', 'UpdDate', 'UpdDateApproved', 'CurrencyBOPDate', 
         'CurrencyEOPDate', 'KeyGeoEntity', 'year',
         'Year', 'key', 'fips_y', 'KeyDemographicGeo']
agg_forecast = agg_forecast.drop(drops, 1)

# turn service provider names into category codes
agg_forecast['ShortName'] = agg_forecast['ShortName'].astype('category')
agg_forecast['ShortName'] = agg_forecast['ShortName'].cat.codes

# remove missing observations (only a few)
agg_forecast = agg_forecast[~np.isnan(agg_forecast.Share)]
agg_forecast = agg_forecast.dropna(axis=1, how='any')
################


### MATCHING COLUMNS IN BOTH FILES ###
# a few columns (older features) weren't tracked throughout all years
# this removes those columns so that the models won't run into any issues with extra columns

keep = agg_final.columns
cols = [col for col in agg_forecast.columns if col in temp1]
agg_forecast = agg_forecast[cols]

# check this list in future years to ensure it includes all columns that need to be dropped
drops = ['AnnualAverageHHIPercentChange', 'AnnualHHI0_25PercentChange',
       'AnnualHHI25_50PercentChange', 'AnnualMedianHHIPercentChange',
       'PerCapitaIncomePercentChange', 'PopulationEducChange',
       'PopulationEducHSChange', 'PopulationEducHSGradChange']
agg_final = agg_final.drop(drops, 1)
################


### TRAINING, VALIDATION, AND TESTING SPLIT ###

# split files into training, validation, and testing files
training = agg_final[agg_final.demographicyearreported < 2010]
val_test = agg_final[agg_final.demographicyearreported  > 2009]
validation, testing = train_test_split(val_test, test_size = 0.5)

# export to xlsx files
training.to_excel('training.xlsx', sheet_name='sheet1', index=True)
validation.to_excel('validation.xlsx', sheet_name='sheet1', index=True)
testing.to_excel('testing.xlsx', sheet_name='sheet1', index=True)
################
