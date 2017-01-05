# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:40:18 2015

@author: hmn
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import csv as csv
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn import cross_validation
import numpy as np
import pandas as pd
from datetime import datetime
from math import log10,pow,exp,log
#os.chdir(r'C:\Users\negmhu\Dropbox\Kaggle Shit\Restaurant Revenue')

def Preprocessing( df):
   "All preprocessing in a single place called heaven"
   
   df['City Group'] = df['City Group'].map( {'Other': 1, 'Big Cities': 2} ).astype(int)
   df['Type'] = df['Type'].map( {'FC': 4, 'IL': 3, 'DT': 2, 'MB': 1} ).astype(int)
   df['revenue'] = np.log10(df['Type'] * df['City Group'])
   df[['P1','P2','P4','P5','P6','P7','P8', 'P9','P10','P11','P12','P13','P19','P20','P21','P22','P23','P28']] = np.log10(df[['P1','P2','P4','P5','P6','P7','P8', 'P9','P10','P11','P12','P13','P19','P20','P21','P22','P23','P28']])
   df[['P3','P13','P14','P15','P16','P17','P18', 'P24','P25','P26','P27','P29','P30','P31','P32','P33','P34','P35','P36', 'P37']] = np.sqrt(df[['P3','P13','P14','P15','P16','P17','P18', 'P24','P25','P26','P27','P29','P30','P31','P32','P33','P34','P35','P36', 'P37']])

   df['Open Date'] = (datetime.now() - df['Open Date']).astype('timedelta64[D]') / 365
   df['new'] = np.log10(df['Open Date']*df['Type'] * df['City Group'])
   df['Open Date'] = np.log10(df['Open Date'] )
   #df = df.drop(['City','Type','P5', 'P9', 'P12', 'P30', 'P32', 'P34'], axis=1)
   df = df.drop(['City'], axis=1)
   return df
# read data
df = pd.read_excel('train.xls', header=0, parse_dates=True, infer_datetime_format=True )
train_target = np.log10(df['revenue'].values)+0.003
#train_target = df['revenue'].values
df_t = pd.read_excel('test.xlsx', header=0, parse_dates=True, infer_datetime_format=True )

df = Preprocessing(df)
df_t = Preprocessing(df_t)
#preprocessing and transformations

# 1- convert city type to 0, 1



x=range(1,42)
y = range(1,42)
train = df[x].values

 

test_data = df_t[y].values
ids = df_t['Id'].values

#model & prediction
params = {'n_estimators': 25000, 'max_depth': 3, 'subsample': 0.4,
          'learning_rate': 0.02, 'min_samples_leaf': 1, 'random_state': 3}
rf = BaggingRegressor(n_estimators = 800000)
#scores = cross_validation.cross_val_score(rf, train, train_target, cv=5)
rf = rf.fit(train,train_target)
output = rf.predict(test_data)
output2 = np.power(10,output)





#post processing and writing result file

predictions_file = open("AdvBagging.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Id","Prediction"])
open_file_object.writerows(zip(ids, output2))
predictions_file.close()
print 'Done.'
