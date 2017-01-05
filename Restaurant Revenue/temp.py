# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: hnegme.
"""

import os
import csv as csv
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from math import log10, pow, exp, log
os.chdir(r'C:\Users\negmhu\Dropbox\Kaggle Shit\Restaurant Revenue')

def Preprocessing( df):
   "All preprocessing in a single place called heaven"
   
   df['City Group'] = df['City Group'].map( {'Other': 1, 'Big Cities': 2} ).astype(int)
   df['Type'] = df['Type'].map( {'FC': 4, 'IL': 3, 'DT': 2, 'MB': 1} ).astype(int)
   df['revenue'] = df['Type'] * df['City Group']
   df[['P1','P2','P4','P5','P6','P7','P8', 'P9','P10','P11','P12','P13','P19','P20','P21','P22','P23','P28']] = np.log10(df[['P1','P2','P4','P5','P6','P7','P8', 'P9','P10','P11','P12','P13','P19','P20','P21','P22','P23','P28']])
   df[['P3','P13','P14','P15','P16','P17','P18', 'P24','P25','P26','P27','P29','P30','P31','P32','P33','P34','P35','P36', 'P37']] = np.sqrt(df[['P3','P13','P14','P15','P16','P17','P18', 'P24','P25','P26','P27','P29','P30','P31','P32','P33','P34','P35','P36', 'P37']])
   #df['Open Date'] = np.log10(df['Open Date'].rank())
   #df = df.drop(['Open Date','City','Type','P5', 'P9', 'P12', 'P30', 'P32', 'P34'], axis=1)
   #df[range(3,42)] = StandardScaler().fit_transform(df[range(3,42)])
   return df
# read data
df = pd.read_csv('train.csv', header=0)
train_target = np.log10(df['revenue'].values)
df_t = pd.read_csv('test.csv',header=0)

df = Preprocessing(df)
df_t = Preprocessing(df_t)
#preprocessing and transformations

# 1- convert city type to 0, 1

x=range(3,43)
y = range(3,43)
train = df[x].values

 

test_data = df_t[y].values
ids = df_t['Id'].values

#model & prediction

rf = AdaBoostRegressor(n_estimators = 500, base_estimator = BaggingRegressor(n_estimators = 5000))
rf = rf.fit(train,train_target)
output = rf.predict(test_data)
output = np.power(10,output)





#post processing and writing result file

predictions_file = open("myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Id","Prediction"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
