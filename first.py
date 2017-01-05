# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 09:46:55 2016

@author: hnegm
"""

import time
start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import pipeline, grid_search
from sklearn.metrics import mean_squared_error, make_scorer
#from nltk.metrics import edit_distance
from nltk.stem.porter import *
stemmer = PorterStemmer()
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
#import enchant
import random
from sklearn.metrics import mean_squared_error
import distance
#import matplotlib.pyplot as plt
random.seed(313)
"""
def str_stem(str1):
    str1 = str1.lower()
    str1 = str1.replace(" in.","in.")
    str1 = str1.replace(" inch","in.")
    str1 = str1.replace("inch"," in.")
    str1 = str1.replace(" in "," in. ")
    str1 = str1.replace(" ' "," in. ")
    str1 = str1.replace("*","x")
    str1 = str1.replace(",","")
    str1 = str1.replace("height","h")
    str1 = str1.replace("width","w")
    str1 = str1.replace("-volt","v")
    str1 = str1.replace("r-","r")
   
    str1 = (" ").join([stemmer.stem(z) for z in str1.split(" ")])
    return str1
    
"""

def str_stem(str1):
    str1 = str1.lower().strip()
    str1 = str1.replace(" inch"," in. ")
    str1 = str1.replace("inch"," in. ")
    str1 = str1.replace(" in "," in. ")
    str1 = str1.replace("0in","0 in. ")
    str1 = str1.replace("1in","1 in. ")
    str1 = str1.replace("2in","2 in. ")
    str1 = str1.replace("3in","3 in. ")
    str1 = str1.replace("4in","4 in. ")
    str1 = str1.replace("5in","5 in. ")
    str1 = str1.replace("6in","6 in. ")
    str1 = str1.replace("7in","7 in. ")
    str1 = str1.replace("8in","8 in. ")
    str1 = str1.replace("9in","9 in. ")
    str1 = str1.replace("'"," in. ")
    str1 = str1.replace("*"," x ")
    str1 = str1.replace("x0"," x 0")
    str1 = str1.replace("x1"," x 1")
    str1 = str1.replace("x2"," x 2")
    str1 = str1.replace("x3"," x 3")
    str1 = str1.replace("x4"," x 4")
    str1 = str1.replace("x5"," x 5")
    str1 = str1.replace("x6"," x 6")
    str1 = str1.replace("x7"," x 7")
    str1 = str1.replace("x8"," x 8")
    str1 = str1.replace("x9"," x 9")
    str1 = str1.replace("-"," ")
    str1 = str1.replace(",","")
    str1 = str1.replace("height","h")
    str1 = str1.replace("width","w")
#    str1 = str1.replace("ft.","")
#    str1 = str1.replace("sq.","")
#    str1 = str1.replace("oz.","")
#    str1 = str1.replace("cu.","")
#    str1 = str1.replace("lb.","")
#    str1 = str1.replace("gal.","")
    str1 = str1.replace("by"," x ")
    str1 = str1.replace("-volt"," v")
    str1 = str1.replace("r-","r ")
   
    str1 = (" ").join([stemmer.stem(z) for z in str1.split(" ")])
    return str1
   

def str_common_word(str1, str2):
    str1, str2 = str1, str2
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

def str_whole_word(str1, str2):
    #str1, str2 = str1.lower().strip(), str2.lower().strip()
    cnt = 0
    i_ = 0
    i_ = str2.find(str1, i_)
    if i_ == -1:
       return cnt
    else:
       cnt += 1
    return cnt
def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RSME  = make_scorer(fmean_squared_error, greater_is_better=False)

def func2(df):
    #df = df[df['value'] != 'No']
    dfout = pd.DataFrame({ 'product_uid' : df['product_uid'].unique() ,
                           'names': ' '.join(df['name']),
                            'values': ' '.join( df['value'])
                           })
    return  dfout

def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist



df_train = pd.read_csv('trainU.csv' , encoding='utf-8')
df_test = pd.read_csv('testU.csv' , encoding='utf-8' )
#df_attr = pd.read_csv('../input/attributes.csv')
df_desc = pd.read_csv("product_descriptions.csv" , encoding='utf-8')
attr = pd.read_csv('attrU.csv',  encoding='utf-8')
attr['value']= attr['value'].str.encode('utf-8')
attr['name']= attr['name'].str.encode('utf-8')

attr['value']= attr['value'].astype(str)
attr['name']= attr['name'].astype(str)

#remove almost empty search queries
TBR=[3782,5302,8832,8999,9105,17860,17946,18072,18109,22214,22456,22751,24922,26487,31249,34447,35506,37412,39088,51361]
df_train = df_train.drop(TBR)

## REMOVE STOP words
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
df_train['product_title'] = df_train['product_title'].apply(lambda x: " ".join([item.lower() for item in x.split() if item.lower() not in stops]))
df_train['search_term'] = df_train['search_term'].apply(lambda x: " ".join([item.lower() for item in x.split() if item.lower() not in stops]))

df_test['product_title'] = df_test['product_title'].apply(lambda x: " ".join([item.lower() for item in x.split() if item.lower() not in stops]))
df_test['search_term'] = df_test['search_term'].apply(lambda x: " ".join([item.lower() for item in x.split() if item.lower() not in stops]))

df_desc['product_description'] = df_desc['product_description'].apply(lambda x: " ".join([item.lower() for item in x.split() if item.lower() not in stops]))

df_attr['values'] = df_attr['values'].apply(lambda x: " ".join([item.lower() for item in x.split() if item.lower() not in stops]))

df_attr['names'] = df_attr['names'].apply(lambda x: " ".join([item.lower() for item in x.split() if item.lower() not in stops]))


attr = attr[attr['value'] != 'No']
df_brands = attr[attr['name']== 'MFG Brand Name']
df_brands= df_brands.drop('name', axis=1)
df_brands= df_brands.rename(columns={'value': 'brand_name'})
df_attr = attr.groupby('product_uid').apply(func2)

num_train = df_train.shape[0]
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brands, how='left', on='product_uid')
df_all = pd.merge(df_all, df_attr, how='left', on='product_uid')

#df_all['brand_name'] = df_all['brand_name'].str.decode('utf-8')
#df_all['names'] = df_all['names'].str.decode('utf-8')
#df_all['values'] = df_all['values'].str.decode('utf-8')
#

df_all[['search_term','product_title','product_description']] = df_all[['search_term','product_title','product_description']].applymap(lambda x:str_stem(x))
#df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
#df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))


df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)

df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title'] +"\t"+df_all['product_description'] +"\t"+df_all['brand_name'] +"\t"+df_all['names'] +"\t"+df_all['values'] 
#df_all['product_info'] = df_all['product_info'].astype(str).encode('utf-8')


df_all['query_in_title'] = df_all['product_info'].apply(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1]))
df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2]))

df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))


#df_all['query_title_len_prop']=df_all['len_of_query']/df_all['len_of_title']
#df_all['query_desc_len_prop']=df_all['len_of_query']/df_all['len_of_description']

df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']

# distances

df_all['sorensen_distance_query_title'] = df_all['product_info'].map(lambda x:(distance.sorensen(x.split('\t')[0],x.split('\t')[1])))
df_all['sorensen_distance_query_desc'] = df_all['product_info'].map(lambda x: (distance.sorensen(x.split('\t')[0],x.split('\t')[2])))

#df_all['hamming_distance_query_title'] = df_all['product_info'].map(lambda x:1- (distance.hamming(x.split('\t')[0],x.split('\t')[1])))
#df_all['hamming_distance_query_desc'] = df_all['product_info'].map(lambda x:1- (distance.hamming(x.split('\t')[0],x.split('\t')[2])))


print("Sorenson acquired")

df_all['levenshtein_distance_query_title'] = df_all['product_info'].map(lambda x: (distance.levenshtein(x.split('\t')[0],x.split('\t')[1])))
df_all['levenshtein_distance_query_desc'] = df_all['product_info'].map(lambda x: (distance.levenshtein(x.split('\t')[0],x.split('\t')[2])))

print("levenshtein acquired")

df_all['jaccard_distance_query_title'] = df_all['product_info'].map(lambda x: (distance.jaccard(x.split('\t')[0],x.split('\t')[1])))
df_all['jaccard_distance_query_desc'] = df_all['product_info'].map(lambda x: (distance.jaccard(x.split('\t')[0],x.split('\t')[2])))

print("jaccard acquired")
#df_all = df_all.drop(['Unnamed: 0_x', 'Unnamed: 0_y'], axis=1)

df_all = df_all.replace([np.inf, -np.inf], 0)


df_all.to_csv("df_all2distancesStop.csv", encoding="utf-8", index=False)

#df_all = pd.read_csv("df_all2distancesStop.csv", encoding="utf-8")
#num_train = 74047


df_all = df_all.drop(['search_term','product_title','product_description','product_info', 'product_uid' ],axis=1)

df_train = df_all.iloc[:num_train]

df_test = df_all.iloc[num_train:]
id_test = df_test['id']
y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values

#clf = RandomForestRegressor(n_estimators=109, max_depth = 9)

rfr = RandomForestRegressor()
clf = pipeline.Pipeline([('rfr', rfr)])
param_grid = {'rfr__n_estimators' : list(range(109,110,1)), 'rfr__max_depth':list(range(9,10,1)) }
model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = 1, cv = 5, verbose = 120, scoring=RSME)
model.fit(X_train, y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

y_pred = model.predict(X_test)
print(len(y_pred))
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission3.csv',index=False)
print("--- Training & Testing: %s minutes ---" % ((time.time() - start_time)/60))