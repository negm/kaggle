import time
start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import pipeline, grid_search
from sklearn.metrics import mean_squared_error, make_scorer
from nltk.metrics import edit_distance
from nltk.stem.porter import *
stemmer = PorterStemmer()
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
#import enchant
import random
random.seed(313)

def str_stem(str1):
    str1 = str1.lower()
    str1 = str1.replace(" in.","in.")
    str1 = str1.replace(" inch","in.")
    str1 = str1.replace("inch","in.")
    str1 = str1.replace(" in ","in. ")
    str1 = (" ").join([stemmer.stem(z) for z in str1.split(" ")])
    return str1
    

def str_common_word(str1, str2):
    str1, str2 = str1.lower(), str2.lower()
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

def str_title_distance(str1, str2):
    xx = edit_distance(str1,str2)
    return xx

def str_whole_word(str1, str2, i_):
    str1, str2 = str1.lower().strip(), str2.lower().strip()
    cnt = 0
    #if len(str1)>0 and len(str2)>0:
    #    cnt = len(re.findall(str1,str2))
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt
def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RSME  = make_scorer(fmean_squared_error, greater_is_better=False)


def func2(df):
    df = df[df['value'] != 'No']
    dfout = pd.DataFrame({ 'product_uid' : df['product_uid'].unique() ,
                           'brand_name': ' '.join(df['name']) })
    return  dfout

#df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
df_train = pd.read_csv('train.csv', encoding="ISO-8859-1")
#df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('test.csv', encoding="ISO-8859-1")
#df_desc = pd.read_csv('../input/product_descriptions.csv')
df_desc = pd.read_csv('product_descriptions.csv', encoding="ISO-8859-1")
attr = pd.read_csv('attributes.csv', encoding="ISO-8859-1")

num_train = df_train.shape[0]
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_desc, how='left', on='product_uid')
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title'] +"\t"+df_all['product_description']
df_all['query_distance_title'] = df_all['product_info'].map(lambda x:str_title_distance(x.split('\t')[0],x.split('\t')[1])).astype(np.float)
df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))
df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
df_all['query_title_len_prop']=df_all['len_of_title']/df_all['len_of_query']
df_all['query_desc_len_prop']=df_all['len_of_description']/df_all['len_of_query']
df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']


df_all.to_csv("df_all2.csv" , encoding="ISO-8859-1")  #no need to keep reprocessing for further grid searches

#df_all = pd.read_csv('df_all2.csv', encoding="ISO-8859-1")

df_all = df_all.drop(['search_term','product_title','product_description','product_info' ],axis=1)
df_all.head()
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']

#y_train = df_train['relevance'].astype(str)
y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values

rfr = RandomForestRegressor()
clf = pipeline.Pipeline([('rfr', rfr)])
param_grid = {'rfr__n_estimators' : list(range(109,129,2)), 'rfr__max_depth': list(range(9,15,2))}
model = grid_search.GridSearchCV(estimator = clf,param_grid = param_grid, n_jobs = 1, cv = 3, verbose = 100, scoring=RSME)
model.fit(X_train, y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

y_pred = model.predict(X_test)
print(len(y_pred))
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission2.csv',index=False)
print("--- Training & Testing: %s minutes ---" % ((time.time() - start_time)/60))



