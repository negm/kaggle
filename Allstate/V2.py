# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 08:46:41 2016

@author: negmhu
changes from V1:
--add a classifier to detect outliers
update it for the linux
add xgb & ensemble

"""

# -*- coding: utf-8 -*-
"""

"""
import os
import pandas as pd
import numpy as np
from sklearn import ensemble, feature_extraction, preprocessing, cross_validation
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
os.chdir(r'C:\Users\negmhu\Desktop\things\kaggle\Allstate')

def fout(x):
    if (x['loss'] > 40000):
        x['catout'] = 1
    else:
        x['catout'] = 0
    return x
# import data
ID = 'id'
TARGET = 'loss'
OUT_TARGET = 'catout'
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')

y_train = train[TARGET].ravel()
#add the dummy for extremes
train['catout']=0
train = train.apply(fout, axis=1)
y_cat = train[OUT_TARGET].ravel()
train.drop([ID, TARGET,OUT_TARGET], axis=1, inplace=True)
test.drop([ID], axis=1, inplace=True)

print("{},{}".format(train.shape, test.shape))

ntrain = train.shape[0]
ntest = test.shape[0]
train_test = pd.concat((train, test)).reset_index(drop=True)

features = train.columns

cats = [feat for feat in features if 'cat' in feat]
for feat in cats:
    train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]

print(train_test.head())

x_train = np.array(train_test.iloc[:ntrain,:])
x_test = np.array(train_test.iloc[ntrain:,:])
class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

def get_oof(clf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

et_params = {
    'n_jobs': 1,
    'n_estimators': 1000,
    'max_features': 0.5,
    'max_depth': 20,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 1,
    'n_estimators': 1000,
    'max_features': 0.2,
    'max_depth': 20,
    'min_samples_leaf': 2,
}
SEED = 313
NFOLDS = 3
kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)

clf = RandomForestClassifier(**rf_params)
clf.fit(x_train,y_train)
ext = clf.predict(x_test)
ext = clf.predict_proba(x_test)

temp= y_train
y_train = y_cat

rf_extremes = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)

et_oof_train, et_oof_test = get_oof(et)
rf_oof_train, rf_oof_test = get_oof(rf)

print("ET-CV: {}".format(mean_absolute_error(y_train, et_oof_train)))
print("RF-CV: {}".format(mean_absolute_error(y_train, rf_oof_train)))