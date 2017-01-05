'''
Building on the success of the stack with log of the target
This script will add a NN classifier for the extreme values > 40,000
to try to capture the extreme values in the test set as well
'''
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import BayesianRidge, LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
#os.chdir(r'C:\Users\negmhu\Desktop\things\kaggle\Allstate')
ID = 'id'
TARGET = 'loss'
NFOLDS = 3
SEED = 313
NROWS = None
DATA_DIR = "../input"

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SUBMISSION_FILE = "sample_submission.csv"

train = pd.read_csv(TRAIN_FILE, nrows=NROWS)
test = pd.read_csv(TEST_FILE, nrows=NROWS)

y_train = train[TARGET].ravel()
y_train_c = y_train > 50000
y_train = np.log(y_train)

train.drop([ID, TARGET], axis=1, inplace=True)
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
del(train)
del(test)
del(train_test)
del(cats)
kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        #params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 1000)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


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
    'n_estimators': 200,
    'random_state': 313,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 200,
    'random_state': 313,
    'max_features': 0.2,
    'max_depth': 8,
    
    'min_samples_leaf': 2,
}

xgb_params = {
    'seed': 313,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.01,
    'objective': 'reg:linear',
    'max_depth': 7,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'mae',
    'nrounds': 500
}
ada_params = {
'base_estimator': DecisionTreeRegressor(),
'n_estimators': 200,
'random_state':313
}

default_params = {
}

xg = XgbWrapper(seed=SEED, params=xgb_params)
et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
rdg = SklearnWrapper(clf=BayesianRidge, seed= SEED, params=default_params)
grd = SklearnWrapper(clf=GradientBoostingRegressor, seed= SEED, params=default_params)
ada = SklearnWrapper(clf=AdaBoostRegressor, seed=SEED, params= ada_params)

xg_oof_train, xg_oof_test = get_oof(xg)
et_oof_train, et_oof_test = get_oof(et)
rf_oof_train, rf_oof_test = get_oof(rf)
rdg_oof_train, rdg_oof_test = get_oof(rdg)
grd_oof_train, grd_oof_test = get_oof(grd)
ada_oof_train, ada_oof_test = get_oof(ada)

x_train = np.c_[x_train,xg_oof_train,et_oof_train, rf_oof_train,rdg_oof_train,grd_oof_train,ada_oof_train]
x_test = np.c_[x_test,xg_oof_test,et_oof_test, rf_oof_test,rdg_oof_test,grd_oof_test,ada_oof_test]


mlpC = MLPClassifier (hidden_layer_sizes = (100,20,10), alpha=1e-5, random_state = 313)
mlpR = MLPRegressor (hidden_layer_sizes = (130,50,30), alpha=1e-5, random_state = 313)
mlpR.fit(x_train,y_train)
mlpC.fit(x_train,y_train_c)
nnc_train = mlpC.predict(x_train)
nnr_train = mlpR.predict(x_train)
nnc_test = mlpC.predict(x_test)
nnr_test = mlpR.predict(x_test)

print("XG-CV: {}".format(mean_absolute_error(np.exp(y_train), np.exp(xg_oof_train))))
print("ET-CV: {}".format(mean_absolute_error(np.exp(y_train), np.exp(et_oof_train))))
print("RF-CV: {}".format(mean_absolute_error(np.exp(y_train), np.exp(rf_oof_train))))
#print("LR-CV: {}".format(mean_absolute_error(y_train, lr_oof_train)))
print("RDG-CV: {}".format(mean_absolute_error(np.exp(y_train), np.exp(rdg_oof_train))))
print("GRD-CV: {}".format(mean_absolute_error(np.exp(y_train), np.exp(grd_oof_train))))
print("ADA-CV: {}".format(mean_absolute_error(np.exp(y_train), np.exp(ada_oof_train))))


x_train = np.c_[xg_oof_train, et_oof_train, rf_oof_train,rdg_oof_train,
                           grd_oof_train,ada_oof_train, nnr_train,nnc_train]
x_test = np.c_[xg_oof_test, et_oof_test, rf_oof_test,
                         rdg_oof_test, grd_oof_test,ada_oof_test,nnr_test,nnc_test]

print("{},{}".format(x_train.shape, x_test.shape))

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)
 
xgb_params = {
    'seed': 313,
    'colsample_bytree': 0.8,
    'silent': 0,
    'subsample': 0.6,
    'learning_rate': 0.001,
    'objective': 'reg:linear',
    'max_depth': 6,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'mae',
    'nrounds':10000
}

res = xgb.cv(xgb_params, dtrain, num_boost_round=500, nfold=4, seed=SEED, stratified=False,
             early_stopping_rounds=25, verbose_eval=10, show_stdv=True)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))

gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

submission = pd.read_csv(SUBMISSION_FILE)
submission.iloc[:, 1] = np.exp(gbdt.predict(dtest))
submission.to_csv('xgstacker_starter.sub.csv', index=None)