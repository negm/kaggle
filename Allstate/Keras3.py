''' 
Author: Danijel Kivaranovic 
Title: Neural network (Keras) with sparse data
'''

## import libraries
import numpy as np
import math
np.random.seed(313)

import pandas as pd
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU

## Batch generators ##################################################################################################################################

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

########################################################################################################################################################

## read data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

## set test loss to NaN
test['loss'] = np.nan

## response and IDs
y = train['loss'].values
id_train = train['id'].values
id_test = test['id'].values
feats = ["cat1","cat99","cat98","cat97","cat96","cat95","cat94","cat93","cat92","cat91","cat90","cat89","cat100","cat87","cat84","cat83","cat82","cat81","cat80","cat79","cat76","cat75","cat73","cat72","cat71","cat86","cat101","cat102","cat103","cont12","cont11","cont10","cont9","cont8","cont7","cont6","cont5","cont4","cont3","cont2","cont1","cat116","cat115","cat114","cat113","cat112","cat111","cat110","cat109","cat108","cat107","cat106","cat105","cat104","cont13","cat57","cont14","cat36","cat25","cat26","cat44","cat38","cat23","cat37","cat50","cat13","cat27","cat40","cat53","cat12","cat52","cat10","cat9","cat2","cat3","cat7","cat28","cat6","cat4","cat11","cat5"]
train = train[feats]
test = test[feats]
## stack train test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis = 0)

## Preprocessing and transforming to sparse data
sparse_data = []

f_cat = [f for f in tr_te.columns if 'cat' in f]
for f in f_cat:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

f_num = [f for f in tr_te.columns if 'cont' in f]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
sparse_data.append(tmp)

del(tr_te, train, test)

## sparse train and test data
xtr_te = hstack(sparse_data, format = 'csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

del(xtr_te, sparse_data, tmp)

## neural net
def nn_model():
    model = Sequential()
    model.add(Dense(math.ceil(xtrain.shape[1]/1.5), input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(math.ceil(xtrain.shape[1]/1.2), init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'nadam')
    return(model)

## cv-folds
nfolds = 3
folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 313)

## train models
i = 0
nbags = 5
nepochs = 55
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])

for (inTr, inTe) in folds:
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe]
    yte = y[inTe]
    pred = np.zeros(xte.shape[0])
    for j in range(nbags):
        model = nn_model()
        fit = model.fit_generator(generator = batch_generator(xtr, ytr, 128, True),
                                  nb_epoch = nepochs,
                                  samples_per_epoch = xtr.shape[0],
                                  verbose = 0)
        pred += model.predict_generator(generator = batch_generatorp(xte, 800, False), val_samples = xte.shape[0])[:,0]
        pred_test += model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])[:,0]
    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(yte, pred)
    i += 1
    print('Fold ', i, '- MAE:', score)

print('Total - MAE:', mean_absolute_error(y, pred_oob))

## train predictions
df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
df.to_csv('preds_oob.csv', index = False)

## test predictions
pred_test /= (nfolds*nbags)
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv('submission_keras_feats.csv', index = False)

