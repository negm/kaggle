# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 13:18:18 2016

@author: negmhu
"""

cats = [feat for feat in features if 'cat' in feat]
x = 0
while(x < len(cats)):
    train_test[cats[x]+cats[x+1]] = train_test[cats[x]]+train_test[cats[x+1]]
    x = x+2
    
for feat in cats:
    if (max(train_test[feat])>32):
        train_test.drop([feat], axis=1, inplace=True)
x = itertools.combinations(train,
for feat in cats:
    for feat2 in cats:
        if (feat != feat2):
            train[feat+feat2]=train[feat]+train[feat2]