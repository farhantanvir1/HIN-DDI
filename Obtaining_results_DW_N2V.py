#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 10:08:32 2021

@author: illusionist
"""

# save numpy array as csv file
from numpy import asarray
from numpy import savetxt
import numpy as np
import sklearn
import pandas as pd

from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score, f1_score, average_precision_score, precision_recall_curve

def evaluate_results(y_test, y_predict):
    print('Classification results:')
    f1 = f1_score(y_test, y_predict)
    print("f1: %.2f%%" % (f1 * 100.0)) 
    roc = roc_auc_score(y_test, y_predict)
    print("roc: %.2f%%" % (roc * 100.0)) 
    rec = recall_score(y_test, y_predict, average='binary')
    print("recall: %.2f%%" % (rec * 100.0)) 
    prc = precision_score(y_test, y_predict, average='binary')
    print("precision: %.2f%%" % (prc * 100.0)) 
    aupr = average_precision_score(y_test, y_predict)
    print("AUPR: %.2f%%" % (aupr * 100.0))  

DDI_train_dw = np.load('/home/ftanvir/Downloads/DDI_train_new_dw_lb.npy', allow_pickle=True)
DDI_test_dw = np.load('/home/ftanvir/Downloads/DDI_test_new_dw_lb.npy', allow_pickle=True)
DDI_train_n2v = np.load('/home/ftanvir/Downloads/DDI_train_new_n2v_lb.npy', allow_pickle=True)
DDI_test_n2v = np.load('/home/ftanvir/Downloads/DDI_test_new_n2v_lb.npy', allow_pickle=True)

DDI_train_dw=DDI_train_dw.item(0)
DDI_test_dw=DDI_test_dw.item(0)
DDI_train_n2v=DDI_train_n2v.item(0)
DDI_test_n2v=DDI_test_n2v.item(0)


train_frames = pd.DataFrame(DDI_train_dw)
test_frames = pd.DataFrame(DDI_test_dw)

train_frames = train_frames.T
test_frames = test_frames.T


train_1 = train_frames[train_frames[128] == 1].sample(n=300*24, replace=True)
#print(main_datatrain.shape)
train_2 = train_frames[train_frames[128] == 0].sample(n=300*56, replace=True)

#print(main_test.shape)
test_1 = test_frames[test_frames[128] == 1].sample(n=300*6, replace=True)
test_2 = test_frames[test_frames[128] == 0].sample(n=300*14, replace=True)


train_frames = [train_1, train_2]
test_frames = [test_1, test_2]


train_frames = pd.concat(train_frames)
test_frames = pd.concat(test_frames)

train_frames = sklearn.utils.shuffle(train_frames)
test_frames = sklearn.utils.shuffle(test_frames)


trainX = train_frames.iloc[:,0:128]
trainy = train_frames.loc[:,128]

testX = test_frames.iloc[:,0:128]
testy = test_frames.loc[:,128]


slc=np.r_[:,0:128]
trainX[slc]=trainX[slc].astype(np.float64)
testX[slc]=testX[slc].astype(np.float64)


trainy=trainy.to_frame()
testy=testy.to_frame()

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM
import pandas as pd

# define outlier detection model
model = OneClassSVM(gamma='scale', nu=0.01)
# fit on majority class
trainX = trainX[trainy==1]

        
model.fit(trainX)
# detect outliers in the test set
yhat = model.predict(testX)
# mark inliers 1, outliers -1
        
testy[testy == 1] = 1
testy[testy == 0] = -1

# calculate score
evaluate_results(testy, yhat)