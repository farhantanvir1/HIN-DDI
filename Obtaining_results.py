#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 09:53:35 2021

@author: illusionist
"""
"""
    importing necessary libraries
"""
import findspark
findspark.init()
import sys
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
import pyspark.sql.functions as f
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import TimestampType, StringType, FloatType 
from pyspark.streaming import StreamingContext
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import when
from pyspark.ml.feature import Imputer
# Let us import the vector assembler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import explode, col, udf, stddev as _stddev
import pandas as pd
import numpy as np
import csv
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error

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



# Getting Spark Context
sc = SparkContext.getOrCreate()
# Sql Context for the spark context
sqlContext = SQLContext(sc)

"""
    reading csv file which contains meta-path data
"""
url = "drug_data.csv"
data = pd.read_csv(url, skiprows=1, header=None)

"""
    converting features to appropriate datat types
"""
data.fillna(0)


slc = np.r_[:, 2:9]
slc_2 = np.r_[:, 9:30]

data[slc] = data[slc].astype(int)
data[slc_2] = data[slc_2].astype(np.float64)

"""
    taking sampled data. in this case, training and testing data
    with 30% DDI prevalence are taken.
    
"""

train_1 = data[data[30] == 1].sample(n=300*24)
train_2 = data[data[30] == 0].sample(n=300*56)

test_1 = data[data[30] == 1].sample(n=300*6)
test_2 = data[data[30] == 0].sample(n=300*14)

train_frames = [train_1, train_2]
test_frames = [test_1, test_2]

train_frames = sklearn.utils.shuffle(train_frames)
test_frames = sklearn.utils.shuffle(test_frames)

train = pd.concat(train_frames)
test = pd.concat(test_frames)

trainX = train.iloc[:,2:30]
trainy = train.iloc[:,-1]

testX = test.iloc[:,2:30]
testy = test.iloc[:,-1]

"""
    getting results for One class SVM
"""

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
print("Results for One class SVM")
# calculate score
evaluate_results(testy, yhat)
#score = f1_score(testy, yhat)
#print('F1 Score: %.3f' % score)


"""
    getting results for SVM
"""

svclassifier = SVC(kernel='linear')
svclassifier.fit(trainX, trainy)
y_pred = svclassifier.predict(testX)
print("Results for SVM")
evaluate_results(testy, y_pred)

"""
    getting results for Logistic regression
"""

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(trainX,trainy)
y_pred=logreg.predict(testX)
print("Results for Logistic Regression")
evaluate_results(testy, y_pred)

"""
    getting results for random forest
"""

rf = RandormForestRegressor(n_estimators=1000, random_state=42)
rf.fit(trainX, trainy)

predictions=rf.predict(testX)
print("Results for random forest")
evaluate_results(testy, predictions.round())


"""
    getting results for gradient boosting tree
"""

regressor = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=3,
    learning_rate=1.0
)
regressor.fit(trainX, trainy)

errors = [mean_squared_error(testy, y_pred) for y_pred in regressor.staged_predict(testX)]
best_n_estimators = np.argmin(errors)

best_regressor = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=best_n_estimators,
    learning_rate=1.0
)
best_regressor.fit(trainX, trainy)

y_pred = best_regressor.predict(testX)

print("Results for gradient boosting tree")
evaluate_results(testy, y_pred.round())