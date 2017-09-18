#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys, os
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

print os.getcwd()
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
start_time_fit = time()
clf = clf.fit(features_train, labels_train)
end_time_fit = time()
print "fit time is %f.3 s"%round(end_time_fit-start_time_fit,3)

start_time_predict = time()
pred = clf.predict(features_test)
end_time_predict = time()
print "predict time is %f.3 s"%round(end_time_predict-start_time_predict,3)

from sklearn.metrics import  accuracy_score
acc = accuracy_score(labels_test, pred)
print acc

#########################################################


