#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
# data wrangling
print "Shape of training data: ({}, {})".format(features_train.shape[0], features_train.shape[1])
print "Shape of testing data: ({}, {})".format(features_test.shape[0], features_test.shape[1])
print features_train[0]
# quit()

## Data modeling with decision tree
from sklearn import  tree
from sklearn.metrics import  accuracy_score
import graphviz
clf = tree.DecisionTreeClassifier(min_samples_split=40)
pred = clf.fit(features_train, labels_train).predict(features_test)
acc = accuracy_score(labels_test, pred)
print acc
dot_data = tree.export_graphviz(clf, out_file='tree.dot')
# graph = graphviz.Source(dot_data)

# print len(features_train[0])

#########################################################


