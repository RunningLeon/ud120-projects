#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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

### reduce train data set by 99%
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]


#########################################################
### your code goes here ###

#########################################################
def get_svc_acc( c=1., k="rbf"):
    from sklearn.svm import SVC
    # clf = SVC(C=1., kernel="linear")
    clf = SVC(C=c, kernel=k)
    pred = clf.fit(features_train, labels_train).predict(features_test)
    class_1 = sum(pred)
    class_0 = len(pred) - class_1

    print "There are %d samples classified in 1 class and \n %d samples classified in 0 class!"%(class_1, class_0)
    from sklearn.metrics import  accuracy_score

    acc = accuracy_score(labels_test, pred)
    print "C=%f.1, acc=%.3f" %( c, acc )

get_svc_acc(k='linear')
# get_svc_acc(c=100.)
# get_svc_acc(c=1000.)
# get_svc_acc(c=10000.)