#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
### split data
features_train, features_test, labels_train , labels_test = train_test_split(features,labels, test_size=0.3, random_state=42)

print 'Start train on DT'
clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print 'Totally %d POI was predicted on test set which has %d persons'%(sum(pred), len(labels_test))
print 'No. of POI on test set is %d'%sum(labels_test)
num_true_poi = 0
for idx, label in enumerate(labels_test):
    if label == 1 and label == pred[idx]:
        num_true_poi += 1
print "No. of predicted true poi is %d "%num_true_poi
print precision_score(labels_test, pred)
print 'Precision %s, recall %s, f1_score %s'%(precision_score(labels_test, pred), recall_score(labels_test, pred),
                                              f1_score(labels_test, pred))
acc = accuracy_score(labels_test, pred)
print 'Accuracy for decision tree is %.3f'%acc





