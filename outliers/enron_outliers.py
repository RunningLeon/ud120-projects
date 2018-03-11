#!/usr/bin/python

import pickle
import sys
import numpy as np
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )

### find outlier in bonus and sallary, key is TOTAL {'Name': 'TOTAL', 'Max bonus': 97343619}
max_bonus = {"Name":"", "Max bonus":0, "salary_500M":0, }
bonus_500M = { "key":"", "value":0}
salary_100M = { "key":"", "value":0}
for k in data_dict:
    bo = data_dict[k]["bonus"]
    sa = data_dict[k]["salary"]
    if bo >= 5e6 and sa >= 1e6 and bo != "NaN" and sa != "NaN":
        print "key: %s, bonus: %s, salary: %s"%( k, bo, sa )
    # print bo
    # print "key: %s, bonus: %s "%( k, bo )
    if max_bonus["Max bonus"] < bo and bo != "NaN":
        max_bonus["Max bonus"] = bo
        max_bonus["Name"] = k
### ew==remove outlier
data_dict.pop("TOTAL",0)
print max_bonus
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
# print np.shape(data)

### your code below
pp = matplotlib.pyplot
for point in data:
    salary = point[0]
    bonus = point[1]
    if salary > 2e7 :
        print "Outlier: salary %s, bonus %s"%( salary, bonus )
        ##Outlier: salary 26704229.0, bonus 97343619.0
    pp.scatter( salary, bonus )
pp.xlabel("salary")
pp.ylabel("bonus")
pp.show()


