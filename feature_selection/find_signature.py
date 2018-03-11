#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
# words_file = "word_data_overfit.pkl"
# authors_file = "email_authors_overfit.pkl"
f_words = open(words_file, 'r')
f_authors = open(authors_file, 'r')

word_data = pickle.load(f_words)
authors = pickle.load(f_authors)

f_authors.close()
f_words.close()



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()

features_names = vectorizer.get_feature_names()
# for name in features_names:
#     print name
# quit()
### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
use_small_data = False
if use_small_data:
    features_train = features_train[:150].toarray()
    labels_train   = labels_train[:150]
else:
    features_train = features_train.toarray()
    labels_train   = labels_train

### your code goes here
from  sklearn import  tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

features_importance = clf.feature_importances_
for idx, feature in enumerate(features_importance):
    if feature > 0.2:
        print 'index: %d, feature importance: %s, feature name: %s'%(idx, feature, features_names[idx])

from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, pred)
print acc


