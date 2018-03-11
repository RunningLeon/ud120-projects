#!/usr/bin/python

import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score



from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

### check data
# print type(features_train), type(labels_train)
# print len(features_train)
# print features_train[0], labels_train[0]
# quit()

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
# print type(features_train)
# quit()

# print performance of one specific algorithm
def perform(classifier, algori):
    classifier = classifier.fit(features_train, labels_train)
    pred = classifier.predict(features_test)
    acc = accuracy_score(labels_test, pred)
    print "Acc for {} is {}".format(algori, acc)
    try:
        prettyPicture(classifier, features_test, labels_test)
        print "Done !"
    except NameError:
        pass
        print "Error!"

### fine tune hyper-params using grid search
def my_grid_search(clf_obj=None, tuned_params=None, cv_n=5):
    scores = ['precision', 'recall']
    for score in scores:
        print 'Tune hyper-param for %s'%score
        clf = GridSearchCV(clf_obj, tuned_params, cv=cv_n, scoring='%s_macro'%score)
        clf.fit(features_train, labels_train)
        print 'Best param set:'
        print(clf.best_params_)
        print 'Grid scores on validation set:'
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print '%0.3f (+/-%0.03f) for %r'%(mean, std*2, params)

        print 'Detailed classification report'
        y_true, y_pred = labels_test, clf.predict(features_test)
        print classification_report(y_true, y_pred)

### k nearest neighbors algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
clf = KNeighborsClassifier(n_neighbors=3)
perform(clf, 'KNN')

### randomforest classifier
from sklearn.ensemble import RandomForestClassifier
import  math
sqrt_n_features = int(math.sqrt(len(features_train[0])))

params = [{'n_estimators':[i for i in range(5,50)],
           'min_samples_split':[i for i in range(2,10)],
           'max_features':[sqrt_n_features]}]
# my_grid_search(RandomForestClassifier(), params)
# best estimator
clf = RandomForestClassifier(n_estimators=15, max_depth=None, min_samples_split=9, max_features=1)
scores =  cross_val_score(clf, features_train, labels_train, cv=5)
print 'cross validation mean acc is {}, std is {}'.format(scores.mean(), scores.std()*2)
# perform(clf, 'RandomForest')


### Adaboost
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=8, random_state=123)
perform(clf, 'AdaBosst')

### Gradient boosted regression tree
from sklearn.ensemble import  GradientBoostingClassifier
tuned_params = [{'n_estimators':range(2, 100),
                 'learning_rate':[i/10. for i in range(1, 10)]}]

clf = GradientBoostingClassifier(n_estimators=34, learning_rate=0.1, random_state=123)
perform(clf, 'GBRT')


# try:
#     prettyPicture(clf, features_test, labels_test)
#     print "Done !"
# except NameError:
#     pass
#     print "Error!"
