
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from mlxtend.classifier import EnsembleVoteClassifier

import numpy as np

#
# Different classification models
#
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
clf4 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[1,1,1])

print('5-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, clf4], ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Ensemble']):
  scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
  print("Accuracy: %0.2f (+/-) %02.f) [%s]" % (scores.mean(), scores.std(), label))


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mlxtend.evaluate import plot_decision_regions
import itertools

gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(10, 8))
for clf, lab, grd in zip([clf1, clf2, clf3, clf4],
                         ['Logistic Regression', 'Random Forest', 'Navie Bayes', 'Ensemble'],
                         itertools.product([0, 1], repeat=2)):
  clf.fit(X, y)
  ax = plt.subplot(gs[grd[0], grd[1]])
  fig = plot_decision_regions(X=X, y=y, clf=clf)
  plt.title(lab)

plt.show()

#
# Grid search
#

clf4 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='soft')

params = {'logisticregression__C': [1.0, 100.0],
          'randomforestclassifier__n_estimators': [20, 200],}
grid = GridSearchCV(estimator=clf4, param_grid=params, cv=5)
grid.fit(iris.data, iris.target)

for params, mean_score, scores in grid.grid_scores_:
  print("%0.3f (+/- %0.03f for %r" % (mean_score, scores.std() / 2, params))


#
# Majority voting with classifiers trained on different feature subsets
#

from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector

sfs1 = SequentialFeatureSelector(clf1,
                                 k_features=4,
                                 floating=False,
                                 scoring='accuracy',
                                 print_progress=False,
                                 cv=0)
clf1_pipe = Pipeline([('sfs', sfs1),
                      ('logreg', clf1)])

eclf = EnsembleVoteClassifier(clfs=[clf1_pipe, clf2, clf3], voting='soft')

params = {'pipeline__sfs__k_features': [1, 2, 3],
          #'pipeline__logreg__C': [1,0, 100.0],
          'randomforestclassifier__n_estimators': [20, 200]}
grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
grid.fit(iris.data, iris.target)

for  params, mean_score, scores in grid.grid_scores_:
  print("%0.3f (+/-%0.03f) for %r"
        % (mean_score, scores.std()/ 2, params))

print grid.best_params_

eclf = eclf.set_params(**grid.best_params_)
print eclf.fit(X, y).predict(X[[1, 51, 149]])