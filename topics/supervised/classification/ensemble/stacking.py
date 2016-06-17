
from sklearn import datasets
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from mlxtend.classifier import StackingClassifier
from mlxtend.evaluate import plot_decision_regions

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import itertools


iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

# Simple stacked Classification

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                          meta_classifier=lr)

print '3-fold cross validation:\n'

for clf, label in zip([clf1, clf2, clf3, sclf],
                      ['KNN', 'Random Forest', 'Naive Bayes', 'StackingClassifier']):
  scores = cross_validation.cross_val_score(clf, X, y, cv=3, scoring='accuracy')
  print 'Accuracy: %0.2f (+/- %0.2f [%s]' % (scores.mean(), scores.std(), label)


gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(10, 8))

for clf, lab, grd in zip([clf1, clf2, clf3, sclf],
                         ['KNN', 'Random Forest', 'Naive bayes', 'StackingClassifier'],
                         itertools.product([0, 1], repeat=2)):
  clf.fit(X, y)
  ax = plt.subplot(gs[grd[0], grd[1]])
  fig = plot_decision_regions(X=X, y=y, clf=clf)
  plt.title(lab)

plt.show()

# Stacked classification and grid search

params = {'kneighborsclassifier__n_neighbors': [1, 5],
          'randomforestclassifier__n_estimators': [10, 50],
          'meta-logisticregression__C': [0.1, 10.0]}

grid = GridSearchCV(estimator=sclf,
                    param_grid=params,
                    cv=5,
                    refit=True)

grid.fit(X, y)
for params, mean_score, scores in grid.grid_scores_:
  print '%0.3f (+/- %0.03f) for %r' % (mean_score, scores.std() / 2, params)

