

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import datasets
from mlxtend.evaluate import plot_decision_regions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
clf4 = SVC()

iris = datasets.load_iris()
X = iris.data[:, [0,2]]
y = iris.target

gs = gridspec.GridSpec(2, 2)

fig = plt.figure(figsize=(10, 8))

for clf, lab, grd in zip([clf1, clf2, clf3, clf4],
                         ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'SVM'],
                         itertools.product([0, 1], repeat=2)):
  clf.fit(X, y)
  ax = plt.subplot(gs[grd[0], grd[1]])
  fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
  plt.title(lab)

plt.show()


