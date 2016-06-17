
from __future__ import division

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

import xgboost as xgb
import numpy as np
import pandas as pd

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

# remove constant columns (standard deviation == 0)
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
cols = train.columns
for i in range(len(cols)-1):
    v = train[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(v,train[cols[j]].values):
            remove.append(cols[j])

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)


# Replace -999999 in var3 column with most common value 2
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details

train = train.replace(-999999, 2)

# Replace 9999999999 with NaN
# See https://www.kaggle.com/c/santander-customer-satisfaction/forums/t/19291/data-dictionary/111360#post111360
# training = training.replace(9999999999, np.nan)
# training.dropna(inplace=True)
# Leads to validation_0-auc:0.839577

X = train.iloc[:, :-1]
y = train.TARGET

# Add zeros per row as extra feature
X['n0'] = (X == 0).sum(axis=1)
# # Add log of var38
# X['logvar38'] = X['var38'].map(np.log1p)
# # Encode var36 as category
# X['var36'] = X['var36'].astype('category')
# X = pd.get_dummies(X)


# Add PCA components as features
print 'add pca features'
X_normalized = normalize(X, axis=0)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)
X['PCA1'] = X_pca[:,0]
X['PCA2'] = X_pca[:,1]

#p = 86 # 308 features validation_1-auc:0.848039
#p = 80 # 284 features validation_1-auc:0.848414
#p = 77 # 267 features validation_1-auc:0.848000
p = 75 # 261 features validation_1-auc:0.848642
# p = 73 # 257 features validation_1-auc:0.848338
# p = 70 # 259 features validation_1-auc:0.848588
# p = 69 # 238 features validation_1-auc:0.848547
# p = 67 # 247 features validation_1-auc:0.847925
# p = 65 # 240 features validation_1-auc:0.846769
# p = 60 # 222 features validation_1-auc:0.848581

# xgboost parameter tuning with p = 75
# recipe: https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/forums/t/19083/best-practices-for-parameter-tuning-on-models/108783#post108783

X_bin = Binarizer().fit_transform(scale(X))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)

chi2_selected = selectChi2.get_support()
chi2_selected_features = [ f for i,f in enumerate(X.columns) if chi2_selected[i]]
print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
   chi2_selected_features))
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
   f_classif_selected_features))
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
features = [ f for f,s in zip(X.columns, selected) if s]
print (features)

X_sel = X[features]

# split data into train and test
print 'split data into train and test ...'
X = train.drop(["TARGET","ID"],axis=1)
#y = train.TARGET.values
X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, stratify=y, random_state=3542)

split_percent = len(X_train) / len(X)
print 'data (train / all) percent: %s' % split_percent

# feature selection
print 'feture selection ...'
clf = DecisionTreeClassifier(random_state=7541, max_features="log2", max_depth=None, min_samples_split=1)
selector = clf.fit(X_train, y_train)

# clf.feature_importances_
fs = SelectFromModel(selector, prefit=True)

print 'transform data select model ...'
X_train = fs.transform(X_train)
X_test = fs.transform(X_test)

ratio = float(np.sum(y == 1)) / np.sum(y==0)
# Initial parameters for the parameter exploration
# clf = xgb.XGBClassifier(missing=9999999999,
#                 max_depth = 10,
#                 n_estimators=1000,
#                 learning_rate=0.1,
#                 nthread=4,
#                 subsample=1.0,
#                 colsample_bytree=0.5,
#                 min_child_weight = 5,
#                 scale_pos_weight = ratio,
#                 seed=4242)

# gives : validation_1-auc:0.845644
# max_depth=8 -> validation_1-auc:0.846341
# max_depth=6 -> validation_1-auc:0.845738
# max_depth=7 -> validation_1-auc:0.846504
# subsample=0.8 -> validation_1-auc:0.844440
# subsample=0.9 -> validation_1-auc:0.844746
# subsample=1.0,  min_child_weight=8 -> validation_1-auc:0.843393
# min_child_weight=3 -> validation_1-auc:0.848534
# min_child_weight=1 -> validation_1-auc:0.846311
# min_child_weight=4 -> validation_1-auc:0.847994
# min_child_weight=2 -> validation_1-auc:0.847934
# min_child_weight=3, colsample_bytree=0.3 -> validation_1-auc:0.847498
# colsample_bytree=0.7 -> validation_1-auc:0.846984
# colsample_bytree=0.6 -> validation_1-auc:0.847856
# colsample_bytree=0.5, learning_rate=0.05 -> validation_1-auc:0.847347
# max_depth=8 -> validation_1-auc:0.847352
# learning_rate = 0.07 -> validation_1-auc:0.847432
# learning_rate = 0.2 -> validation_1-auc:0.846444
# learning_rate = 0.15 -> validation_1-auc:0.846889
# learning_rate = 0.09 -> validation_1-auc:0.846680
# learning_rate = 0.1 -> validation_1-auc:0.847432
# max_depth=7 -> validation_1-auc:0.848534
# learning_rate = 0.05 -> validation_1-auc:0.847347
#

"""
clf = xgb.XGBClassifier(missing=9999999999,
                max_depth = 5,
                n_estimators=1000,
                learning_rate=0.1,
                nthread=4,
                subsample=1.0,
                colsample_bytree=0.5,
                min_child_weight = 3,
                scale_pos_weight = ratio,
                reg_alpha=0.03,
                seed=1301)
"""


# classifier
clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=6744)
# fitting
print 'fit classifier model ...'
clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",  eval_set=[(X_train, y_train), (X_test, y_test)])

y_test_predicted = clf.predict_proba(X_test, ntree_limit=clf.best_iteration)[:,1]
auc = roc_auc_score(y_test, y_test_predicted)

print 'AUC: %s' % auc

# submission
test['n0'] = (test == 0).sum(axis=1)
#test = test.drop(["ID"],axis=1)
# test['logvar38'] = test['var38'].map(np.log1p)
# # Encode var36 as category
# test['var36'] = test['var36'].astype('category')
# test = pd.get_dummies(test)
test_normalized = normalize(test, axis=0)
test_pca = pca.fit_transform(test_normalized)
test['PCA1'] = test_pca[:,0]
test['PCA2'] = test_pca[:,1]
sel_test = test[features]
sel_test = fs.transform(sel_test)
probs = clf.predict_proba(sel_test, ntree_limit=clf.best_iteration)

submission = pd.DataFrame({"ID":test.index, "TARGET":probs[:,1]})
submission.to_csv("submission.csv", index=False)


# plot feature importance
print 'plot feature importance ...'
#mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
mapFeat = dict(zip(features,features))

ts = pd.Series(clf.booster().get_fscore())
#ts.index = ts.reset_index()['index'].map(mapFeat)
ts.sort_values()[-15:].plot(kind="barh", title=("features importance"))

featp = ts.sort_values()[-15:].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)

