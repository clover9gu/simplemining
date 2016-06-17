
import numpy as np
import pandas as pd
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt

from pandas.tools.plotting import radviz

def v_columns(sdf):
  sorted_columns = sorted([c for c in sdf.columns])
  print 'Column count: %s' % len(sorted_columns)
  for c in sorted_columns:
    print c


def v_target_percent(sdf, target='TARGET'):
  df = pd.DataFrame(sdf[target].value_counts())
  df['Percentage'] = 100 * df[target] / sdf.shape[0]
  print df

def v_value_count(sdf, feature, top_count=100000, excluded_value=None):
  if excluded_value:
    print sdf.loc[~np.isclose(sdf[feature], excluded_value), feature].value_counts()[:top_count]
  else:
    print sdf[feature].value_counts()[:top_count]

def v_feature_hist(sdf, feature, bins, title='', xlabel='', ylabel=''):
  sdf[feature].hist(bins=bins)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()

# styles: 'h': plt.hist 'p': sns.kdeplot
def v_feature_hist_bytarget(sdf, feature, target, style='h', title='', xlabel='', ylabel=''):
  styles = {
    'h': plt.hist,
    'p': sns.kdeplot
  }
  sns.FacetGrid(sdf, hue=target, size=6) \
    .map(styles[style], feature) \
    .add_legend()
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()

def echo(e):
  return e

def v_hist(sdf, feature, bins, func=echo, excluded_value=None):
  if excluded_value:
    sdf.loc[~np.isclose(sdf[feature], excluded_value), feature].map(func).hist(bins=bins);
  else:
    sdf[feature].map(func).hist(bins=bins)

def v_interaction(sdf, featureA, featureB, target):
  sns.FacetGrid(sdf, hue=target, size=10) \
   .map(plt.scatter, featureA, featureB) \
   .add_legend()
  plt.show()

def v_pair(sdf, features, target):
  sns.pairplot(sdf[features], hue=target, size=2, diag_kind="kde")

def v_box(sdf, features, target):
  sdf[features].boxplot(by=target, figsize=(12, 6))


# A final multivariate visualization technique pandas has is radviz
# Which puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted
# by the relative value for that feature
def v_radviz(sdf, features, target):
  radviz(sdf[features], target)


def select_features(X, y):
  from sklearn.feature_selection import SelectPercentile
  from sklearn.feature_selection import f_classif,chi2
  from sklearn.preprocessing import Binarizer, scale

  # First select features based on chi2 and f_classif
  p = 3

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
  return features

def v_correlations(sdf, features, threshold=0.7):

  attrs = sdf[features].corr()

  # only important correlations and not auto-correlations
  important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0]) \
      .unstack().dropna().to_dict()
  unique_important_corrs = pd.DataFrame(
      list(set([(tuple(sorted(key)), important_corrs[key]) \
      for key in important_corrs])), columns=['attribute pair', 'correlation'])
  # sorted by absolute value
  unique_important_corrs = unique_important_corrs.ix[
      abs(unique_important_corrs['correlation']).argsort()[::-1]]
  return unique_important_corrs
