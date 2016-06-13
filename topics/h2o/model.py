

import sys
import h2o
import numpy as np

from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

h2o.init(ip='127.0.0.1', port=54321)

##
## Boost
##
#iris_df = h2o.upload_file(path='iris.csv')
#iris_df.describe()
#print iris_df.head()
#gbm_regressor = H2OGradientBoostingEstimator(distribution="gaussian",ntrees=10, max_depth=3, min_rows=2, learn_rate="0.2")
#gbm_regressor.train(x=range(1,iris_df.ncol), y=0, training_frame=iris_df)
#print gbm_regressor
#gbm_classifier = H2OGradientBoostingEstimator(distribution="multinomial",ntrees=10, max_depth=3, min_rows=2, learn_rate="0.2")
#gbm_classifier.train(x=range(0, iris_df.ncol-1), y=iris_df.ncol-1, training_frame=iris_df)
#print gbm_classifier


##
## Linear
##
#prostate_df = h2o.upload_file(path='prostate.csv')
#prostate_df.describe()
#prostate_df['RACE'] = prostate_df['RACE'].asfactor()
#glm_classifier = H2OGeneralizedLinearEstimator(family="binomial", nfolds=10, alpha=0.5)
#glm_classifier.train(x=['AGE', 'RACE', 'PSA', 'DCAPS'], y="CAPSULE", training_frame=prostate_df)


##
## Gird Search
##
iris_df = h2o.upload_file(path='iris.csv')
#ntrees_opt = [5, 10, 15]
#max_depth_opt = [2, 3, 4]
#learn_rate_opt = [0.1, 0.2]

#hyper_parameters = {"ntrees": ntrees_opt, "max_depth":max_depth_opt, "learn_rate":learn_rate_opt}
#from h2o.grid.grid_search import H2OGridSearch

#gs = H2OGridSearch(H2OGradientBoostingEstimator(distribution="multinomial"), hyper_params=hyper_parameters)
#gs.train(x=range(0, iris_df.ncol-1), y=iris_df.ncol-1, training_frame=iris_df, nfold=10)


##
## Pipeline
##

from h2o.transforms.preprocessing import H2OScaler
from h2o.transforms.decomposition import H2OPCA
from sklearn.pipeline import Pipeline

h2o.no_progress()

pipeline = Pipeline([("standardize", H2OScaler()),
                     ("pca", H2OPCA(k=2)),
                     ("gbm", H2OGradientBoostingEstimator(distribution="multinomial"))])

print pipeline.fit(iris_df[:4], iris_df[4])


##
## Randomized Gird Search
##
from sklearn.grid_search import RandomizedSearchCV
from h2o.cross_validation import H2OKFold
from h2o.model.regression import h2o_r2_score
from sklearn.metrics.scorer import make_scorer

params = { "standardize__center": [True, False],
           "standardize__scale": [True, False],
           "pca__k": [2,3],
           "gbm__ntrees": [10,20],
           "gbm__max_depth": [1,2,3],
           "gbm__learn_rate": [0.1,0.2]
}

custom_cv = H2OKFold(iris_df, n_folds=5, seed=42)

pipeline = Pipeline([("standardize", H2OScaler()),
                    ('pca', H2OPCA(k=2)),
                    ('gbm', H2OGradientBoostingEstimator(distribution='gaussian'))])

random_search = RandomizedSearchCV(pipeline, params, n_iter=5,
                                   scoring=make_scorer(h2o_r2_score),
                                   cv=custom_cv,
                                   random_state=42,
                                   n_jobs=1)

random_search.fit(iris_df[1:], iris_df[0])


