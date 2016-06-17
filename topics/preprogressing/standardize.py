
import pandas as pd
import numpy as np
from mlxtend.preprocessing import standardize

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=(range(6)))
s2 = pd.Series([10, 9, 8, 7, 6, 5], index=(range(6)))
df = pd.DataFrame(s1, columns=['s1'])
df['s2'] = s2
print df

print standardize(df, columns=['s1', 's2'])


X = np.array([[1, 10], [2, 9], [3, 8], [4, 7], [5, 6], [6, 5]])
print standardize(X, columns=[0, 1])


# re-using parameters

X_train = np.array([[1, 10], [4, 7], [3, 8]])
X_test = np.array([[1, 2], [3, 4], [5, 6]])

X_train_std, params = standardize(X_train, columns=[0, 1], return_params=True)
X_test_std = standardize(X_test, columns=[0, 1], params=params)

print params
print X_train_std
print X_test_std
