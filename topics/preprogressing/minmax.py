
import pandas as pd
import numpy as np

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=(range(6)))
s2 = pd.Series([10, 9, 8, 7, 6, 4], index=(range(6)))

# Scaling a pandas dataframe
df = pd.DataFrame(s1, columns=['s1'])
df['s2'] = s2
print df

from mlxtend.preprocessing import minmax_scaling
print minmax_scaling(df, columns=['s1', 's2'])

# Scaling a numpy array
X = np.array([[1, 10], [2, 9], [3, 8], [4, 7], [5, 6], [6, 5]])
print minmax_scaling(X, columns=[0, 1])

