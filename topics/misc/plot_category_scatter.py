
from io import StringIO
from mlxtend.general_plotting import category_scatter
import pandas as pd
import matplotlib.pyplot as plt


csvfile = u"""label,x,y
class1,10.0,8.04
class1,10.5,7.30
class2,8.3,5.5
class2,8.1,5.9
class3,3.5,3.5
class3,3.8,5.1"""

df = pd.read_csv(StringIO(csvfile))

fig = category_scatter(x='x', y='y', label_col='label', data=df, legend_loc='upper left')

plt.show()