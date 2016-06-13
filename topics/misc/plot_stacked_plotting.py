
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.general_plotting import stacked_barplot


data = [
  [1.0, 2.0, 3.0, 4.0],
  [1.4, 2.1, 2.9, 5.1],
  [1.9, 2.2, 3.5, 4.1],
  [1.4, 2.5, 3.5, 4.2]
]

df = pd.DataFrame(data, columns=['X1', 'X2', 'X3', 'X4'])
df.columns = ['X1', 'X2', 'X3', 'X4']
df.index = ['Sample1', 'Sample2', 'Sample3', 'Sample4']

fig = stacked_barplot(df, rotation=45, legend_loc='best')

plt.show()
