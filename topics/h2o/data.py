
import sys
import h2o
import numpy as np

h2o.init(ip='192.168.0.107', port=54321)

data = zip(*(
  (1,2,3),
  ('a', 'b', 'c'),
  (0.1, 0.2, 0.3)))

data = [
  (1, 'a', 0.1),
  (2, 'b', 0.2),
  (3, 'c', 0.3)
]

data = {
   'A': [1, 2, 3],
   'B': ['a', 'b', 'c'],
   'C': [0.1, 0.2, 0.3]
}

#df  = h2o.H2OFrame(data)

data = [
  [1, 1, 1, 1],
  [2, 2, 2, 2],
  [3, 3, 3, 3],
  [4, 4, 4, 4]
]

data = np.random.randn(100, 4).tolist()

df  = h2o.H2OFrame.from_python(data, column_names=list('ABCD'))

##
## Viewing Data
##

print df.head()

print df.columns

print df.tail(5)

print df.describe()

##
## Selection
##

print df['A']

print df[0]

print df[['B', 'C']]

print df[0:2]

print df[2:7, :] # 5 rows and 4 columns

print df[df['B'] == 'a', :]

##
## Missing data
##

df = h2o.H2OFrame.from_python(
  { 'A': [1, 2, 3, None, ''],
    'B': ['a', 'a', 'b', 'NA', 'NA'],
    'C': ['hello', 'all', 'world', None, None],
    'D': ['12MAR2015:11:00:00', None,
          '13MAR2015:12:00:00', None,
          '14MAR2015:13:00:00']},
  column_types=['numeric', 'enum', 'string', 'time'])

print df.types

print df['D'].day()
print df['D'].dayOfWeek()

sys.exit()

print df["A"].isna()


# ?? change missing


##
## Operations
##

print df['A'].mean()
print df['A'].mean(na_rm=True)

data = np.random.randn(100, 4).tolist()
df  = h2o.H2OFrame.from_python(data, column_names=list('ABCD'))

print df.apply(lambda x: x.mean(na_rm=True))
print df.apply(lambda x: x.sum(), axis=1)

df = h2o.H2OFrame(
  [[e] for e in np.random.randint(0, 7, size=100).tolist()]
)
print df
print df.hist(plot=False)


df = h2o.H2OFrame.from_python(
  [['Hello'], ['World'], ['Welcome'], ['To'], ['H2O'], ['World']]
)
print df.countmatches('l')
print df.sub('l', 'x') # replace
print df.strsplit('(1)+')


##
## Merging
##

df1 = h2o.H2OFrame.from_python(np.random.rand(100, 4).tolist(), column_names=list('ABCD'))
df2 = h2o.H2OFrame.from_python(np.random.rand(100, 4).tolist(), column_names=list('ABCD'))

print df1.rbind(df2) # append df2 rows to df1, the columns names and columns types must match


df1 = h2o.H2OFrame.from_python({
  'A': ['Hello', 'World', 'Welcome', 'To', 'H2O', 'World'],
  'n': [1, 2, 3, 4, 5]
})

df2 = h2o.H2OFrame.from_python(
  [[e] for e in np.random.randint(0, 10, size=100)], column_names=['n']
)
print df1.merge(df2) # merging from two frames together by matching column names



##
## Grouping
##
## * splitting the data into groups based on some criteria
## * apply a function to each group independently
## * combining the results into an H2OFrame
##

df = h2o.H2OFrame(
  { 'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
    'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
    'C': np.random.randn(8),
    'D': np.random.randn(8)
    })

df1 = df.group_by('A').sum().frame
df2 = df.group_by(['A', 'B']).sum().frame
print df.merge(df2) # merge the group results



##
## Load and save
##
## Supported format : csv, xls, xlst, svmlite
##

# h20.upload_file('$file') # file in h2o machine
# h2o.import_file('$file') # file in python machine
# h2o.export_file(df, '$file') save file in h2o machine
# h2o.download_csv(df, '$file') save file in python machine
