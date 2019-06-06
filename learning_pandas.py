import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Series object
s = pd.Series([1, 3, 5, np.nan, 6, 8])  # default index from 0 to 5
print(s)

# Series with index and values
s = pd.Series(data=[1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print(s)

# get the list of index
print(s.index)
# get the list of values
print(s.values)

# dataframe object

df2 = pd.DataFrame({'A': 1.,  # normal number data
                    'B': pd.Timestamp('20180412'),  # time Stamp data
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),  # Series object
                    'D': np.array([3] * 4, dtype='int32'),  # array object
                    'E': pd.Categorical(['test', 'train', 'test', 'train']),  # category
                    'F': 'foo'})  # string object
print(df2)

# create different time index
dates = pd.date_range('20130101', periods=6)  # create a list of time Stamp data
print(dates)

# create dataframe with data, index and columns
df = pd.DataFrame(np.random.randn(6, 4),  # np array as data
                  index=dates,  # time Stamp list as index
                  columns=list('ABCD'))  # String as column (list can create list from string)

# get data type in every column
print(df2.dtypes)

# get the data from head
print(df.head())  # with default settings to return head with 5 lines
# get the data from tail
print(df.tail(3))

# get the index of dataframe
print(df.index)
# get the column of dataframe
print(df.columns)
# get all the data of dataframe
print(df.values)

# get the description of dataframe
print(df.describe([]))  # std means 标准差

# the average of lines of columns
print(df.mean(0))  # average of every column
print(df.mean(1))  # average of every line

# transpose the dataframe
print(df.T)

# sorted by column value
print(df.sort_index(axis=1, ascending=False))  # axis=1 according to column axis=0 according to index
# ascending control A-Z or Z-A

# sorted by values in specific column
print(df.sort_values(by='B'))

# select the data

# select according to the column or index
print(df['A'])
print(df[0: 3])  # consider dataframe as a matrix
print(df['20130102': '20130104'])

# select according to key value
print(df.loc[dates[0]])
print(df.loc[:, ['A', 'B']])
print(df.loc['20130102': '20130104', ['A', 'B']]) # select according to specific rules

# select according to position
print(df.iloc[3])
print(df.iloc[3:5, 0:2])
print(df.iloc[[1, 2, 4], [0, 2]])

# select by boolean values
print(df[df.A > 0])  # print lines that value in A is larger than 0

df2 = df.copy()
df2['E'] = ['one', 'one','two','three','four','three']
print(df2)
print(df2[df2['E'].isin(['two', 'four'])])  # according to the range

# modify data

# use series to assign lines
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130101', periods=6))
df['F'] = s1  # assign the values with same index
print(df)

# select and assign
df.at[dates[0], 'A'] = 0
df2 = df.copy()
df2[df2 > 0] = 0
print(df2)

# dealing with the NaN

# delete lines with NaN
df.dropna(how='any')

# fill NaN with values
df.fillna(value=5)

# judge where is NaN
pd.isnull(df)

# apply function to elements
print(df.apply(lambda x: x.max() - x.min()))

# histgraph
s = pd.Series(np.random.randint(0, 7, size=10))
print(s.value_counts()) # calculate total count of each number

# string method
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
print(s.str.lower())  # change str into lowercase

# comcat the dataframe
df = pd.DataFrame(np.random.randn(10, 4))
print(df)
pieces = [df[:3], df[3: 7], df[: 7]]
print(pd.concat(pieces))
# pd.concat(objs, axis=) axis=0 纵向合并 axis=1 横向合并(叠汉堡)

# append 根据column来合并 横向纵向都扩充 用NaN来填充空位
df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
print(df)
s = df.iloc[3]
df.append(s, ignore_index=True)

# 读取存储
df.to_csv('foo.csv')  # 写入文件
df.read_csv('foo.csv')  # 读取文件

df.to_excel('foo.xlsx', sheet_name='Sheet1')
df.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])

