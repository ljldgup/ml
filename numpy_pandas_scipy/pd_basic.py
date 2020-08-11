import numpy as np
import matplotlib
import pandas as pd

# 最大显示行数
# pd.options.display.max_rows	= 10

# 读写
df = pd.read_csv(r"../data/000513_d_bas.csv")
# 生成默认列
pd.DataFrame(np.random.rand(100, 3))
pd.read_csv(r"../data/000513_d_bas.csv", nrows=5)
# 逐块迭代读取，返回生成器
pd.read_csv(r"../data/000513_d_bas.csv", chunksize=10)

df.to_csv("1.csv")

# 不给列名，列名和index一样默认123.。
df = pd.DataFrame(np.random.rand(100, 3))
# 使用np生成,注意columns=，pandas很多函数参数很多，需要指定参数名
df = pd.DataFrame(np.arange(30).reshape(6, 5), columns=['A', 'B', 'C', 'D', 'E'])
# 生成Series，注意series是一列,不是一行，这里的ABC65是index
sdata = pd.Series({'A': 11, 'B': 22, 'C': 33, 6: 44, 5: 55})
# 也可以这样
pd.Series([11, 22, 33, 44, 55], ['A', 'B', 'C', 6, 5])

# D项为NaN
pd.DataFrame(sdata, index=['A', 'B', 'C', 'D'])

# 两种基本类型，DataFrame，Series
type(df)
type(df['A'])
df['A'].dtype
type(sdata)

# series可以看做是一个字典， index是key
5 in sdata
1 in df['E']

# 基本组成:索引,值
df.values
df.index
df['A'].values
df['A'].index
# df['A'] 会发生变化
df['A'].values[0] = 1

# 基本信息
df.info()
# 数据信息
df.describe()

# 默认按索引删除
df.drop(1)
# 按列删除axis=1，pandas中axis=1对应列
df.drop('A', axis=1)
# inplace 为真改变原来的数组
df.drop('A', axis=1, inplace=True)

# 转字典 可已选择'list', 'split', 'records', 'index'等
src = df.to_dict('dict')

# 使用字典生成,并重排列顺序
order = ['E', 'B', 'C', 'D', 'A']
pd.DataFrame(src)[order]

# 设置索引
df.index
df = df.set_index('A')
df.reindex(index=['B', 'A'])

# 根据index获取
df.take([5, 3, 4, 2])

sdata.index
sdata[['C', 'A']]
sdata[[1, 2]]
# 这就会错。。需要用iloc
# sdata[[5, 6]]
# 索引可直接设置，并且重复
sdata.index = range(5)
sdata['aa']

# 自定义索引
df = pd.DataFrame([1, 2, 3, 4, 5], columns=['test'], index=['a', 'b', 'c', 'd', 'e'])
df['test']['a']
'a' in df['test']

# 注意index是根据输入参数，筛选重排数据，不是重新生成index
df.reindex(index=['a', 'd', 'e', 'b', 'c'])

# 列重命名
df.rename(index=str, columns={"A": "A-", "C": "-C"})
# 列重排


# 提取行
print(df['A'])
print(df['A'])

# 提取列,切片
df[1:2]
df['A'][1::1]
df['A'][1:]
df[['A', 'B']][1:2]

# 多列切片loc 用于标签， iloc则用于整数位置，这两个用于弥补[]加index引起的歧义
df.loc[1:4, ['A', 'B']]
df.iloc[1:3, [1, 2]]
df.iloc[1:2, 1:4]

# 头尾
df.head(2)
df.tail(2)

# 类似numpy 矩阵
df > 1
df['A'] > 1

# 使用一列筛选整个数据
df[df['A'] > 1]
df.query(r'A > 1')

# 条件与或，注意要加括号
df[(df['A'] > 2) & (df['B'] < 10)]
df[(df['A'] > 2) | (df['B'] < 10)]

# 范围帅选
df.isin([1, 2, 3])

# unique类似python set
df['A'].unique()

# 排序
df.sort_values(by='A', ascending=False)
df.sort_index(ascending=False)
# 最大最小索引
df.idxmax()
df.idxmin()

# 相加,这里可以发现series的index实际上匹配的dataframe的columns
# 相加后自动扩充了index，并且非共有的值为NA
df + sdata
df + df
df + (df + sdata)

# 采样
df.sample()

# 层次化索引
df1 = pd.DataFrame(np.random.randint(80, 120, size=(2, 4)),
                   index=['girl', 'boy'],
                   columns=[['English', 'English', 'Chinese', 'Chinese'],
                            ['like', 'dislike', 'like', 'dislike']])
# 把columns往index上移动
df1.stack()
df1.stack(0)
df1.stack().stack()

df1.unstack()

# 这里的形状6=3*2,4=2*2
df2 = pd.DataFrame(np.random.randint(80, 120, size=(6, 4)),
                   index=pd.MultiIndex.from_product([[1, 2, 3], ['girl', 'boy']]),
                   columns=pd.MultiIndex.from_product([['English', 'Chinese'],
                                                       ['Y', 'N']]))

df2.columns.names = ['Language', 'Pass']  # 设置列索引名
df2.index.names = ['Class', 'Sex']  # 设置行索引名

df2.stack('Language')
df2.unstack('Class')

df2.swaplevel('Sex', 'Class')
df2.sort_index(level=1)
df2.sum(level='Language', axis=1)

df = pd.DataFrame(np.random.randint(20, size=(6, 3)), columns=['A', 'B', 'C'])
df.pivot('A', 'B')
df.set_index('A').unstack('A')
pd.melt(df.set_index('A').unstack('A'), ['A'])

t = df[df['A'] > 10]
t['A'].sort_values()
t['A'].sort_index()
# 注意修改了t['A'].values, 整个t中的A列都会变，而df不会变
t['A'].values.sort()
