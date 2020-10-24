from datetime import datetime

import numpy as np
import pandas as pd
from pandas.tseries.offsets import Hour, Minute, Day, MonthEnd

df = pd.DataFrame(np.random.randint(40, size=(8, 5)), columns=['A', 'B', 'C', 'D', 'E'])
sdata = pd.Series(['AA', 'BB', 'CC', 'DD', 'EE'], [22, 44, 66, 88, 1010])
df_str = pd.DataFrame(
    [['USA', 'Right-handed'],
     ['Japan', 'Left-handed'],
     ['USA', 'Right-handed'],
     ['Japan', 'Right-handed'],
     ['Japan', 'Left-handed'],
     ['Japan', 'Right-handed'],
     ['USA', 'Right-handed'],
     ['USA', 'Left-handed'],
     ['Japan', 'Right-handed'],
     ['USA', 'Right-handed'], ], columns=['Nationality', 'Handedness'])

# list（dataframe）返回列名
list(df_str)

# 可以使用大部分python string内置的函数和正则表达式
sdata.str.lower()
sdata.str.contain('A')
sdata.str.findall('A')
sdata.str.count('A')

# 一列转多列
pd.Series(['1,2', '2,3', '4,5', '5,6']).str.split(',',expand=True)

# 保留两位小数
t = (df / 7)
t.round(decimals=2)

# 和sum，同样的有mean, std, var, median, max, min, nunique, count, mode, prod
# mode返回众数，可能有多个
# prod返回乘积
df.sum()
df.A.sum()
# 直接返回条件的个数
(df > 10).sum()
# 和上面效果一样，<10的变成na，count会忽略na
df[df > 10].count()

# 按行求和
(df > 10).sum(axis=1)

# 各列的相关系数corr函数，1为完全相关
df.corr()

# 对行累加操作
df.cumsum()
df.cummax()
df.cummin()

# 通过拼接df的series得到结果，很适合按列统计,注意这里是nunique 不是unique
pd.concat([df.dtypes, df.nunique(), df.sum()], axis=1, keys=['dtypes', 'types', 'sum'])
# 完全不一样的index都能concat
pd.concat([df.dtypes, df.nunique(), df.cummax()], axis=1, keys=['dtypes', 'types', 'cummax'])

# 变动百分比
df.pct_change()
df['A'].pct_change(1)

# 变动数量，
# 与前1个数据比
df.diff(1)
# 与后1个数据比
df.diff(-1)

# shift整体向后移动一位， df.diff(1) = df - df.shift(1)
df.shift(1)

# 上述不加参数默认为1

# 排序
df.sort_values(by=['A', 'B'])

# series值统计
df['A'].value_counts()

# 这里1位行，0为列
(df > 3).any(1)
(df > 3).any(0)
(df > 1).all(1)
(df > 1).all(0)

# 聚合
df.groupby('A').sum()
df.groupby('A').first()
# ohlc 开盘 最高 最低 收盘
df.groupby('A').ohlc()
# 分位数
df.quantile(0.9)

# 数据转换map 对一列，applymap对所有
df['A'] = df['A'].map(lambda x: x % 2)
df['B'] = df['B'].map(lambda x: x % 3)
# 使用np.vectorize通常比apply，map要快，但切片赋值，数组操作更快
df['B'] = np.vectorize(lambda x: x % 2)(df['B'])

# 使用np.where 代替apply，map
df['C'] = np.where((df['C'] == 0) & (df['C'] == 1), 1, 0)

formater = '{:.2f}'.format
df = df.applymap(formater)
# apply主要用于聚合运算,对dataframe操作
df.apply(np.sum, axis=1)
# 也可以对行直接操作，用于一些复杂映射
df[['A', 'B']].apply(lambda col: (col[0] + col[1]) / 10, axis=1)

df1 = pd.DataFrame(np.arange(30).reshape(6, 5), columns=['A', 'B', 'C', 'D', 'E'])
df2 = pd.DataFrame(np.arange(0, 60, 2).reshape(6, 5), columns=['A', 'B', 'C', 'D', 'E'])
# 取交集：
pd.merge(df1, df2, on=['A', ])
# 取并集：
pd.merge(df1, df2, on=['A', ], how='outer')

# rolling 正值默认窗口往前
# 创建日期序列
index = pd.date_range('2019-01-01', periods=20)
data = pd.DataFrame(np.arange(len(index)), index=index, columns=['test'])

# 打乱顺序pd.take
data = data.take(np.random.permutation(20))

# 移动3个值，进行求和
data['sum'] = data['test'].rolling(3).sum()
# 移动3个值，进行求平均数
data['mean'] = data['test'].rolling(3).mean()

# 移动3个值，最小计数为2
data['mean_min_periods_2'] = data['test'].rolling(3, min_periods=2).mean()

# 指数加权平均 默认com α=1/(1+com)
data['ewm_mean'] = data['test'].ewm(3).mean()
# 加权均值，span窗口，衰减 α=2/(span+1)
data['ewm_mean'] = data['test'].ewm(span=30).mean()

# rolling默认窗口向前，无法向后，可以采用rolling + shift的操作来向后
# 这里采用自定义函数求了今天开始向后三天内最大值位置，x是一个ndarray
# 注意这里shift 只需要3-1=2位
data['max_id'] = data['test'].rolling(3).apply(lambda x: x.argmax()).shift(-2)

# 向后rolling， 将series逆向，然后rolling，再将结果逆向
data['test'][::-1].rolling(window=3, min_periods=0).sum()[::-1]

# 排名，默认相等排名相同，
data.rank()
data.rank(method='first')

ages = np.arange(1, 100, 7)
# 将ages平分成5个区间
pd.cut(ages, 5)

# 指定labels
pd.cut(ages, 5, labels=["婴儿", "青年", "中年", "壮年", "老年"])
# 给定区间(0, 5],(5, 20],(20, 30],(30,50],(50,100]
human = pd.cut(ages, [0, 5, 20, 30, 50, 100], labels=["婴儿", "青年", "中年", "壮年", "老年"])
human.categories
human.codes
pd.value_counts(human)
human.value_counts

# 不要区间标签，直接用0-4标识，可以用于数据离散化
pd.cut(ages, 5, labels=False)

# 根据样本分位数划分，数字是比列累加
cuts = pd.qcut(df['A'], [0, 0.1, 0.5, 0.9, 1.])
# 通过cut groupby分组
groups = df.groupby(cuts)
groups.count()

# 修改闭区间
human = pd.cut(ages, [0, 5, 20, 30, 50, 100], labels=["婴儿", "青年", "中年", "壮年", "老年"], right=False)

# 转换成列名-值二列，可用于聚类
pd.melt(df, value_vars=['B', 'C'])

# 虚拟变量 ( Dummy Variables) 又称虚设变量、名义变量或哑变量
# 将列中值变为为独立列，对应列名值的位置取1，其余取0。
# 可以作为onehot编码,但所需内存较大，使用sklearn的onehot编码返回稀疏矩阵，效果更适合
pd.get_dummies(df['A'])
pd.get_dummies(pd.cut(ages, 5))
pd.get_dummies(pd.qcut(ages, [0, 0.1, 0.5, 0.9, 1.]))

# 采用多个列会出现多层次索引
t = df.groupby(['A', 'B']).count()

# 多级索引访问
t.loc['0.00'].loc['0.00']

# pandas会更具列自动绘制多图，返回对应axis
t.plot.bar()

# 这里绘制出来的是按照索引作为x，列作为分类的效果
t.index
t['C'].plot.bar()

# unstack索引放到列上面, stack是逆操作
t.unstack()
t.stack()
t.unstack().index
# 可以通过这种方式绘制和两个列有关的柱状图
t['C'].plot.bar()
t.unstack()['C'].plot.bar()

# 可以对数据直接绘制多图，注意尺寸第一个是行数
pd.DataFrame(np.random.randint(40, size=(500, 8))).hist()

# 使用Handedness做索引
t = df_str.set_index(['Handedness'])
t.unstack()
t.stack()

# crosstab，交叉表是用于统计分组频率的特殊透视表，默认groupby，对连续值可以配合cut，qcut使用
# crosstab + qcut可以到到不同分位的成分变化
pd.crosstab(df_str.Nationality, df_str.Handedness)

# 使用groupby实现crosstab，如果不加一列，这两列groupby就没有可统计内容了，返回空
df_str['sample'] = 1
df_str.groupby(['Handedness', 'Handedness']).sum().unstack()

# 透视表就是将指定原有DataFrame的列分别作为行索引和列索引，然后对指定的列应用聚集函数
# 这里只是将A,B作为index，没有进行聚合
df.pivot_table(index=['A', 'B'])
df.pivot_table(index=['A', 'B']).unstack()
df.pivot_table(index=['A', 'B']).stack()

# groupby的结构
for (k1, k2), group in df.groupby(['A', 'B']):
    print((k1, k2))
print(group)

# 字典化
dict(list(df.groupby(['A', 'B'])))

# 传入函数对index聚合
df.groupby(lambda x: x % 2).count()

# 传入处理过的列，只要index来源一样就行
df.groupby(df['C'].map(float) % 2).count()

# 不使用聚合函数, 不会改变结构， 内部使用pd.contact合并 可以用于一些分块操作
df.groupby(['A', 'B']).apply(lambda x: x.applymap(float))

# 这样使用了会保留A, B
df.groupby(['A', 'B']).apply(lambda x: x.applymap(float).sum())

# 第二种无论时速度快很多，能不用apply就不用apply
df.groupby(['A']).apply(lambda x: x['B'].sum())
df.groupby(['A'])['B'].sum()

# 与apply类似， transform的函数会返回Series，但是结果必须与输⼊⼤⼩相同
groups.transform(lambda x: x.mean())

# mode 众数
groups.transform(lambda x: x.mode())

df.groupby(['A', 'B'])['C'].quantile(0.9)

# 时间偏移
pd.date_range('2000-01-01', '2000-01-03', freq=Hour(12))
Hour(2) + Minute(30)
ts = pd.Series(np.random.randn(5), pd.date_range('2000-01-01', '2000-01-03', freq=Hour(12)))

# 注意这里直接移动的是作为时间的index
ts.shift(2, freq=Hour(12))

now = datetime(2020, 3, 1)
now + 3 * Day()
now + MonthEnd()
now + MonthEnd(2)

index = pd.date_range('2000-01-01', '2000-03-03', freq=Day(4))
offset = MonthEnd()
ts = pd.Series(np.random.randn(len(index)), index)
ts.groupby(offset.rollforward).count()
ts.groupby(offset.rollback).count()

# 按频率提取
ts.asfreq('W', how='start')
ts.asfreq('W', how='end')

# 根据时间区间进行重采样，提取平均
ts.resample('M')
ts.resample('M').mean()
ts.sample(5)
