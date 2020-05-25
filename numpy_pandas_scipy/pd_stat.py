from datetime import datetime

import numpy as np
import matplotlib
import pandas as pd
from pandas.tseries.offsets import Hour, Minute, Day, MonthEnd

df = pd.DataFrame(np.random.randint(6, size=(8, 5)), columns=['A', 'B', 'C', 'D', 'E'])

# 和sum，同样的有mean, std, var, median, max, min, nunique, count, mode, prod
# mode返回中众数，可能有多个
# prod返回乘积
df.sum()
df.A.sum()

# 各列的相关系数corr函数，1为完全相关
df.corr()

# 对行累加操作
df.cumsum()
df.cummax()
df.cummin()

# 变动百分比
df.pct_change()
df.A.pct_change(1)

# 变动数量，
# 与前1个数据比
df.diff(1)
# 与后1个数据比
df.diff(-1)

# shift整体向后移动一位， df.diff(1) = df - df.shift(1)
df.shift(1)

# 上述不加参数默认为1


# 聚合
df.groupby('A').sum()

# 数据转换map 对一列，applymap对所有
df['A'] = df['A'].map(lambda x: x % 2)
df['B'] = df['B'].map(lambda x: x % 3)
formater = '{:.2f}'.format
df = df.applymap(formater)
# apply主要用于聚合运算
df.apply(np.sum, axis=1)

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

# 打乱顺序
data = data.take(np.random.permutation(20))

# 移动3个值，进行求和
data['sum'] = data['test'].rolling(3).sum()
# 移动3个值，进行求平均数
data['mean'] = data['test'].rolling(3).mean()
# 移动3个值，最小计数为2
data['mean_min_periods_2'] = data['test'].rolling(3, min_periods=2).mean()
# 加权均值
data['ewm_mean'] = data['test'].ewm(span=30).mean()
# rolling默认窗口向前，无法向后，可以采用rolling + shift的操作来向后
# 这里采用自定义函数求了今天开始向后三天内最大值位置，x是一个ndarray
# 注意这里shift 只需要3-1=2位
data['max_id'] = data['test'].rolling(3).apply(lambda x: x.argmax()).shift(-2)

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

# 修改闭区间
human = pd.cut(ages, [0, 5, 20, 30, 50, 100], labels=["婴儿", "青年", "中年", "壮年", "老年"], right=False)

# 根据样本分位数划分，数字是比列累加
human = pd.qcut(ages, [0, 0.1, 0.5, 0.9, 1.])

# 虚拟变量 ( Dummy Variables) 又称虚设变量、名义变量或哑变量
# 用以反映质的属性的一个人工变量，是量化了的自变量，通常取值为0或1。
pd.get_dummies(df['A'])
pd.get_dummies(pd.cut(ages, 5))
pd.get_dummies(pd.qcut(ages, [0, 0.1, 0.5, 0.9, 1.]))

# 聚合
# 传入函数对index聚合
df.groupby(lambda x: x % 2).count()

# 传入处理过的列，只要index来源一样就行
df.groupby(df['C'].map(float) % 2).count()

# 采用多个列会出现多层次索引
t = df.groupby(['A', 'B']).count()

# 多级索引访问
t.loc['0.00'].loc['0.00']

# 这里绘制出来的是按照索引作为x，列作为分类的效果
t.index
t['C'].plot.bar()

# unstack 后部分索引放到列上面,重新绘制，可以通过这种方式绘制和两个列有关的柱状图
t.unstack().index
t.unstack()['C'].plot.bar()

# 不使用聚合函数, 不会改变结构， 内部使用pd.contact合并 可以用于一些分块操作
df.groupby(['A', 'B']).apply(lambda x: x.applymap(float))

# 这样使用了会保留A, B
df.groupby(['A', 'B']).apply(lambda x: x.applymap(float).sum())

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

# 重采样，提取平均
ts.resample('M').mean()
