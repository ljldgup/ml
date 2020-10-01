import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, skew
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, MinMaxScaler
from scipy import stats

from mine.tools import muti_scatter

df_train_ori = pd.read_csv("train_housin.csv")
df_test_ori = pd.read_csv('test_housing.csv')

'''
# 共有81列，除去价格仍然有80各特征
df_train.columns
df_train.info()

# 画出所有散点图看看是否还有离散点
muti_scatter(df_train, 'SalePrice', df_train)


# 右偏分布（也叫正偏分布） 分位数特征也可以看出来
sns.distplot(df_train['SalePrice'])
df_train['SalePrice'].describe()

# skewness and kurtosis 偏态系数 峰度
# 偏度定义中包括正态分布（偏度=0），右偏分布（也叫正偏分布，其偏度>0），左偏分布（也叫负偏分布，其偏度<0）。
# 峰度包括正态分布（峰度值=3），厚尾（峰度值>3），瘦尾（峰度值<3）
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

# id 与售价，明显没有任何关系，可以作为其他的参考
plt.scatter(df_train[numeric_feats[0]], df_train['SalePrice'])

# GrLivArea 和 SalePrice呈线性关系
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

# 线性或指数关系，波动较大
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)

# 箱型图 OverallQual 增大带来价格明显提升
# 箱线图的绘制方法是：先找出一组数据的上边缘、下边缘、中位数和两个四分位数；
# 然后， 连接两个四分位数画出箱体；再将上边缘和下边缘与箱体相连接，中位数在箱体中间。
# 上方的点事离群点
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)

# 两头有一定影响，
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)


# correlation matrix 相关性矩阵热力图
# 从热力图中看 'TotalBsmtSF'，'1stFlrSF'相关性非常强，和可能信息是基本相同的，可以选择和 SalePrice 相关性更高的列
# GarageYrBlt GarageCar 也一样
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)

# 与 saleprice 相关性前k个特征热力图, nlargest 返回按某列倒序排列的前k个值，这里去按照SalePrice列的值排列的前k行
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                 xticklabels=cols.values)
plt.show()

# 每队特征散点图
#  'TotalBsmtSF' 'GrLiveArea' 有线性特征
# 'SalePrice' 'YearBuilt' 的关系比箱型图更加明显
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size=2.5)
plt.show()
'''


'''
# 对售价进行进一步分析，标准化后，查看前十个和最后十个数据，最后是个数据离散很严重
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

# 观察重要特征的可视化结果
# GrLivArea增大后， 售价开始变得不稳定
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
'''



'''
# 没什么问题
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

# Histogram 画出峰度，偏度
# probplot 根据分位数据推测其正态分布并与数据画在一起，横坐标是理论分位数，0的地方应该是均值
# 理论分位数应该给出的是20%，40%，60%，80%，100%的分位数，但是100%的分位数一般会无限大，因此在这里需要进行一下 数据处理 ，找一个“近似”的理论样本，来替代“真正”的理论样本
sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
# 理论分布和实际分布很明显不吻合
res = stats.probplot(df_train['SalePrice'], plot=plt)


# 对SalePrice取对数后很明显数据和正态分布的吻合度更高，可以将此作为标准化后的数据
# 神经网络的目标不能太大，log效果
df_train['SalePrice'] = np.log(df_train['SalePrice'])

sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)

# 对GrLivArea进行同样的操作
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)


# HasBsmt 对于=0的值取1，然后再取log
# 这里len(df_train['TotalBsmtSF'])实际上只是这一列的值
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
df_train.loc[df_train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
'''

'''
sns.distplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)

# 进行变换后从三点图上可以看出，线性关系更强了,便于机器学习
plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
plt.scatter(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF'] > 0]['SalePrice'])
'''