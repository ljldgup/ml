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
df_train_test = pd.concat([df_train_ori.drop(columns='SalePrice'), df_test_ori], ignore_index=True).drop(columns='Id')

'''
# 共有81列，除去价格仍然有80各特征
df_test_ori.columns
df_test_ori.info()

# 画出所有散点图看看是否还有离散点
muti_scatter(df_train_ori.columns, 'SalePrice', df_train_ori)


# 右偏分布（也叫正偏分布） 分位数特征也可以看出来
sns.distplot(df_train_ori['SalePrice'])
df_train_ori['SalePrice'].describe()

# skewness and kurtosis 偏态系数 峰度
# 偏度定义中包括正态分布（偏度=0），右偏分布（也叫正偏分布，其偏度>0），左偏分布（也叫负偏分布，其偏度<0）。
# 峰度包括正态分布（峰度值=3），厚尾（峰度值>3），瘦尾（峰度值<3）
print("Skewness: %f" % df_train_ori['SalePrice'].skew())
print("Kurtosis: %f" % df_train_ori['SalePrice'].kurt())

# id 与售价，明显没有任何关系，可以作为其他的参考
plt.scatter(df_train_ori[numeric_feats[0]], df_train_ori['SalePrice'])

# GrLivArea 和 SalePrice呈线性关系
var = 'GrLivArea'
data = pd.concat([df_train_ori['SalePrice'], df_train_ori[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

var = 'TotalBsmtSF'
data = pd.concat([df_train_ori['SalePrice'], df_train_ori[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

# 线性或指数关系，波动较大
var = 'YearBuilt'
data = pd.concat([df_train_ori['SalePrice'], df_train_ori[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)

# 箱型图 OverallQual 增大带来价格明显提升
# 箱线图的绘制方法是：先找出一组数据的上边缘、下边缘、中位数和两个四分位数；
# 然后， 连接两个四分位数画出箱体；再将上边缘和下边缘与箱体相连接，中位数在箱体中间。
# 上方的点事离群点
var = 'OverallQual'
data = pd.concat([df_train_ori['SalePrice'], df_train_ori[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)

# 两头有一定影响，
var = 'YearBuilt'
data = pd.concat([df_train_ori['SalePrice'], df_train_ori[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)


# correlation matrix 相关性矩阵热力图
# 从热力图中看 'TotalBsmtSF'，'1stFlrSF'相关性非常强，和可能信息是基本相同的，可以选择和 SalePrice 相关性更高的列
# GarageYrBlt GarageCar 也一样
corrmat = df_test_ori.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)

# 与 saleprice 相关性前k个特征热力图, nlargest 返回按某列倒序排列的前k个值，这里去按照SalePrice列的值排列的前k行
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train_ori[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                 xticklabels=cols.values)
plt.show()

# 每队特征散点图
#  'TotalBsmtSF' 'GrLiveArea' 有线性特征
# 'SalePrice' 'YearBuilt' 的关系比箱型图更加明显
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train_ori[cols], size=2.5)
plt.show()
'''

'''
# 对售价进行进一步分析，标准化后，查看前十个和最后十个数据，最后是个数据离散很严重
saleprice_scaled = StandardScaler().fit_transform(df_train_ori['SalePrice'][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

# 观察重要特征的可视化结果
# GrLivArea增大后， 售价开始变得不稳定
var = 'GrLivArea'
data = pd.concat([df_train_ori['SalePrice'], df_train_ori[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
'''

'''
# 没什么问题
var = 'TotalBsmtSF'
data = pd.concat([df_train_ori['SalePrice'], df_train_ori[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

# Histogram 画出峰度，偏度
# probplot 根据分位数据推测其正态分布并与数据画在一起，横坐标是理论分位数，0的地方应该是均值
# 理论分位数应该给出的是20%，40%，60%，80%，100%的分位数，但是100%的分位数一般会无限大，因此在这里需要进行一下 数据处理 ，找一个“近似”的理论样本，来替代“真正”的理论样本
sns.distplot(df_train_ori['SalePrice'], fit=norm)
fig = plt.figure()
# 理论分布和实际分布很明显不吻合
res = stats.probplot(df_train_ori['SalePrice'], plot=plt)


# 对SalePrice取对数后很明显数据和正态分布的吻合度更高，可以将此作为标准化后的数据
# 神经网络的目标不能太大，log效果
df_train_ori['SalePrice'] = np.log(df_train_ori['SalePrice'])

sns.distplot(df_train_ori['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train_ori['SalePrice'], plot=plt)

# 对GrLivArea进行同样的操作
sns.distplot(df_train_ori['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train_ori['GrLivArea'], plot=plt)

df_train_ori['GrLivArea'] = np.log(df_train_ori['GrLivArea'])
sns.distplot(df_train_ori['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train_ori['GrLivArea'], plot=plt)


# HasBsmt 对于=0的值取1，然后再取log
# 这里len(df_train_ori['TotalBsmtSF'])实际上只是这一列的值
df_train_ori['HasBsmt'] = pd.Series(len(df_train_ori['TotalBsmtSF']), index=df_test_ori.index)
df_train_ori['HasBsmt'] = 0
df_test_ori.loc[df_train_ori['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
df_test_ori.loc[df_train_ori['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df_train_ori['TotalBsmtSF'])
'''

'''
sns.distplot(df_train_ori[df_train_ori['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train_ori[df_train_ori['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)

# 进行变换后从三点图上可以看出，线性关系更强了,便于机器学习
plt.scatter(df_train_ori['GrLivArea'], df_train_ori['SalePrice'])
plt.scatter(df_train_ori[df_train_ori['TotalBsmtSF'] > 0]['TotalBsmtSF'], df_train_ori[df_train_ori['TotalBsmtSF'] > 0]['SalePrice'])
'''

'''
# PoolArea与PoolQC箱型图，PoolArea可用于推测PoolQC
fig = sns.boxplot(x='PoolQC', y='PoolArea', data=df_train_test)

fig = sns.boxplot(x='MSZoning', y='MSSubClass', data=df_train_test)
# MSZoning 由 MSSubClass推测
sns.stripplot(x='MSZoning', y='MSSubClass', data=df_train_test)

# LotFrontage LotArea存在关系
sns.scatterplot(x='LotFrontage', y='LotArea', data=df_train_test)

sns.scatterplot(x='FireplaceQu', y='Fireplaces', data=df_train_test)
FireplaceQu
'''
'''
# 可以看到有三类数据极少，且只有在训练集有，可以直接删除
df_train_test['RoofMatl'].value_counts()
df_train_ori['RoofMatl'].value_counts()
'''

'''

# OverallQual很明显在0-5，5-10的情况下表现不一样
sns.boxplot(df_train_ori['OverallQual'], df_train_ori['SalePrice'])
sns.stripplot(df_train_ori['OverallCond'], df_train_ori['SalePrice'])

# OverallCond 同上
sns.boxplot(df_train_ori['OverallCond'], df_train_ori['SalePrice'])
sns.stripplot(df_train_ori['OverallCond'], df_train_ori['SalePrice'])

sns.boxplot(df_train_ori['BsmtCond'], df_train_ori['SalePrice'])

OverallQual在售价取log后效果已经很好,OverallCond仍然一般，可以考虑使用新特征
sns.boxplot(df_train_ori['OverallQual'], df_train_ori['SalePrice'])
sns.boxplot(df_train_ori['OverallQual'], np.log(df_train_ori['SalePrice']))
sns.boxplot(df_train['OverallCond'],np.log(df_train['SalePrice']))
'''

'''
# 缺失数据
all_data_na = (df_train_test.isnull().sum() / len(df_train_test)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='45')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

# 基本同时缺失，可以补成0和zero
df_train_test[['MasVnrType','MasVnrArea']][df_train_test['MasVnrType'].isna()]

#Bsmt的缺失并不一致，考虑统一赋值
df_train_test[bsmt_list][df_train_test[bsmt_list].isna().any(axis=1)]
'''

'''
# 总面积 略小于居住面积
df_train_ori['TotalSF'] = df_train_ori['TotalBsmtSF'] + df_train_ori['1stFlrSF'] + df_train_ori['2ndFlrSF']
sns.scatterplot(df_train_ori['TotalSF'],np.log(df_train_ori['GrLivArea']))
'''