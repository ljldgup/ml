import math
from functools import reduce

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, skew
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import warnings

from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from tensorflow.keras import layers, models
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from mine.tools import muti_scatter

warnings.filterwarnings("ignore")
'''
过程思路首选观察数据，通过直觉判断哪些列有影响，然后通过散点图，箱型图等确定是否存在明显关系
研究目标变量的分布，峰度，偏度

查看相关性热力图，确定与目标变量相关性较大的变量，同时观察非目标变量中是否存在相关性较大的特征，判断是否应该只保留一个
查看每队因素的散点图，查找是否有忽略的关系。

缺失数据：
缺失较多，缺失数据与目标变量基本无关的直接删除
缺失较少，缺失数据与目标变量基本有关的可以考虑直接删除样本

通过可视化查看离群值，删除明显与趋势不符合的点

对重要的连续值进行线性变化，使其更加符合正态分布
'''

df_train = pd.read_csv("train_housin.csv")

#####################################################################################################
## 可视化了解大体信息
'''
# 共有81列，除去价格仍然有80各特征
df_train.columns
df_train.info()


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

#####################################################################################################
## 处理缺失值，离群值

# 缺失数据统计,聚合操作sum后列转到了索引上
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# 增加没个列的类别数，用来判断填充方法, 注意这里missing_data和df_train.nunique()虽然顺序不同但是index相同，会自动调整
# df_train.nunique()[df_train.isnull().sum()>0].reindex(df_train.isnull().sum().sort_values(ascending=False).index).head(20)
missing_data['classes'] = df_train.nunique()
missing_data['dtypes'] = df_train.dtypes
# 查看前20个缺失列的分类数
missing_data.head(20)
# 通过分类散点图，查看某一分类缺失是否对售价有影响
# fig = sns.stripplot(x='Alley', y="SalePrice", data=df_train.fillna(0))

# 丢弃数据
# 对于缺失较多得，经过分析发现和目标没有太多关系，删除这些列,这里要和test一致
df_train = df_train.drop((missing_data[missing_data['Percent'] > 0.8]).index, axis=1)
# Electrical仅仅缺失1个，删除缺失的数据，这里的loc和直接用中括号是一样的
# df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)


# 填充数据
# FireplaceQu缺失较多单独分一类，其余用众数或中位数
# 效果不如加入众数
# df_train['FireplaceQu'] = df_train['FireplaceQu'].fillna('NA')
for col in missing_data[(missing_data['Percent'] > 0) & (missing_data['Percent'] < 0.5)].index:
    if missing_data.loc[col]['classes'] < 50:
        df_train[col] = df_train[col].fillna(df_train[col].value_counts().index[0])
    else:
        df_train[col] = df_train[col].fillna(df_train[col].median())

# 确定没有缺失
df_train.isnull().sum().max()

#####################################################################################################
## 对数据的分布进行转换，标准化，onehot等
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
cat_nums = pd.DataFrame(df_train.nunique(), columns=['classes'])
most_cat_nums = cat_nums.index.map(lambda c: df_train[c].value_counts().iloc[0]).to_list()
cat_nums['most_cat_nums'] = most_cat_nums
cat_nums['most_cat_pct'] = cat_nums['most_cat_nums'] / len(df_train)
cat_nums['dtype'] = df_train.dtypes
cat_nums.sort_values(by='most_cat_pct', ascending=False).head(30)

# 对比较集中的列进行散点图标识
cat_cols = df_train.columns
# cat_cols = cat_nums[cat_nums['classes'] > 30].sort_values(by='classes', ascending=False).index.to_list()
# cat_cols = cat_nums.sort_values(by='most_cat_pct', ascending=False).head(30).index.to_list()


# 直接删 Utilities，只有一个点不一样
df_train = df_train.drop(columns=['Utilities'])

# 二分类，基本主要都是一个类，另外的类没有规律
binary_col = ['PoolArea', 'Condition2', 'KitchenAbvGr', 'LowQualFinSF', 'MiscVal']
for col in binary_col:
    most_category = df_train[col].value_counts().index[0]
    df_train[col] = df_train[col].map(lambda x: '1' if x == most_category else '0')

# 离群点  ['GarageQual', 'BsmtFinType2', 'OverallCond', 'PoolArea', 'BsmtFinSF1', 'TotalBsmtSF',
#             '1stFlrSF', 'LowQualFinSF', 'BsmtHalfBath', 'Functional', 'LotArea', 'GrLivArea'
#             'Exterior2nd','LotFrontage']
delete_index = []
delete_index.extend(df_train[(df_train['GarageQual'] == 'Ex') & (df_train['SalePrice'] > 400000)].index.to_list())
delete_index.extend(df_train[(df_train['BsmtFinType2'] == 'ALQ') & (df_train['SalePrice'] > 400000)].index.to_list())
delete_index.extend(df_train[(df_train['OverallCond'] == 2) & (df_train['SalePrice'] > 390000)].index.to_list())
delete_index.extend(df_train[(df_train['PoolArea'] == '0') & (df_train['SalePrice'] > 600000)].index.to_list())
delete_index.extend(df_train[(df_train['BsmtFinSF1'] > 2000) & (df_train['SalePrice'] < 200000)].index.to_list())
delete_index.extend(df_train[(df_train['TotalBsmtSF'] > 3000) & (df_train['SalePrice'] < 200000)].index.to_list())
delete_index.extend(df_train[(df_train['1stFlrSF'] > 3000) & (df_train['SalePrice'] < 200000)].index.to_list())
delete_index.extend(df_train[(df_train['LowQualFinSF'] == '0') & (df_train['SalePrice'] > 400000)].index.to_list())
delete_index.extend(df_train[(df_train['BsmtHalfBath'] == 1.0) & (df_train['SalePrice'] > 600000)].index.to_list())
delete_index.extend(df_train[(df_train['Functional'] == 'Mod') & (df_train['SalePrice'] > 400000)].index.to_list())
delete_index.extend(df_train[df_train['LotArea'] > 100000].index.to_list())
delete_index.extend(df_train[(df_train['GrLivArea'] > 4500) & (df_train['SalePrice'] < 200000)].index.to_list())
delete_index.extend(df_train[df_train['LotFrontage'] > 300].index.to_list())
df_train = df_train.drop(index=delete_index)

cat_cols = cat_nums[cat_nums['classes'] < 30].index.to_list()
df_train[cat_cols] = df_train[cat_cols].applymap(str)
# 画出所有散点图看看是否还有离散点
# muti_scatter(cat_cols, 'SalePrice', df_train)

'''
# 对偏度较大的执行log(1+x),使其分布更加正态化
numeric_feats = df_train.dtypes[df_train.dtypes != "object"].index
skewed_feats = df_train[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index.to_list()
df_train[skewed_feats] = np.log1p(df_train[skewed_feats])


sns.distplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)

# 进行变换后从三点图上可以看出，线性关系更强了,便于机器学习
plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
plt.scatter(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF'] > 0]['SalePrice'])

# 这里的原始数据会被后面替换掉，要用这边把后面注释

# 类转换成onehot
df_train_oh = pd.get_dummies(df_train)

input_scaler = StandardScaler()
X = input_scaler.fit_transform(df_train_oh.drop(columns=['SalePrice', 'Id']).values)
y = df_train_oh['SalePrice'].values
target_scaler = StandardScaler()
y = target_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)

#####################################################################################################
## 初步使用各类机器学西进行计算，提取重要的列
def regressors_test(X, y):
    regressors = [LinearSVR(C=1), SVR(kernel="rbf", C=1),
                  KNeighborsRegressor(n_neighbors=6),
                  RandomForestRegressor(),
                  GradientBoostingRegressor()]

    for regressor in regressors:
        print(regressor.__class__)
        regressor.fit(X, y)
        y_pred = regressor.predict(X)
        ans = np.c_[y, y_pred, y_pred - y]
        print(mean_squared_error(ans[:, 0], ans[:, 1]))
        print(cross_val_score(regressor, X, y, cv=3, scoring="neg_mean_squared_error"))
    return regressors


regressors = regressors_test(X, y)
cols = df_train_oh.drop(columns=['SalePrice', 'Id']).columns
corrmat = df_train.corr()
# 统计决策树,提升树，相关性重要性靠前的特征

print(cols[regressors[-1].feature_importances_.argsort()[-10:]])
print(cols[regressors[-2].feature_importances_.argsort()[-10:]])
# 注意这里是倒序，所以1放在后面
print(corrmat.nlargest(10, 'SalePrice')['SalePrice'].index[:1:-1])

# 将前k个重要的列合并
k = 20
cols = cols[regressors[-2].feature_importances_.argsort()[-k:]].to_list() + \
       cols[regressors[-1].feature_importances_.argsort()[-k:]].to_list() + \
       corrmat.nlargest(k + 1, 'SalePrice')['SalePrice'].index.to_list()

# 统计特征出现次数，作为重要度进行排序
cols = pd.Series(cols).value_counts().index.to_list()
cols.remove('SalePrice')

numeric_cols = list(filter(lambda x: '_' not in x, cols))
# onehot,过滤出含有_的列, 在还原列名，再用set去重, 方便后续查看转为set
cat_cols = list(set(
    map(lambda x: x.split('_')[0],
        (filter(lambda x: '_' in x, cols)))))

# 查看分类数
cat_nums = list(zip(cat_cols, map(lambda col: len(df_train[col].unique()), cat_cols)))
# 只有一个分类的可以直接删掉，[('Utilities', 1), ('Street', 1), ('CentralAir', 1)]
one_cat = list(filter(lambda x: x[1] == 1, cat_nums))


for col in ['GarageArea', '1stFlrSF', 'Utilities', 'Street', 'CentralAir']:
    if col in numeric_cols:
        numeric_cols.remove(col)
    elif col in cat_cols:
        cat_cols.remove(col)


# 将对每个类别列filter出对应的onehot列，在reduce
# 多层嵌套的函数式，最好从里向外写
# 这里map返回的是一次性生成器，无法重复用
cat_oh_lists = map(lambda cat: list(filter(lambda x: x.startswith(cat), df_train_oh.columns)), cat_cols)
cat_oh_cols = reduce(lambda cat1, cat2: cat1 + cat2, cat_oh_lists)

train_numeric_x = input_scaler.fit_transform(df_train_oh[numeric_cols].values)
train_onehot_x = df_train_oh[cat_oh_cols]

# np.c_[train_numeric_x,cat_x].shape
X = np.concatenate([train_numeric_x, train_onehot_x], axis=1)

# 通过图示可以看到 'LotArea', 'OpenPorchSF' WoodDeckSF‘存在较大的离群值离群值
# sns.histplot(train_numeric_x.flatten())
# df_train_oh[numeric_cols[5:14]].hist()

# 删除标准化后任然较大的值，这里剔除过多数据会有很大影响，去除少数极值就可以了
index = np.argwhere(abs(X) > 6)
row = index[:, 0]
filter_array = np.full(X.shape[0], True, np.bool)
filter_array[row] = False
X = X[filter_array]
y = y[filter_array]

# regressors = regressors_test(X, y)

#####################################################################################################
## 将测试集处理成和训练集相同的模式
df_test = pd.read_csv('test_housing.csv')

# 有不少列缺失值
# df_test.dtypes[df_test.isna().sum() > 0]
# df_test.isna().sum()[df_test.isna().sum() > 0]
skewed_feats.remove('SalePrice')
df_test[skewed_feats] = np.log1p(df_test[skewed_feats])
df_test = df_test.drop((missing_data[missing_data['Percent'] > 0.8]).index, 1)

# df_train['FireplaceQu'] = df_train['FireplaceQu'].fillna('NA')
# 用众数和中位数填充空值,注意应该和训练集使用的相同，这里统一使用训练集
for col in df_train:
    # NaN补全
    if col in cat_cols:
        df_test[col] = df_test[col].fillna(df_train[col].value_counts().index[0])
    elif col in numeric_cols:
        df_test[col] = df_test[col].fillna(df_train[col].median())

df_test_oh = pd.get_dummies(df_test)
# 测试集某些分类没有，手动补0
for col in df_train_oh.columns.difference(df_test_oh.columns):
    df_test_oh[col] = 0

test_numeric_x = input_scaler.transform(df_test_oh[numeric_cols].values)
test_onehot_x = df_test_oh[cat_oh_cols]
test_X = np.c_[test_numeric_x, test_onehot_x]


#####################################################################################################
## 神经网络模型
def network_without_embedding(input_x):
    model = models.Sequential([
        layers.Input(shape=(input_x.shape[1],)),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1)
    ])
    model.compile(optimizer=RMSprop(lr=1e-4), loss='mse', metrics=['mae'])
    return model


def network_with_embedding(numric_x, label_x):
    embedding_layers = []
    inputs_layers = []
    numeric_input = layers.Input(shape=(numric_x.shape[1],))
    inputs_layers.append(numeric_input)

    for label in label_x:
        t_input = layers.Input(shape=(1,))
        # 嵌入维度取对数, 注意这里的input维度是最大值+1，因为有0
        t_embedding = layers.Embedding(label.max() + 1, 2 * int(math.log2(label.max())) + 1, input_length=1)(t_input)
        t_Flatten = layers.Flatten()(t_embedding)
        inputs_layers.append(t_input)
        embedding_layers.append(t_Flatten)

    numeric_layers = layers.Dense(64)(numeric_input)
    concatenate_layers = layers.concatenate(embedding_layers + [numeric_layers])

    t = layers.Dense(256, activation='relu')(concatenate_layers)
    t = layers.Dense(512, activation='relu')(t)
    t = layers.Dropout(0.1)(t)
    t = layers.Dense(256, activation='relu')(t)
    t = layers.Dropout(0.1)(t)
    t = layers.Dense(64, activation='relu')(t)
    t = layers.Dropout(0.1)(t)
    output = layers.Dense(1)(t)
    model = Model(inputs_layers, output)
    model.compile(optimizer=RMSprop(lr=1e-4), loss='mse', metrics=['mae'])
    return model


def network_kfold(model):
    print('神经网络k折验证')
    kfold = KFold(n_splits=10, shuffle=True, random_state=24)
    for train_idx, test_idx in kfold.split(X, y):
        model.fit(X[train_idx], y[train_idx], epochs=60, batch_size=64, verbose=0)
        # evaluate the model
        scores = model.evaluate(X[test_idx], y[test_idx], verbose=0)
        print("%s: %.2f%" % (model.metrics_names[1], scores[1] * 100))



# 神经网络一起输入，val_loss到达0.09的时候验证集没有在明显的进步，但直接停输出的数据匹配度很低
dense_regressor = network_without_embedding(X)
dense_regressor.fit(X, y, epochs=80, batch_size=64, validation_split=0.20)
ans = dense_regressor.predict(X)
idx = np.random.randint(0,len(X),size=10)
print(np.c_[ans[idx, 0], y[idx]])


# 使用lable + embedding
# embedding 随着训练的增加，训练集loss下降，val的loss反增
# 效果不如普通模型，说明列列之间无明显联系
label_encoders = {}
train_label_x = []
for col in cat_cols:
    label_encoders[col] = LabelEncoder()
    train_label_x.append(label_encoders[col].fit_transform(df_train[col].values))

test_label_x = []
for col in cat_cols:
    test_label_x.append(label_encoders[col].transform(df_test[col].values))

embedding_regressor = network_with_embedding(train_numeric_x, train_label_x)
embedding_regressor.fit([train_numeric_x] + train_label_x, y, epochs=40, batch_size=64, validation_split=0.20)
ans = embedding_regressor.predict([test_numeric_x] + test_label_x)
print(np.c_[ans[:10, 0], y[:10]])


y_pred = dense_regressor.predict(test_X)
test_y = np.exp(target_scaler.inverse_transform(y_pred))
df_test['SalePrice'] = test_y
df_test[['Id', 'SalePrice']].to_csv('submission.csv', index=None)
'''
