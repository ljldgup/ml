import math
from functools import reduce

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, skew
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OrdinalEncoder
from scipy import stats
from scipy.special import boxcox1p
import warnings

'''
from tensorflow.keras import layers, models
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
'''

from mine.tools import muti_scatter, regressors_test, StackingAveragedModels, rmsle_cv,custom_kfold

warnings.filterwarnings("ignore")
'''
过程思路首选观察数据，通过直觉判断哪些列有影响，然后通过散点图，箱型图等确定是否存在明显关系
研究目标变量的分布，峰度，偏度

查看相关性热力图，确定与目标变量相关性较大的变量，同时观察非目标变量中是否存在相关性较大的特征，判断是否应该只保留一个
查看每队因素的散点图，查找是否有忽略的关系。

缺失数据：
缺失较多，缺失数据与目标变量基本无关的直接删除
缺失较少，缺失数据与目标变量基本有关的可以考虑直接删除样本

离散值
通过可视化散点图查看离群值，删除明显与趋势不符合的点
标准化后可以去除少量过大的点

数据
偏度较大的数据可以取对数使其更加符合正态分布，并缩小范围
取对数后不要再进行标准化，效果很差

类别不明显的让决策树自己剪枝，而不是手动搞成onehot
神经网络输入输出直接做标准化，或者归一化
'''

df_train_ori = pd.read_csv("train_housin.csv")
df_test_ori = pd.read_csv('test_housing.csv')
df_test = df_test_ori.copy()
df_train = df_train_ori.copy()

#####################################################################################################
# 处理离群值,去除离群点有明显的提升，删的太多
# 存在离群点的列  ['GarageQual', 'BsmtFinType2', 'OverallCond', 'PoolArea', 'BsmtFinSF1', 'TotalBsmtSF',
#             '1stFlrSF', 'LowQualFinSF', 'BsmtHalfBath', 'Functional', 'LotArea', 'GrLivArea'
#             'Exterior2nd', 'LotFrontage']

delete_index = []

# delete_index.extend(df_train[(df_train['GarageQual'] == 'Ex') & (df_train['SalePrice'] > 400000)].index.to_list())
# delete_index.extend(df_train[(df_train['BsmtFinType2'] == 'ALQ') & (df_train['SalePrice'] > 400000)].index.to_list())
# delete_index.extend(df_train[(df_train['OverallCond'] == 2) & (df_train['SalePrice'] > 390000)].index.to_list())
delete_index.extend(df_train[(df_train['PoolArea'] > 500) & (df_train['SalePrice'] > 600000)].index.to_list())
delete_index.extend(df_train[(df_train['BsmtFinSF1'] > 2000) & (df_train['SalePrice'] < 200000)].index.to_list())
delete_index.extend(df_train[(df_train['TotalBsmtSF'] > 3000) & (df_train['SalePrice'] < 200000)].index.to_list())
delete_index.extend(df_train[(df_train['1stFlrSF'] > 3000) & (df_train['SalePrice'] < 200000)].index.to_list())
# delete_index.extend(df_train[(df_train['LowQualFinSF'] > 500) & (df_train['SalePrice'] > 400000)].index.to_list())
# delete_index.extend(df_train[(df_train['BsmtHalfBath'] == 1.0) & (df_train['SalePrice'] > 600000)].index.to_list())
# delete_index.extend(df_train[(df_train['Functional'] == 'Mod') & (df_train['SalePrice'] > 400000)].index.to_list())
# delete_index.extend(df_train[df_train['LotArea'] > 100000].index.to_list())
delete_index.extend(df_train[df_train['LotFrontage'] > 300].index.to_list())
# delete_index.extend(df_train[(df_train['LotFrontage'] > 150) & (df_train['SalePrice'] < 50000)].index.to_list())
# delete_index.extend(df_train[(df_train['YearRemodAdd'] < 2000) & (df_train['SalePrice'] > 600000)].index.to_list())
delete_index.extend(df_train[(df_train['OpenPorchSF'] > 500) & (df_train['SalePrice'] < 100000)].index.to_list())

delete_index.extend(df_train[(df_train['GrLivArea'] > 4500) & (df_train['SalePrice'] < 300000)].index.to_list())

'''
# 删除部分类别较少的点,仅仅在训练集存在的类别,
# 现在觉得没必要删，容易过拟合
delete_index.extend(df_train[(df_train['RoofMatl'] == 'Membran') | (df_train['RoofMatl'] == 'Metal') | (
        df_train['RoofMatl'] == 'Roll')].index.to_list())
delete_index.extend(df_train[df_train['Electrical'] == 'Mix'].index.to_list())
delete_index.extend(df_train[df_train['KitchenAbvGr'] == 3].index.to_list())

df_train = df_train.drop(index=delete_index)
'''

# 缺失比列较小的直接丢弃样本
# df_train.drop(columns=missing_data[missing_data['Percent'] < 0.5].index)

# 注意要放弃原来的index不然会出问题
df_train_test = pd.concat([df_train.drop(columns='SalePrice'), df_test], ignore_index=True).drop(columns='Id')

# 缺失数据统计,聚合操作sum后列转到了索引上
total = df_train_test.isnull().sum().sort_values(ascending=False)
percent = (df_train_test.isnull().sum() / df_train_test.isnull().count() * 100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# 增加没个列的类别数，用来判断填充方法, 注意这里missing_data和df_train.nunique()虽然顺序不同但是index相同，会自动调整
missing_data['classes'] = df_train_test.nunique()
missing_data['dtypes'] = df_train_test.dtypes
# 查看前20个缺失列的分类数
missing_data.head(20)
'''
# 使用多项式拟合填充LotFrontage
x = df_train_test.loc[df_train_test["LotFrontage"].notnull(), "LotArea"]
y = df_train_test.loc[df_train_test["LotFrontage"].notnull(), "LotFrontage"]
t = (x <= 25000) & (y <= 150)
p = np.polyfit(x[t], y[t], 1)
df_train_test.loc[df_train_test['LotFrontage'].isnull(), 'LotFrontage'] = \
    np.polyval(p, df_train_test.loc[df_train_test['LotFrontage'].isnull(), 'LotArea'])
'''
# 根据Neighborhood中位数填充LotFrontage
df_train_test["LotFrontage"] = df_train_test.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

# 通过阅读数据说明，'PoolQC', 'Fence', 'FireplaceQu', 'MiscFeature', 'Alley'中NA意味着没有,可以补成0，也可以不补
clear_feats = ['PoolQC', 'Fence', 'FireplaceQu', 'MiscFeature', 'Alley']

# bsmt 和 garage都是重要列 NA代表没有，但也有些数据是缺失,手动查看补全
# 直接查看非常困难，保存成csv用excel查看哪些需要补
df_train_test.loc[332, 'BsmtFinType2'] = df_train_test['BsmtFinType2'].mode()[0]
df_train_test.loc[[948, 1487, 2348], 'BsmtExposure'] = df_train_test['BsmtExposure'].mode()[0]
df_train_test.loc[[2040, 2185, 2524], 'BsmtCond'] = df_train_test['BsmtCond'].mode()[0]
df_train_test.loc[[2217, 2218], 'BsmtQual'] = df_train_test['BsmtQual'].mode()[0]
df_train_test.loc[[2120, 2188], ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath']] = 0
bsmt_list = df_train_test.columns[df_train_test.columns.str.startswith('Bsmt')].to_list()
# df_train_test[bsmt_list][df_train_test[bsmt_list].isna().any(axis=1)].to_csv('bsmt.csv')
''
df_train_test.loc[2126, ['GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']] = [
    df_train_test['GarageYrBlt'].median(),
    df_train_test['GarageFinish'].mode()[0],
    df_train_test['GarageQual'].mode()[0],
    df_train_test['GarageCond'].mode()[0]]
df_train_test.loc[2576, ['GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond',
                         'GarageCars', 'GarageArea']] = [
    df_train_test['GarageYrBlt'].median(),
    df_train_test['GarageFinish'].mode()[0],
    df_train_test['GarageQual'].mode()[0],
    df_train_test['GarageCond'].mode()[0],
    df_train_test['GarageYrBlt'].median(),
    df_train_test['GarageArea'].median()]
garage_list = df_train_test.columns[df_train_test.columns.str.startswith('Garage')].to_list()
# df_train_test[garage_list][df_train_test[garage_list].isna().any(axis=1)].to_csv('garage.csv')

# 有一个MasVnrArea在，MasVnrType缺失，补众数
df_train_test.loc[2610, 'MasVnrType'] = df_train_test['MasVnrType'].mode()[0]

clear_feats += bsmt_list + garage_list + ['MasVnrType', 'MasVnrArea']
df_train_test['MasVnrArea'] = df_train_test['MasVnrArea'].fillna(0)

# 这里如果不分类填充，数字会被转为字符
for feat in clear_feats:
    if df_train_test[feat].dtypes == np.object:
        df_train_test[feat] = df_train_test[feat].fillna('None')
    else:
        df_train_test[feat] = df_train_test[feat].fillna(0)

# 根据kaggle说明将部分int类型但是是分类的转成str，方便get_dummy
num_str_list = ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']
df_train_test[num_str_list] = df_train_test[num_str_list].applymap(str)
'''
'''

# 全部填充众数和中位数，效果不如全部填充均值，舍弃类别getdummy NAN所有列均为0
for col in df_train_test.columns:
    if col not in clear_feats:
        if df_train_test[col].dtypes == np.object or col in ['']:
            df_train_test[col] = df_train_test[col].fillna(df_train_test[col].mode()[0])
        else:
            df_train_test[col] = df_train_test[col].fillna(df_train_test[col].mean())

# 确定没有缺失
df_train_test.isnull().sum().sort_values().tail(len(clear_feats))

#####################################################################################################
## 对数据的分布进行转换，标准化，onehot等

cat_nums = pd.DataFrame(df_train_test.nunique(), columns=['classes'])
most_cat_nums = cat_nums.index.map(lambda c: df_train_test[c].value_counts().iloc[0]).to_list()
cat_nums['most_cat_nums'] = most_cat_nums
cat_nums['most_cat_pct'] = cat_nums['most_cat_nums'] / len(df_train_test)
cat_nums['dtype'] = df_train_test.dtypes
cat_nums.sort_values(by='most_cat_pct', ascending=False).head(30)

# 直接删 Utilities，只有一个点不一样
# df_train_test = df_train_test.drop(columns=['Utilities', 'Alley', 'MiscFeature'])
df_train_test = df_train_test.drop(columns=['Utilities'])

# OverallCond 在5前后表现差距很大，基于此增加一个新特征
# df_train_test['OverallCondMine'] = df_train_test['OverallCond'].map(lambda x: 1 if x > 5 else 0)

'''
# 二分类，基本主要都是一个类，另外的类没有规律 , 效果变差
binary_feat = ['PoolArea', 'KitchenAbvGr']
for col in binary_feat:
    most_category = df_train_test[col].value_counts().index[0]
    df_train_test[col] = df_train_test[col].map(lambda x: '1' if x == most_category else '0')
'''

# 新增特征
df_train_test['TotalSF'] = df_train_test['TotalBsmtSF'] + df_train_test['1stFlrSF'] + df_train_test['2ndFlrSF']

numeric_feats = df_train_test.dtypes[df_train_test.dtypes != "object"].index

# 对偏度较大的执行log(1+x)/box-cox,使其分布更加正态化
# 数据标准化，对数等都在get_dummy的结果上做，不影响原数据
skewed_feats = df_train_test[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[abs(skewed_feats) > 0.75]
skewed_feats = skewed_feats.index.to_list()

# df_train_test_oh[skewed_feats] = np.log1p(df_train_test_oh[skewed_feats])
for feat in skewed_feats:
    df_train_test[feat] = boxcox1p(df_train_test[feat], 0.15)

# 将label转为数字，有明显提升
# 按我的理解应该是没有用的，但实际却又明显提升，可能是使用label减小了特征的数量，树自己找分裂点，提高了利用率
# 全部使用label后,指标明显下降，重要指标应该还是用onehot更好，有助于分离，树不需要自己找分裂点

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')

# 通过箱型图查看发现都是波动小的特征，这类特征适合让树自己寻找分裂点
# muti_scatter(df_train_ori.columns, 'SalePrice', df_train_ori,'b')
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(df_train_test[c].values))
    df_train_test[c] = lbl.transform(list(df_train_test[c].values))

# 使用label后,整个数据集得分变高了，但交叉验证变差了
# oe = OrdinalEncoder()
# df_train_test[category_feats] = oe.fit_transform(df_train_test[category_feats].values)
df_train_test_oh = pd.get_dummies(df_train_test)

'''
# skewed_feats 这部分 log之后进行标准化效果反而变差，不再对这部分数据进行标准化
unchanged_numeric_feats = [col for col in numeric_feats if col not in skewed_feats]
input_scaler = StandardScaler()
df_train_test_oh[unchanged_numeric_feats] = input_scaler.fit_transform(df_train_test_oh[unchanged_numeric_feats])
'''

x = df_train_test_oh.values
train_y = np.log(df_train['SalePrice'])
train_x, test_x = x[:len(df_train)], x[len(df_train):]

regressors = regressors_test(train_x, train_y)

##################################################################`###################################
## 根据各类机器学习计算结果，提取重要的列
'''
cols = df_train_test_oh.drop(columns=['SalePrice', 'Id']).columns
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

numeric_feats = list(filter(lambda x: '_' not in x, cols))
# onehot,过滤出含有_的列, 在还原列名，再用set去重, 方便后续查看转为set
cat_feats = list(set(
    map(lambda x: x.split('_')[0],
        (filter(lambda x: '_' in x, cols)))))

# 查看分类数
cat_nums = list(zip(cat_feats, map(lambda col: len(df_train[col].unique()), cat_feats)))
# 只有一个分类的可以直接删掉，[('Utilities', 1), ('Street', 1), ('CentralAir', 1)]
one_cat = list(filter(lambda x: x[1] == 1, cat_nums))

# 将对每个类别列filter出对应的onehot列，在reduce
# 多层嵌套的函数式，最好从里向外写
# 这里map返回的是一次性生成器，无法重复用
cat_oh_lists = map(lambda cat: list(filter(lambda x: x.startswith(cat), df_train_test_oh.columns)), cat_feats)
cat_oh_feats = reduce(lambda cat1, cat2: cat1 + cat2, cat_oh_lists)
'''

#####################################################################################################
## 将测试集处理成和训练集相同的模式
#

'''
# 分开搜索，一次运行太慢
# 先n_estimators 限制max_features，max_depth不然时间会非常长
param_test1 = {'n_estimators': range(100, 3000, 200), ,
               'max_depth': range(4, 17, 3), 'max_features': range(9, 20, 3),
               'min_samples_split': range(50, 200, 45), 'max_leaf_nodes': range(4, 20, 5),'min_samples_leaf':range(10,51,10)}

gsearch1 = GridSearchCV(
    estimator=GradientBoostingRegressor(learning_rate=0.1, max_depth=6,max_features='sqrt'),
    param_grid=param_test1, scoring='neg_mean_squared_error', iid=False, cv=3,verbose=1,n_jobs=2)
gsearch1.fit(train_x, train_y)
print(gsearch1.best_params_, gsearch1.best_score_)
# GradientBoostingRegressor(learning_rate=0.05, n_estimators=900，max_depth=4，max_features=14,max_leaf_nodes=14,min_samples_split=185)
'''

'''
# 这个自定义的模型不能用cross_val_score
# stack 没有用gbt，效果很好
stacked_averaged_models = StackingAveragedModels(
    base_models=(regressors[0], regressors[2], regressors[4], regressors[5]),
    meta_model=regressors[1])
stacked_averaged_models.fit(train_x, train_y)
stacked_pred = stacked_averaged_models.predict(train_x)
print(np.sqrt(mean_squared_error(train_y, stacked_pred)))
custom_kfold(stacked_averaged_models,train_x,train_y, lambda x: np.sqrt(mean_squared_error(x)))
'''

'''
上传效果很糟糕
er = VotingRegressor([('ridge', regressors[0]),
                      ('LinearSVR', regressors[2]),
                      ('SVR', regressors[3]),
                     ('KNN', regressors[4]),
                     ('RF', regressors[5]),
                     ('GBT', regressors[6],('grid_gbt',gsearch1))])
cross_val_score(er, train_x, train_y, cv=3, scoring="neg_mean_squared_error")
'''
# 上传要选择交叉验证得分最高的
test_y = regressors[-1].predict(test_x)
df_test['SalePrice'] = np.exp(test_y)
df_test[['Id', 'SalePrice']].to_csv('submission.csv', index=None)
'''
#####################################################################################################
## 神经网络模型
def network_without_embedding(input_x):
    model = models.Sequential([
        layers.Input(shape=(input_x.shape[1],)),
        layers.Dense(512, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(256, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=RMSprop(lr=1e-3), loss='mse', metrics=['mae'])
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

    t = layers.Dense(1024, activation='relu')(concatenate_layers)
    t = layers.Dense(512, activation='relu')(t)
    t = layers.Dropout(0.2)(t)
    t = layers.Dense(256, activation='relu')(t)
    t = layers.Dropout(0.1)(t)
    t = layers.Dense(128, activation='relu')(t)
    output = layers.Dense(1)(t)
    model = Model(inputs_layers, output)
    model.compile(optimizer=RMSprop(lr=1e-4), loss='mse', metrics=['mae'])
    return model


def network_kfold(model, x, y):
    print('神经网络k折验证')
    kfold = KFold(n_splits=10, shuffle=True, random_state=24)
    for train_idx, test_idx in kfold.split(x, y):
        model.fit(x[train_idx], y[train_idx], epochs=60, batch_size=64, verbose=0)
        # evaluate the model
        scores = model.evaluate(x[test_idx], y[test_idx], verbose=0)
        print("%s: %.2f%" % (model.metrics_names[1], scores[1] * 100))

df_train_test_oh = pd.get_dummies(df_train_test)
numeric_feats = list(filter(lambda x: '_' not in x, df_train_test_oh.columns))
input_scaler = MinMaxScaler()
df_train_test_oh[numeric_feats] = input_scaler.fit_transform(df_train_test_oh[numeric_feats])

x = df_train_test_oh.values
train_x, test_x = x[:len(df_train)], x[len(df_train):]
# 神经网络 y标准化比log效果好得多
target_scaler = StandardScaler()
train_y = target_scaler.fit_transform(df_train['SalePrice'][:, np.newaxis])

# 加深加宽网络无明显作用
# 训练集的损失可以不断减小至0.05左右 val_mae达到最佳0.24左右，之后没有进步
dense_regressor = network_without_embedding(train_x)
# dense_regressor.fit(train_x, train_y, epochs=240, batch_size=128, validation_split=0.1)
dense_regressor.fit(train_x, train_y, epochs=240, batch_size=128)
ans = dense_regressor.predict(train_x)
idx = np.random.randint(0, len(train_x), size=10)
print(np.c_[input_scaler.inverse_transform(ans[:10, 0]), input_scaler.inverse_transform(train_y[:10])])


# 使用lable embedding 效果不如普通的全连接 说明列列之间无明显联系
df_train_test_oh = pd.get_dummies(df_train_test)
numeric_feats = list(filter(lambda c: df_train_test[c].dtype != np.object, df_train_test.columns))
label_feats = list(filter(lambda c: c in numeric_feats, df_train_test.columns))

input_scaler = MinMaxScaler()
numeric_x = input_scaler.fit_transform(df_train_test[numeric_feats])
train_numeric_x, test_numeric_x = numeric_x[:len(df_train)], numeric_x[len(df_train):]

label_endcoder = OrdinalEncoder()
label_x = label_endcoder.fit_transform(df_train_test[label_feats]).astype(np.int32)
train_label_x, test_label_x = label_x[:len(df_train)], label_x[len(df_train):]

target_scaler = StandardScaler()
train_y = input_scaler.fit_transform(df_train['SalePrice'][:, np.newaxis])

# 按列分割成list
col_to_list = lambda array:[array[:, row] for row in range(array.shape[1])]
embedding_regressor = network_with_embedding(train_numeric_x, col_to_list(label_x))
embedding_regressor.fit([train_numeric_x] + col_to_list(train_label_x), train_y, epochs=40, batch_size=64, validation_split=0.20)
ans = embedding_regressor.predict([test_numeric_x] + col_to_list(test_label_x))
print(np.c_[input_scaler.inverse_transform(ans[:10, 0]), input_scaler.inverse_transform(train_y[:10])])

'''

'''
y_pred = dense_regressor.predict(test_x)
test_y = target_scaler.inverse_transform(y_pred)
df_test['SalePrice'] = test_y
df_test[['Id', 'SalePrice']].to_csv('submission.csv', index=None)
'''
