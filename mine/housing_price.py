import math
from functools import reduce

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, skew
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OrdinalEncoder
from scipy import stats
import warnings

from tensorflow.keras import layers, models
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from mine.tools import muti_scatter, regressors_test

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

df_train_ori = pd.read_csv("train_housin.csv")
df_test = pd.read_csv('test_housing.csv')

df_train = df_train_ori.copy()

#####################################################################################################
# 处理离群值

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

# 缺失比列较小的直接丢弃样本
for col in missing_data[missing_data['Percent'] < 0.01].index:
    df_train.drop(df_train[col].isna().index)

df_train_test = pd.concat([df_train.drop(columns='SalePrice'), df_test]).drop(columns='Id')

# 对于缺失较多得，经过分析发现和目标没有太多关系，删除这些列,这里要和test一致
df_train_test = df_train_test.drop(missing_data[missing_data['Percent'] >= 0.9].index, axis=1)

# 填充众数和中位数
for col in df_train_test.columns:
    if missing_data.loc[col]['classes'] < 50:
        df_train_test[col] = df_train_test[col].fillna(df_train_test[col].value_counts().index[0])
    else:
        df_train_test[col] = df_train_test[col].fillna(df_train_test[col].median())

# 确定没有缺失
df_train_test.isnull().sum().max()

#####################################################################################################
## 对数据的分布进行转换，标准化，onehot等

cat_nums = pd.DataFrame(df_train_test.nunique(), columns=['classes'])
most_cat_nums = cat_nums.index.map(lambda c: df_train_test[c].value_counts().iloc[0]).to_list()
cat_nums['most_cat_nums'] = most_cat_nums
cat_nums['most_cat_pct'] = cat_nums['most_cat_nums'] / len(df_train_test)
cat_nums['dtype'] = df_train_test.dtypes
cat_nums.sort_values(by='most_cat_pct', ascending=False).head(30)

# 直接删 Utilities，只有一个点不一样
df_train_test = df_train_test.drop(columns=['Utilities'])

# 二分类，基本主要都是一个类，另外的类没有规律
binary_col = ['PoolArea', 'Condition2', 'KitchenAbvGr', 'LowQualFinSF', 'MiscVal']
for col in binary_col:
    most_category = df_train_test[col].value_counts().index[0]
    df_train_test[col] = df_train_test[col].map(lambda x: '1' if x == most_category else '0')

# 存在离群点的列  ['GarageQual', 'BsmtFinType2', 'OverallCond', 'PoolArea', 'BsmtFinSF1', 'TotalBsmtSF',
#             '1stFlrSF', 'LowQualFinSF', 'BsmtHalfBath', 'Functional', 'LotArea', 'GrLivArea'
#             'Exterior2nd', 'LotFrontage']


# 将分类较小的int类型数据转为string，get_dummies时能生成onehot
# cat_cols = cat_nums[cat_nums['classes'] < 30].index.intersection(df_train_test.columns).to_list()
# df_train_test[cat_cols] = df_train_test[cat_cols].applymap(str)
'''
# 对偏度较大的执行log(1+x),使其分布更加正态化
df_train_test_oh = pd.get_dummies(df_train_test)
numeric_feats = df_train_test.dtypes[df_train_test.dtypes != "object"].index
skewed_feats = df_train_test[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index.to_list()
# 数据标准化，对数等都在get_dummy的结果上做，不影响原数据
df_train_test_oh[skewed_feats] = np.log1p(df_train_test[skewed_feats])
numeric_cols = list(filter(lambda x: '_' not in x, df_train_test_oh.columns))
unchanged_numeric_cols = [col for col in numeric_cols if col not in skewed_feats]

input_scaler = StandardScaler()
df_train_test_oh[unchanged_numeric_cols] = input_scaler.fit_transform(df_train_test_oh[unchanged_numeric_cols])
# skewed_feats 这部分 log之后进行标准化效果反而变差，不再对这部分数据进行标准化
x = df_train_test_oh.values
train_y = np.log(df_train['SalePrice'])
train_x, test_x = x[:len(df_train)], x[len(df_train):]
regressors = regressors_test(train_x, train_y)


#####################################################################################################
## 根据各类机器学习计算结果，提取重要的列

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

# 将对每个类别列filter出对应的onehot列，在reduce
# 多层嵌套的函数式，最好从里向外写
# 这里map返回的是一次性生成器，无法重复用
cat_oh_lists = map(lambda cat: list(filter(lambda x: x.startswith(cat), df_train_oh.columns)), cat_cols)
cat_oh_cols = reduce(lambda cat1, cat2: cat1 + cat2, cat_oh_lists)
'''
'''
#####################################################################################################
## 将测试集处理成和训练集相同的模式


regressors[-1].fit(train_y, train_x)
y_pred = regressors[-1].predict(train_x)
idx = np.random.randint(0, len(train_y) - 1, size=20)
print(np.c_[np.exp(train_y),
            np.exp(y_pred),
            np.exp(y_pred) - np.exp(train_y)][idx, :])

test_y = regressors[-1].predict(test_x)
df_test['SalePrice'] = np.exp(test_y) - 1
df_test[['Id', 'SalePrice']].to_csv('submission.csv', index=None)

# 可以分开进行，一次运行太慢
param_test1 = {'n_estimators': range(80, 160, 10), 'max_leaf_nodes': range(4, 20, 3),
               'max_depth': range(4, 11, 2), 'min_samples_leaf': range(5, 40, 4),
               'max_features':range(10,30,3)}

gsearch1 = GridSearchCV(
    estimator=GradientBoostingRegressor(learning_rate=0.1, min_samples_leaf=20,
                                        max_features='sqrt', subsample=0.8, random_state=12),
    param_grid=param_test1, scoring='neg_mean_squared_error', iid=False, cv=3,verbose=1)
gsearch1.fit(train_x, train_y)
print(gsearch1.best_params_, gsearch1.best_score_)

'''


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


def network_kfold(model, x, y):
    print('神经网络k折验证')
    kfold = KFold(n_splits=10, shuffle=True, random_state=24)
    for train_idx, test_idx in kfold.split(x, y):
        model.fit(x[train_idx], y[train_idx], epochs=60, batch_size=64, verbose=0)
        # evaluate the model
        scores = model.evaluate(x[test_idx], y[test_idx], verbose=0)
        print("%s: %.2f%" % (model.metrics_names[1], scores[1] * 100))


df_train_test_oh = pd.get_dummies(df_train_test)
numeric_cols = list(filter(lambda x: '_' not in x, df_train_test_oh.columns))
input_scaler = MinMaxScaler()
df_train_test_oh[numeric_cols] = input_scaler.fit_transform(df_train_test_oh[numeric_cols])

x = df_train_test_oh.values
train_x, test_x = x[:len(df_train)], x[len(df_train):]
# 神经网络 y标准化比log效果好得多
input_scaler = StandardScaler()
train_y = input_scaler.fit_transform(df_train['SalePrice'][:, np.newaxis])
# 神经网络一起输入，val_loss到达0.09的时候验证集没有在明显的进步，但直接停输出的数据匹配度很低
dense_regressor = network_without_embedding(train_x)
dense_regressor.fit(train_x, train_y, epochs=80, batch_size=64, validation_split=0.20)
ans = dense_regressor.predict(train_x)
idx = np.random.randint(0, len(train_x), size=10)
print(np.c_[input_scaler.inverse_transform(ans[:10, 0]), input_scaler.inverse_transform(train_y[:10])])
'''
# 使用lable + embedding
# embedding 随着训练的增加，训练集loss下降，val的loss反增
# 效果不如普通模型，说明列列之间无明显联系
df_train_test_oh = pd.get_dummies(df_train_test)
numeric_cols = list(filter(lambda c: df_train_test[c].dtype != np.object, df_train_test.columns))
label_cols = list(filter(lambda c: c in numeric_cols, df_train_test.columns))

input_scaler = MinMaxScaler()
numeric_x = input_scaler.fit_transform(df_train_test[numeric_cols])
train_numeric_x, test_numeric_x = numeric_x[:len(df_train)], numeric_x[len(df_train):]

label_endcoder = OrdinalEncoder()
label_x = label_endcoder.fit_transform(df_train_test[label_cols]).astype(np.int32)
train_label_x, test_label_x = label_x[:len(df_train)], label_x[len(df_train):]

input_scaler = StandardScaler()
train_y = input_scaler.fit_transform(df_train['SalePrice'][:, np.newaxis])

# 按列分割成list
col_to_list = lambda array:[array[:, row] for row in range(array.shape[1])]

embedding_regressor = network_with_embedding(train_numeric_x, col_to_list(label_x))
embedding_regressor.fit([train_numeric_x] + col_to_list(train_label_x), train_y, epochs=40, batch_size=64, validation_split=0.20)
ans = embedding_regressor.predict([test_numeric_x] + col_to_list(test_label_x))
print(np.c_[np.exp(ans[:10, 0]), np.exp(train_y[:10])])
'''
'''
y_pred = dense_regressor.predict(test_x)
test_y = np.exp(target_scaler.inverse_transform(y_pred))
df_test['SalePrice'] = test_y
df_test[['Id', 'SalePrice']].to_csv('submission.csv', index=None)
'''
