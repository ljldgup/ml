import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVC, LinearSVC
from scipy import sparse

'''
查看缺失值，通过可视化确定缺失值是否重要，发现Cabin，Age缺失较多，且于生存相关较多
通过name中的称谓的年龄中位数，补全年龄缺失，cabin有无对生存有明显关系，所以缺失单独一类,将cabin首字母作为cabin等及
Embarked,Fare 缺失较少，且与其他项目没有明显关系， Embarked为离散值填充众数，Fare为连续值填充中位数
Parch，SibSp直系旁系亲属，对生存影响非常相似，将其合并为FamilySize，后续可以考虑加权重

'''
train = pd.read_csv('train_titanic.csv')
test = pd.read_csv('test_titanic.csv')

# 产看类型
train.dtypes.sort_values()
train.select_dtypes(include='int64').head()
train.select_dtypes(include='float64').head()
train.select_dtypes(include='object').head()
# 查看训练集，测试集nan的比列，注意这里用lamdba筛选
# 这里发现主要是Cabin 和 Ageq缺失，Embarked少量缺失
# isna 和 isnull 效果相同
train.isnull().sum()[lambda x: x > 0] / len(train)
test.isnull().sum()[lambda x: x > 0] / len(test)

# info也可以看到有多少个非空数据
train.info()
train.describe()
'''
# Cabin缺失，绘图发现有无cabin对是否幸存有很大影响，所以后面将的缺失值划为单独的一类
train['Cabin_isna'] = pd.isna(train['Cabin'])
# 透视表就是将指定原有DataFrame的列分别作为行索引和列索引，然后对指定的列应用聚集函数
pd.pivot_table(train, index=['Cabin_isna'], values=['Survived']).plot.bar(figsize=(8, 5))
plt.title('Survival Rate')

# 男性女性生存几率
# crosstab，交叉表是用于统计分组频率的特殊透视表
# stacked=True 将同一行堆叠在一列上
pd.crosstab(train.Sex, train.Survived).plot.bar(stacked=True, figsize=(8, 5), color=['#4169E1', '#FF00FF'])
plt.xticks(rotation=0, size='large')
plt.legend(bbox_to_anchor=(0.55, 0.9))

# 各年龄段人数和生存率的关系
pd.crosstab(train['Age'], train.Survived)
human = pd.cut(train['Age'], [0, 10, 20, 30, 40, 50, 60, 70, 80])
t = pd.crosstab(human, train.Survived)
t['survival_rate'] = t[1] / (t[1] + t[0]) * 100
ax1 = t[[0, 1]].plot.bar(stacked=True, figsize=(8, 5), color=['#4169E1', '#FF00FF'])
# 防止图例重合
ax1.legend(loc='upper left')
ax2 = ax1.twinx()
t['survival_rate'].plot.line(ax=ax2, figsize=(8, 5), color='red')
ax2.legend(loc='upper right')
# 绘制折现后，x范围会减小，需要增大
ax1.set_xlim(-1.5, 8)

# 客舱等级（Pclass）
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
Sex1 = ['male', 'female']
for i, ax in zip(Sex1, axes):
    for j, pp in zip(range(1, 4), ax):
        PclassSex = train[(train.Sex == i) & (train.Pclass == j)]['Survived'].value_counts().sort_index(ascending=False)
        pp.bar(range(len(PclassSex)), PclassSex, label=(i, 'Class' + str(j)))
        pp.set_xticks((0, 1))
        pp.set_xticklabels(('Survived', 'Dead'))
        pp.legend(bbox_to_anchor=(0.6, 1.1))

le = LabelEncoder()
for col in train.columns:
    print('-----------------------------------------------------------------------')
    print(str(col), train[col].corr(train['Survived']))
    print('nan ', end='')
    print(pd.isna(train[col]).corr(train['Survived']))
    for item in train[col].unique():
        if not pd.isna(item):
            print(str(item), ' ', end='')
            print((train[col] == item).corr(train['Survived']))

# 对票价进行四分位分割，可以看到票价越高生存率越高
pd.crosstab(pd.qcut(train['Fare'],4),train['Survived']).plot.bar(stacked=True)

# 票价越高pclass高等及占比越高
pd.crosstab(pd.qcut(train['Fare'],4),train['Pclass']).plot.bar(stacked=True)

# Parch（直系亲人）和SibSp（旁系亲人)对生存率的影响
pd.crosstab(train['Parch'],train['Survived']).plot.bar(stacked=True)
pd.crosstab(train['SibSp'],train['Survived']).plot.bar(stacked=True)
'''

# 同时对train和test中的空值进行处理
train_test = pd.concat([train.drop(['PassengerId', 'Survived'], axis=1), test.drop(['PassengerId'], axis=1)])

# 从Name中提取出title,可以根据title对年龄进行补全
train_test['title'] = train_test['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
# 主要是Mr,Miss,Mrs,Master，将其他替换成rare
train_test['title'].value_counts()
train_test['title'] = train_test['title'].map(lambda x: x if x in {'Mr', 'Miss', 'Mrs', 'Master'} else 'rare')

# 查看各个title年龄的中位数
median = train_test['Age'].groupby(train_test['title']).median()

for title, age in median.iteritems():
    print(age, title)
    # 这里的inplace=true无法成功train[train['title'] == title]返回的是一个副本， 所以train_test['Age'][train_test['title'] == title] =
    train_test['Age'][train_test['title'] == title] = train_test['Age'][train_test['title'] == title].fillna(age)


def newage(cols):
    age = cols[0]
    title = cols[1]
    if pd.isnull(age):
        return median[title]
    return age


# 通过apply使用中位数替代nan的年龄，同样是在副本上设置再赋值
# train_test['Age'] = train_test[['Age', 'title']].apply(newage, axis=1)

# Embarked缺失比较小直接填充众数,这里Embarked在索引上
train_test['Embarked'] = train_test['Embarked'].fillna(train_test['Embarked'].value_counts().index[0])

# 票价填充中为数
train_test['Fare'] = train_test['Fare'].fillna(train_test['Fare'].median())

# nan 填充完毕
train_test.info()

# SibSp和Parch两个特征 可视化几乎相同，合并为FamilySize
train_test['FamilySize'] = train_test['Parch'] + train_test['SibSp']

# 将cabin缺失单独列为一类，其余保留首字母
pd.isna(train['Cabin'])
train_test['Cabin'] = train_test['Cabin'].fillna('U')
train_test['Cabin'] = train_test['Cabin'].map(lambda x: x[0])

# 离散化成四个区间
train_test['Age_cat'] = pd.cut(train_test['Age'], 8, labels=False)
# train_test['Age'] 被分为8个类型
# train_test['Age'].unique()
train_test['Fare_cat'] = pd.cut(train_test['Fare'], 4, labels=False)

# 使用onehot相比使用original普通标签，效果更好包括
oh_encoder = OneHotEncoder()
st_scaler = StandardScaler()
# StandardScaler 可以对多列特征操作，LabelEncoder 不行， 需要用OrdinalEncoder
o_encoder = OrdinalEncoder()

# 标签列
label_cols = ['Pclass', 'Sex', 'Embarked', 'Cabin', 'title']
# 需要标准化的列
numeric_cols = ['Age', 'Fare', 'FamilySize']


def all_onehot():
    train_df = train_test[:len(train)][
        ['Pclass', 'Sex', 'Age_cat', 'Fare_cat', 'FamilySize', 'Embarked', 'Cabin', 'title']]
    test_df = train_test[len(train):][
        ['Pclass', 'Sex', 'Age_cat', 'Fare_cat', 'FamilySize', 'Embarked', 'Cabin', 'title']]

    train_x = oh_encoder.fit_transform(train_df.values)
    test_x = oh_encoder.transform(test_df.values)
    return train_x, test_x


def label_and_values():
    # 注意先对整个训练测试集合fit
    train_df = train_test[:len(train)][['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Embarked', 'Cabin', 'title']]
    test_df = train_test[len(train):][['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Embarked', 'Cabin', 'title']]

    X_1 = o_encoder.fit_transform(train_df[label_cols].values)
    X_2 = st_scaler.fit_transform(train_df[numeric_cols].values)
    train_x = np.concatenate([X_1, X_2], axis=1)

    X_1 = o_encoder.transform(test_df[label_cols].values)
    X_2 = st_scaler.transform(test_df[numeric_cols].values)
    test_x = np.concatenate([X_1, X_2], axis=1)
    return train_x, test_x


def onehot_and_values():
    train_df = train_test[:len(train)][['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Embarked', 'Cabin', 'title']]
    test_df = train_test[len(train):][['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Embarked', 'Cabin', 'title']]

    X_1 = oh_encoder.fit_transform(train_df[label_cols].values)
    X_2 = st_scaler.fit_transform(train_df[numeric_cols].values)
    # 稀疏矩阵和普通矩阵相连
    train_x = sparse.hstack([X_1, X_2])

    X_1 = oh_encoder.transform(test_df[label_cols].values)
    X_2 = st_scaler.transform(test_df[numeric_cols].values)
    test_x = sparse.hstack([X_1, X_2])
    return train_x, test_x


train_y = train['Survived']
train_x, test_x = onehot_and_values()
# train_x, test_x = label_and_values()
# train_x, test_x = onehot_and_values()
# 对于非交叉交叉
# 前四个分类器使用onehot有明显提升，knn，随机森林，提升树无变化
# knn是距离敏感的，应该在内部做了onehot处理，不然不可能不变
# 使用label+连续数值后随机森林和提升树，knn有提升

# 交叉验证中使用onehot 线性svm总体得分最高，
classifiers = [SGDClassifier(random_state=42), LogisticRegression(), LinearSVC(C=1), SVC(kernel="rbf", C=1),
               KNeighborsClassifier(n_neighbors=6),
               RandomForestClassifier(),
               # 这里调整最大深度后，精度会提高，但交叉验证变差了
               GradientBoostingClassifier()]

for classifier in classifiers:
    print(classifier)
    classifier.fit(train_x, train_y)
    print(classifier.score(train_x, train_y))
    print(cross_val_score(classifier, train_x, train_y, cv=2, scoring="accuracy"))
    print('\n--------------------------------------------------------------------------\n\n')

# 注意这里不必再fit

classifiers[3].predict(test_x)


# cross_val_score 和 直接fit返回的效果不一样
classifiers[-1].fit(train_x, train_y)
test['Survived'] = classifiers[-1].predict(test_x)

# 目前label_and_values + GradientBoostingClassifier得分最高 77.99
# mine 提交格式不用index
test[['PassengerId', 'Survived']].to_csv('submission.csv', index=None)
