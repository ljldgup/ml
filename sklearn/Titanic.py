import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.svm import SVC, LinearSVC

'''
总体思路：查看缺失值，通过可视化确定缺失值是否重要，发现Cabin，Age缺失较多，且于生存相关较多
通过name中的称谓的年龄中位数，补全年龄缺失，cabin有无对生存有明显关系，所以缺失单独一类,将cabin首字母作为cabin等及

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
train_test['Age'] = pd.cut(train_test['Age'], 8, labels=False)
# train_test['Age'] 被分为8个类型
# train_test['Age'].unique()
train_test['Fare'] = pd.cut(train_test['Fare'], 4, labels=False)

train_x = train_test[:len(train)][['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Embarked', 'Cabin', 'title']]
test_x = train_test[len(train):][['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Embarked', 'Cabin', 'title']]
train_y = train['Survived'].values

oh_encoder = OneHotEncoder()

onehot_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Embarked', 'Cabin', 'title']
normal_list = []

onehot_x = oh_encoder.fit_transform(train_x[onehot_cols].values)
classifiers = [LinearSVC(C=1), SVC(kernel="rbf", C=1), KNeighborsClassifier(n_neighbors=5),
               SGDClassifier(random_state=42), RandomForestClassifier(random_state=42),
               GradientBoostingClassifier(), LogisticRegression()]

for classifier in classifiers:
    print(classifier)
    classifier.fit(onehot_x, train_y)
    print(classifier.score(onehot_x, train_y))
    print(cross_val_score(classifier, onehot_x, train_y, cv=2, scoring="accuracy"))
    print('\n--------------------------------------------------------------------------\n\n')