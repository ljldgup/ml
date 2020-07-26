from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Transformer 转换器 (StandardScaler，MinMaxScaler)
# Transformer有输入有输出，输出可以放入Transformer或者Estimator 当中作为输入。
ss = StandardScaler()
# fit在计算，transform完成输出
X_train = ss.fit_transform([[1, 2], [4, 3], [7, 6]])
Y_train = [3, 5, 6]

# Estimator 估计器（LinearRegression、LogisticRegression、LASSO、Ridge），所有的机器学习算法模型，都被称为估计器。
lr = LinearRegression()
# 训练模型
lr.fit(X_train, Y_train)
# 模型校验
y_predict = lr.predict([[8, 6]])  # 预测结果

# Pipeline 一个模型列表。

# ss 数据标准化。
# Poly多项式扩展。
# fit_intercept=False 表示截距为0

model = Pipeline([
    ('ss', StandardScaler()),
    ('Poly', PolynomialFeatures(degree=3)),  # 给定多项式扩展操作-3阶扩展
    ('Linear', LinearRegression(fit_intercept=False))
])

d = 2
## 设置多项式的阶乘
model.set_params(Poly__degree=d)
model.fit(X_train, Y_train)

lin = model.get_params()['Linear']
output = u'%d阶，系数为：' % d
## 判断Linear模型中是否有alpha这个参数
print('==', output, lin.coef_.ravel())

from sklearn import preprocessing
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.impute import SimpleImputer

# 填补缺失值
# imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit([[1, 2], [np.nan, 3], [7, 6]])
SimpleImputer()
X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))

# 标准化:将特征数据的分布调整成标准正太分布，也叫高斯分布，也就是使得数据的均值维0，方差为1.
iris = datasets.load_iris()
X = preprocessing.scale(iris.data)

# axis=0 每列平均， axis=0 每行平均
# 每列均值接近于0，方差为1，标准正太分布。。
print(X.mean(axis=0))
print(X.mean(axis=1))
print(X.std(axis=0))

# 分类器encoder
# 独热码，在英文文献中称做 one-hot code, 直观来说就是有多少个状态就有多少比特，而且只有一个比特为1，其他全为0的一种码制。
# 如红色：1 0 0 ，黄色: 0 1 0，蓝色：0 0 1 。如此一来每两个向量之间的距离都是根号2，在向量空间距离都相等，所以这样不会出现偏序性，基本不会影响基于向量空间度量算法的效果。
enc = OneHotEncoder(handle_unknown='ignore')
X = [['Male', 1], ['Female', 3], ['Female', 2]]
OneHotEncoder(handle_unknown='ignore')
enc.fit(X)
enc.categories_
enc.transform([['Female', 1], ['Male', 4]]).toarray()

# 标签分类
le = preprocessing.LabelEncoder()
le.fit(["paris", "seattle", "tokyo", "amsterdam"])
# Transform Categories Into Integers
print(le.transform(["paris", "paris", "tokyo", "amsterdam"]))
# Transform Integers Into Categories
print(le.inverse_transform([1, 2, 0, 2]))

from sklearn import datasets, preprocessing
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print(iris_X[:2, :])
print(iris_y)

# 随机分为训练和测试数据集
X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=0.3)

# k近邻分类
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print(knn.predict(X_test))
print(y_test)

# 生成具有2种属性的300笔数据
X, y = make_classification(
    n_samples=300, n_features=3,
    n_redundant=0, n_informative=2,
    random_state=22, n_clusters_per_class=1,
    scale=100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 向量机分类
clf = SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
print("-----------------------------------------------------------------")

# 标准化
X_scale = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.3)
# 绘图 时用的 coef_ is only available when using a linear kernel
clf = SVC(kernel="linear")
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# 可视化数据
#
plt.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)

# sklearn_y = -(clf.coef_[0][0] * X + clf.intercept_) / clf.coef_[0][1]
# plt.plot(X, sklearn_y, 'b-.', label='sklearn', linewidth=0.3)

plt.show()
