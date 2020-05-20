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

#随机分为训练和测试数据集
X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=0.3)

#k近邻分类
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print(knn.predict(X_test))
print(y_test)



#生成具有2种属性的300笔数据
X, y = make_classification(
    n_samples=300, n_features=3,
    n_redundant=0, n_informative=2,
    random_state=22, n_clusters_per_class=1,
    scale=100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#向量机分类
clf = SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
print("-----------------------------------------------------------------")

#标准化
X_scale = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.3)
#绘图 时用的 coef_ is only available when using a linear kernel
clf = SVC(kernel="linear")
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

#可视化数据
#
plt.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)

#sklearn_y = -(clf.coef_[0][0] * X + clf.intercept_) / clf.coef_[0][1]
#plt.plot(X, sklearn_y, 'b-.', label='sklearn', linewidth=0.3)

plt.show()
