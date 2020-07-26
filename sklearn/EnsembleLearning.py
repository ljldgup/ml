from sklearn.datasets import make_moons, load_iris
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

X, y = make_moons(n_samples=1000, noise=0.15)
X_train, X_test, y_train, y_test = X[:800], X[800:], y[:800], y[800:]

voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
                              voting='hard')

voting_clf.fit(X_train, y_train)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            max_samples=100, bootstrap=True, n_jobs=-1)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

# 提升算法
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5)
ada_clf.fit(X_train, y_train)

# 随机森林，根据特征被叶节点使用的情况来给出打分
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import matplotlib.pyplot as plt

m = 200
X = 12 * np.random.rand(m, 1) - 6
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
rand_index = np.random.randint(0, 200, 100)
# 部分参与训练
x_train = X[rand_index, :]
y_train = y[rand_index, :]
# 如果选取数据偏向某一方，会导致严重的过拟合，泛化能力很差，如下数据x>0时没有预测能力
# x_train = X[X<0].reshape(-1,1)
# y_train = y[X<0].reshape(-1,1)
# 调整max_depth能得到不同的结果
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(x_train, y_train)
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(x_train, y_train)
x = np.linspace(-6, 6, 121).reshape(-1, 1)
y_pred1 = gbrt.predict(x)
y_pred2 = tree_reg.predict(x)
plt.scatter(X, y, c='purple', s=4)
plt.plot(x, y_pred1, c='green')
plt.plot(x, y_pred2, c='red')
plt.legend(['GradientBoostingRegressor','decision tree',  'data_set'])
