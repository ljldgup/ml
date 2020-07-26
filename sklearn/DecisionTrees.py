from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

import numpy as np

iris = load_iris()
X = iris.data[:, 2:]  # petal length and width
y = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

with open('./wine.dot', 'w', encoding='utf-8') as f:
    export_graphviz(
        tree_clf,
        out_file=f,
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

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
tree_reg = DecisionTreeRegressor(max_depth=4)
tree_reg.fit(x_train, y_train)
x = np.linspace(-6, 6, 121).reshape(-1, 1)
y_pred = tree_reg.predict(x)
plt.scatter(X, y, c='purple', s=4)
plt.plot(x, y_pred)
