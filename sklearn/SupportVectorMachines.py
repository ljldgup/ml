import numpy as np
from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC, SVC, SVR
import matplotlib.pyplot as plt

iris = datasets.load_iris()
# 注意这个写法 适应元组来取两个轴上的数据
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64)  # Iris virginica
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
])
svm_clf.fit(X, y)

# 使用多项式的拟合,次数最大为3
X, y = make_moons(n_samples=100, noise=0.15)
polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge"))
])
polynomial_svm_clf.fit(X, y)

# 直接使用svc，多项式的参数
poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])
poly_kernel_svm_clf.fit(X, y)

# 高斯核函数
rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    # 这里的gamma 和高斯核的宽度有关，gamma越大，支持向量越少，gamma值越小，支持向量越多。
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
])
rbf_kernel_svm_clf.fit(X, y)

# svm 回归预测
m = 200
X = 12 * np.random.rand(m, 1) - 6
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
rand_index = np.random.randint(0, 200, 100)
# 部分参与训练
x_train = X[rand_index, :]
y_train = y[rand_index, :]
# 如果选取数据偏向某一方，会导致严重的过拟合，泛化能力很差，如下数据效果极差，x>0时没有预测能力
# x_train = X[X < 0].reshape(-1, 1)
# y_train = y[X < 0].reshape(-1, 1)

# 调整 kernel, degree能到不同的结果,总体而言数据正常是rbf核效果更好
svm_poly_reg1 = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg2 = SVR(kernel="poly", degree=2, C=10, epsilon=1)
svm_poly_reg3 = SVR(kernel="poly", degree=4, C=10, epsilon=0.1)
svm_poly_reg4 = SVR(kernel="rbf", C=100, epsilon=0.1)
x = np.linspace(-6, 6, 121).reshape(-1, 1)

svm_poly_reg1.fit(x_train, y_train)
y_pred1 = svm_poly_reg1.predict(x)
svm_poly_reg2.fit(x_train, y_train)
y_pred2 = svm_poly_reg2.predict(x)
svm_poly_reg3.fit(x_train, y_train)
y_pred3 = svm_poly_reg3.predict(x)
svm_poly_reg4.fit(x_train, y_train)
y_pred4 = svm_poly_reg4.predict(x)
plt.scatter(X, y, c='purple', s=4)
plt.plot(x, y_pred1, c='green')
plt.plot(x, y_pred2, c='red')
plt.plot(x, y_pred3, c='yellow')
plt.plot(x, y_pred4, c='blue')
