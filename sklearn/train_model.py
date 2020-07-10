import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

# Batch Gradient Descent,所有数据参与运算
eta = 0.1  # learning rate
n_iterations = 1000
m = 100
theta = np.random.randn(2, 1)  # random initialization
for iteration in range(n_iterations):
    # 这里是二阶范数求导的结果，将x转置提到前方保证特征数就是行数，
    # 这个线性规划共有两个特征，其中偏置b的x始终为1
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

# SGD Stochastic Gradient Descent 每步随机抽取一部分运算
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters


def learning_schedule(t):
    return t0 / (t + t1)


theta = np.random.randn(2, 1)  # random initialization
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        # 模拟退火算法，逐渐减小学习率
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients

# 多项式拟合
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# 这里里把x的数据 改成x,x^2两个维度，然后按照线性拟合来做
poly_features = PolynomialFeatures(degree=6, include_bias=False)
X_poly = poly_features.fit_transform(X)
print(X[1, 0], ' ', X_poly[1, 0], ' ', X_poly[1, 0])

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_

t = np.linspace(X.min(), X.max(), 100)
plt.plot(t, lin_reg.predict(poly_features.fit_transform(t.reshape(*t.shape, 1))))
plt.scatter(X, y, color='red')
plt.show()


# RMSE 验证集均方根误差随着训练集的增大而减小并稳定
# 训练集均方根误差随着训练集的增大而增大并稳定
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")


lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)

polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10,
                                         include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(polynomial_regression, X, y)

ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
plt.plot(t, ridge_reg.predict(t.reshape(*t.shape, 1)))




iris = datasets.load_iris()
list(iris.keys())
# ['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename']
X = iris["data"][:, 3:] # petal width
y = (iris["target"] == 2).astype(np.int) #
# 逻辑回归
log_reg = LogisticRegression()
log_reg.fit(X, y)
# 注意这里reshape(-1, 1) 这届增加了一个维度
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)

plt.plot(X_new, y_proba[:, 1], "g-", label="Iris virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris virginica")
