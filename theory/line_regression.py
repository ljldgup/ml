import numpy as np
import matplotlib.pyplot as plt

# 通过最小二乘法求mse最小的解
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 为b添加系数1
X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance

# 参数理论解为（xT*x)-1 *xT * y
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

# 等效
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
lin_reg.predict(X_new)

theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
theta_best = np.linalg.pinv(X_b).dot(y)

# batch Gradient Descent 整个batch梯度下降
eta = 0.1  # learning rate
n_iterations = 1000
m = 100
bgd_theta = np.random.randn(2, 1)  # random initialization
bgd_ans = [bgd_theta]
for iteration in range(n_iterations):
    # 对1/m*(xθ-y)^2求导θ  ->  2/m*x(xθ-y)
    gradients = 2 / m * X_b.T.dot(X_b.dot(bgd_theta) - y)
    bgd_theta = bgd_theta - eta * gradients
    bgd_ans.append(bgd_theta)
bgd_ans = np.array(bgd_ans)

# Stochastic Gradient Descent
# 随机梯度下降，随机选取一个样本计算
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters


# 模拟退火算法，逐渐减小学习率
def learning_schedule(t):
    return t0 / (t + t1)


sgd_theta = np.random.randn(2, 1)
sgd_ans = [sgd_theta]
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(sgd_theta) - yi)
        eta = learning_schedule(epoch * m + i)
        sgd_theta = sgd_theta - eta * gradients
        sgd_ans.append(sgd_theta)
sgd_ans = np.array(sgd_ans)

# Mini-batch Gradient Descent
# 随机梯度下降，随机选取一个样本计算
n_epochs = 50
batch_size = 32
t0, t1 = 5, 50  # learning schedule hyperparameters


# 模拟退火算法，逐渐减小学习率
def learning_schedule(t):
    return t0 / (t + t1)


mini_bgd_theta = np.random.randn(2, 1)
mini_bgd_ans = [mini_bgd_theta]
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + batch_size]
        yi = y[random_index:random_index + batch_size]
        # 梯度一定要取均值，不然会得到极其大的值
        gradients = 2 / batch_size * xi.T.dot(xi.dot(mini_bgd_theta) - yi)
        eta = learning_schedule(epoch * m + i)
        mini_bgd_theta = mini_bgd_theta - eta * gradients
        mini_bgd_ans.append(mini_bgd_theta)

mini_bgd_ans = np.array(mini_bgd_ans)

# Batch GD最平稳
plt.clf()
plt.plot(bgd_ans[:, 0, 0], bgd_ans[:, 1, 0], "r-", c='red', label='bgd')
plt.plot(sgd_ans[:, 0, 0], sgd_ans[:, 1, 0], "r-", c='blue', label='sgd')
plt.plot(mini_bgd_ans[:, 0, 0], mini_bgd_ans[:, 1, 0], "r-", c='green', label='mini_bgd')
plt.legend()
