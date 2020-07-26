import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

plt.scatter(X, y, c='purple', s=10)

from sklearn.preprocessing import PolynomialFeatures

plt.scatter(X, y, c='purple', s=10)
x_grid = np.linspace(-3, 3, 12001).reshape(-1, 1)

for i in [1, 2, 12]:
    poly_features = PolynomialFeatures(degree=i, include_bias=False)
    # 由x生成对应多项式x,x^2数组,然后进行线性规划
    X_poly = poly_features.fit_transform(X)

    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    # 截距b, 斜率
    # lin_reg.intercept_, lin_reg.coef_

    plt.plot(x_grid, lin_reg.predict(poly_features.transform(x_grid)))

plt.legend(['1', '2', '12'])

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# 绘制 训练测试损失曲线
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m],
                                               y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")


plt.legend(['train_errors', 'val_errors'])
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)


from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10,
                                         include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(polynomial_regression, X, y)
