import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression, Perceptron

#导入波士顿房价数据集
loaded_data = datasets.load_boston()
#data_X是训练数据
data_X = loaded_data.data
#data_y是导入的标签数据
data_y = loaded_data.target

#建立模型，线性回归模型
model = LinearRegression()

#进行数据拟合，通过训练得到模型参数
model.fit(data_X,data_y)

#使用训练过的模型对数据进行预测，预测是前四行的数据
print(model.predict(data_X[:4,:]))

#斜率
print(model.coef_)

#截距
#等价于model.predict(data_X[:1,:] * 0), 输入为0时的输出
print(model.intercept_)

#取出之前定义的模型的参数
print(model.get_params())

#使用均方误差对其进行打分，输出精确度，
#即利用训练好的模型对data_X进行预测，得到预测后，和原本标签进行比较
print(model.score(data_X,data_y))

#noise 越大点就会越来越离散
X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)

#感知机
clf = LinearRegression()
clf.fit(X, y)

#scatter 散点图
plt.scatter(X, y)

#拟合直线
sklearn_y = -(clf.coef_[0] * X + clf.intercept_)
plt.plot(X, sklearn_y, 'b-.', label='sklearn', linewidth=0.3)
plt.show()
