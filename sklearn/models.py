from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

#Transformer 转换器 (StandardScaler，MinMaxScaler)
#Transformer有输入有输出，输出可以放入Transformer或者Estimator 当中作为输入。
ss = StandardScaler()
# fit在计算，transform完成输出
X_train = ss.fit_transform([[1, 2], [4, 3], [7, 6]])
Y_train = [3, 5, 6]

#Estimator 估计器（LinearRegression、LogisticRegression、LASSO、Ridge），所有的机器学习算法模型，都被称为估计器。
lr = LinearRegression()
#训练模型
lr.fit(X_train, Y_train)
#模型校验
y_predict = lr.predict([[8, 6]]) #预测结果

#Pipeline 一个模型列表。

#ss 数据标准化。
#Poly多项式扩展。
#fit_intercept=False 表示截距为0

model = Pipeline([
            ('ss',StandardScaler()),
            ('Poly',PolynomialFeatures(degree=3)),#给定多项式扩展操作-3阶扩展
            ('Linear',LinearRegression(fit_intercept=False))
        ])

d = 2
## 设置多项式的阶乘
model.set_params(Poly__degree = d)
model.fit(X_train, Y_train)

lin = model.get_params()['Linear']
output = u'%d阶，系数为：' % d
## 判断Linear模型中是否有alpha这个参数
print ('==', output, lin.coef_.ravel())


