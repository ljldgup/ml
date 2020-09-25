import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)
np.random.seed(sum(map(ord, "distributions")))

# kde表示生成核密度估计
x = np.random.normal(size=100)
sns.distplot(x, kde=True)

# 调节bins，对数据更具体的做分桶操作。
sns.distplot(x, kde=True, bins=20)

# hist直方图,dist是根据密度绘制
# 效果基本一样，但很纵坐标有区别，dist更快
sns.histplot(x, kde=True, bins=20)

# rug生成实例
sns.distplot(x, kde=False, bins=20, rug=True)

# 更具分布类型绘制
sns.distplot(np.random.randn(100), fit=stats.norm)
sns.distplot(np.exp(np.random.randn(100)), fit=stats.norm)

x = np.random.gamma(6, size=200)
sns.distplot(x, kde=False, fit=stats.gamma)

# 核密度估计绘制
sns.kdeplot(x)

# bandwith，用于近似的正态分布曲线的宽度
sns.kdeplot(x)
sns.kdeplot(x, bw=.2, label="bw: 0.2")
sns.kdeplot(x, bw=2, label="bw: 2")
plt.legend()

# 双变量分布
mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])

# 散点图+柱状图
sns.jointplot(x="x", y="y", data=df)

# 六角形箱
x, y = np.random.multivariate_normal(mean, cov, 1000).T
with sns.axes_style("ticks"):
    sns.jointplot(x=x, y=y, kind="hex")

# 核密度估计
sns.jointplot(x="x", y="y", data=df, kind="kde")
f, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(df.x, df.y, ax=ax)
sns.rugplot(df.x, color="g", ax=ax)
sns.rugplot(df.y, vertical=True, ax=ax)

# 散点图 + 线性回归 + 95%置信区间
x = np.random.randint(10, 20, size=(30, 1))
y = x * 2
cond = np.random.randint(10, 20, size=(30, 1)) % 2
data = pd.DataFrame(np.c_[x + np.random.randn(30, 1), y + np.random.randn(30, 1), cond],
                    columns=['x', 'y', 'cond'])
# 属性名 + dataframe
sns.lmplot(x="x", y="y", data=data)
# 根据条件分开拟合
sns.lmplot(x="x", y="y", hue='cond', data=data)
# size控制图片大小
sns.lmplot(x="x", y="y", data=data, size=6)

# 箱型图
x = np.random.randint(10, 20, size=(100, 1))
y = x % 5
data = pd.DataFrame(np.c_[x, y + np.random.randn(100, 1)], columns=['x', 'y'])
sns.boxplot(x="x", y="y", data=data)

# 柱状图,中间黑线误差棒 注明所测量数据的不确定度的大小
sns.set(style="whitegrid", color_codes=True)
titanic = pd.read_csv('../mine/train_titanic.csv')
# 统计柱状图
sns.barplot(x="Sex", y="Survived", hue="Pclass", data=titanic)
sns.barplot(x="Sex", y="Pclass", hue="Survived", data=titanic)
# 灰度柱状图，不要y参数
sns.countplot(x="Sex", hue="Pclass", data=titanic)

# 分类散点图，一维数据是分类数据时，散点图成为了条带形状
sns.stripplot(x="Pclass", y="Fare", data=titanic)
# 去除抖动后是几条线
sns.stripplot(x="Pclass", y="Fare", data=titanic, jitter=False)

# 蜂窝图
sns.swarmplot(x="Pclass", y="Fare", data=titanic)

# 两两作图
sns.pairplot(data=titanic[['Survived', 'Pclass', 'Fare']])

# 热力图
sns.heatmap(titanic.corr(), fmt='.2f', annot=True)
