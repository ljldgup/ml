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

# rug生成实例
sns.distplot(x, kde=False, bins=20, rug=True)

# 核密度估计
sns.kdeplot(x)

# bandwith
sns.kdeplot(x)
sns.kdeplot(x, bw=.2, label="bw: 0.2")
sns.kdeplot(x, bw=2, label="bw: 2")
plt.legend()

x = np.random.gamma(6, size=200)
sns.distplot(x, kde=False, fit=stats.gamma)

# 双变量分布
mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])

# 散点图
sns.jointplot(x="x", y="y", data=df)