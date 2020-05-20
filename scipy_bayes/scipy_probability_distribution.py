import scipy.stats as st
import numpy as np
from matplotlib import pyplot as plt
'''
rvs：产生服从指定分布的随机数
pdf：概率密度函数
cdf：累计分布函数
sf：残存函数（1-CDF）
ppf：分位点函数（CDF的逆）
isf：逆残存函数（sf的逆）
fit：对一组随机取样进行拟合，最大似然估计方法找出最适合取样数据的概率密度函数系数。
*离散分布的简单方法大多数与连续分布很类似，但是pdf被更换为密度函数pmf。
'''
st.norm.cdf(0)  # 标准正态分布在 0 处的累计分布概率值

st.norm.cdf([-1, 0, 1])  # 标准正态分布分别在 -1， 0， 1 处的累计分布概率值

st.norm.pdf(0)  # 标准正态分布在 0 处的概率密度值

st.norm.ppf(0.975)  # 标准正态分布在 0.975 处的逆函数值

st.norm.lsf(0.975)  # 标准正态分布在 0.025 处的生存函数的逆函数值

st.norm.cdf(0, loc=2, scale=1)  # 均值为 2，标准差为 1 的正态分布在 0 处的累计分布概率值

st.binom.pmf(4, n=100, p=0.05)  # 参数值 n=100, p=0.05 的二项分布在 4 处的概率密度值

st.geom.pmf(4, p=0.05)  # 参数值 p=0.05 的几何分布在 4 处的概率密度值

st.poisson.pmf(2, mu=3)  # 参数值 mu=3 的泊松分布在 2 处的概率密度值


#创建随机变量（rv：random variable）
# 泊松分布为离散型概率分布
F_true = 1000
N = 50
F = st.poisson(F_true).rvs(N)

# 随机变量X：投100次硬币正面出现的个数
# 用二项分布表示
n, p = 100, .5
X = st.binom(n, p)

mu, sigma = 1, 1
xs = np.linspace(-5, 5, 1000)
plt.plot(xs, st.norm.pdf(xs, loc=mu, scale=sigma))


#用rv_discrete 类自定义离散概率分布
x = range(1,7)
p = (0.4, 0.2, 0.1, 0.1, 0.1, 0.1)
dice = st.rv_discrete(values=(x,p))
dice.rvs(size=20)
# 返回1,1出现的范围为0-0.4
dice.ppf(0.33)
#返回6
dice.ppf(1)

#返回0.6
dice.pmf(4)

#返回0.2
dice.pmf(1)
