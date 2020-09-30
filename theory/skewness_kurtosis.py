import numpy as np
import pandas as pd

'''
峰度太高，数据集中在小范围
偏度太高，数据偏向一个范围
这两者容易造成学习时的偏差
对于右偏（波峰在左）可以考虑对数log
对于左偏（波峰在右）可以考虑指数exp，或者幂指数
'''

# 普通正太分布
df = pd.DataFrame(np.random.randn(1000) + 10)
ax = df.hist(bins=20)
print("Skewness: %f" % df.skew())
print("Kurtosis: %f" % df.kurt())

# 指数化之后波峰完全偏左，成为右偏分布（也叫正偏分布，其偏度>0），这种情况取对数可以使其接近正态分布
df = pd.DataFrame(np.exp(np.random.randn(1000) + 10))
ax = df.hist(bins=20)
print("Skewness: %f" % df.skew())
print("Kurtosis: %f" % df.kurt())

# 对正太分布取对数，正太分布没有特别大的变化,稍微左偏分布（也叫负偏分布，其偏度<0），峰度降低
df = pd.DataFrame(np.log(np.random.randn(1000) + 10))
ax = df.hist(bins=20)
print("Skewness: %f" % df.skew())
print("Kurtosis: %f" % df.kurt())

# 稍微左偏分布，峰度降低
df = pd.DataFrame(np.sqrt(np.random.randn(1000) + 10))
ax = df.hist(bins=20)
print("Skewness: %f" % df.skew())
print("Kurtosis: %f" % df.kurt())

# 对均布分布取log可以得到明显的左偏分布，波峰在右边
df = pd.DataFrame(np.log(np.random.uniform(1, 1000, 1000)))
ax = df.hist(bins=20)
print("Skewness: %f" % df.skew())
print("Kurtosis: %f" % df.kurt())

# 对左偏取6次方，呈均布效果
df = pd.DataFrame(np.power(np.log(np.random.uniform(1, 1000, 1000)), 6))
ax = df.hist(bins=20)
print("Skewness: %f" % df.skew())
print("Kurtosis: %f" % df.kurt())

# 对均布分布取指数exp可以得到明显的右偏分布，但没有正太分布偏的严重
df = pd.DataFrame(np.exp(np.random.uniform(1, 10, 1000)))
ax = df.hist(bins=20)
print("Skewness: %f" % df.skew())
print("Kurtosis: %f" % df.kurt())

# 对均布分布开方后从左往右成线性增加
# 均布时F(x) = x^2/100，F(x) = x^2/100, f(x) = x/50
df = pd.DataFrame(np.sqrt(np.random.uniform(1, 100, 10000)))
ax = df.hist(bins=20)
print("Skewness: %f" % df.skew())
print("Kurtosis: %f" % df.kurt())
