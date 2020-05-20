import numpy as np

# np.random
# 生成3x2矩阵,0-1
np.random.rand(3, 2)

# 正太分布 N(3, 2.5^2)
2.5 * np.random.randn(2, 4) + 3

# 随机整数, 开区间[0,3)
np.random.randint(3, size=(10))
np.random.randint(3, size=(2, 4))
# 随机整数, 闭区间[0,3]
np.random.random_integers(2, size=(2, 4))

# 随机整数, 闭区间[1,6],
np.random.random_integers(1, 6, 100)

# 2X3 随机浮点数[0,1)
np.random_sample((2, 3))

# Generate a non-uniform random sample from np.arange(5) of size 3:
# np.random.choice(5, 3)
np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])

# shuffle打乱顺序
arr = np.arange(10)
np.random.shuffle(arr)

# permutation返回一个随机排列,可以用于打乱数据
np.random.permutation(10)
np.random.permutation([1, 4, 9, 12, 15])

arr = np.arange(9).reshape((3, 3))
np.random.permutation(arr)

# 二项分布
# binomial(n,p,size=None)
# n表示n次的试验，p表示的试验成功的概率
np.random.binomial(9, 0.1, 20000)

# 泊松分布
np.random.poisson(251 / 115, size=10000)

# 正太分布
np.random.normal(0, 0.2, 1000)
