import numpy as np

# any,all也可以按轴计算，没有轴全局，和max,min一样
arr = np.random.randint(1, 80, (8, 4))
(arr > 50).any(0)
(arr > 50).any(1)
(arr > 30).all(0)
(arr > 30).all(1)
'''
广播的原则
    数组维数不同，后缘维度的轴长相符
    数组维数相同，其中有个轴为1
'''

# 后缘维度的轴长相符
np.arange(6).reshape(3, 2) + np.arange(2).reshape(2)
np.arange(18).reshape(3, 2, 3) + np.arange(6).reshape(2, 3)

# 数组维度相同，其中有个轴长度为1，操作后改轴长度将会被另一个矩阵对应轴长度取代
np.arange(6).reshape(3, 1, 2) + np.arange(24).reshape(3, 4, 2)

# 注意这里连个矩阵分别在1,0维度上长度为1，所以可以相加 (6, 1)+(1, 5)->(6,5)
np.arange(6).reshape(6, 1) + np.arange(5).reshape(1, 5)

# 同上但中间维度2 不变  (3, 2, 1)+(1, 2, 3) -> (3, 2, 3)
np.arange(6).reshape(3, 2, 1) + np.arange(6).reshape(1, 2, 3)

# 计算元素累积乘积
np.prod(np.arange(1, 7))
np.prod(np.arange(1, 7).reshape(3, 2))
np.prod(np.arange(1, 7).reshape(3, 2), axis=0)
np.prod(np.arange(1, 7).reshape(3, 2), axis=1)

# 沿不同的方向累加，累积
np.cumsum(np.arange(12).reshape(3, 4), axis=0)
np.cumsum(np.arange(12).reshape(3, 4), axis=1)
np.cumprod(np.arange(12).reshape(3, 4), axis=0)
np.cumprod(np.arange(12).reshape(3, 4), axis=1)

# 去重并排序
np.unique(np.arange(12).reshape(3, 4))

# 交集
np.intersect1d(np.arange(12).reshape(3, 4), np.arange(8, 16).reshape(2, 4))

# 并集
np.union1d(np.arange(12).reshape(3, 4), np.arange(8, 16).reshape(2, 4))

# x是否存在于y
np.in1d(np.arange(12).reshape(3, 4), np.arange(8, 16).reshape(2, 4))

# 重复元素
np.tile([1, 2], 14)
# 只扩充到最后一个维度
np.tile([[1, 2], [3, 4]], 14)
np.tile([1, 2], 14).reshape(2, 14)

# condition为真取x， 假取y
x = np.arange(0, 20, 3)
y = np.arange(0, 200, 30)
condition = x % 2
np.where(condition, x, y)
np.where(condition, 1, 0)

# 数组拼接
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])

np.concatenate([arr1, arr2], axis=0)
np.concatenate([arr1, arr2], axis=1)
np.vstack((arr1, arr2))
np.hstack((arr1, arr2))

np.r_[arr1, arr2]
np.c_[arr1, arr2]
# 可以用在切片上
np.c_[1:6, -10:-5]

# 不reshape的方法
t = np.random.rand(2, 3, 4)
t[:, np.newaxis, :]
t[np.newaxis, :]

arr = np.random.rand(2, 3, 4)
# 纵向，横向分割,2,3代表分成几块
np.vsplit(arr, 2)
np.hsplit(arr, 3)
