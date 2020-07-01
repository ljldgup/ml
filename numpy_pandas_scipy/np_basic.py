import numpy as np

# 矩阵操作

# 0-11, 1-3-...-11
arr = np.arange(12)
arr = np.arange(1, 12, 2)

# 使用like直接创建形状相等的矩阵
np.ones(shape=(1, 2))
np.ones_like(arr)
np.zeros_like(arr)

# 使用函数生成矩阵，参数是索引
arr = np.fromfunction(lambda x, y: 10 * x + y, (4, 3), dtype=int)

# size元素个数，ndim维度，shape维度的具体大小
arr.size
arr.ndim
arr.shape

# 创建相同形状举证
# 创建相同形状举证
np.zeros(arr.shape)
np.ones(arr.shape)

# 重塑
arr.reshape(3, 4)

# 增加一维
arr.reshape(-1, 1)
# 索引
arr[2, 1]
# 切片
arr[0:2, :]
arr[:, 1]

arr[:, [1, 2]]
arr[range(3), :]

# 这里的2是切片间隔
arr[1::2, :]
arr[1:3:2, :]

# 转型
arr.astype(np.float64)

# 一块内存均被赋值为12
arr[2] = 12
arr[arr > 20] = 20

# 普通乘法
arr * arr

# 矩阵乘法， 转置arr.T
np.dot(arr, arr.T)

# 指数，三角函数（弧度制），开方
np.exp(arr)
np.sin(arr)
np.sqrt(arr)
# 按行累计总和
arr.cumsum(1)

# 纵向，横向分割,2,3代表分成几块
np.vsplit(arr, 2)
np.hsplit(arr, 3)

# ------------------------------------------
# 统计
nums = np.random.randint(1, 6, (15, 5))

# 最大值，最大值位置索引，排序索引，排序（会改变本身的值）
nums.max()
nums.argmax()
nums.argsort()
nums.sort()

# 参数维度相同，同一位置保留最大
np.maximum(nums, np.random.randint(2, 5, (15, 5)))

# 根据条件生成
np.where(nums % 2, 1, 0)

# 均值
# 矩阵只有两维，这两个等价
np.mean(nums)
np.mean(nums, axis=(0, 1))

# 每列均值
np.mean(nums, axis=1)

# 每行均值
np.mean(nums, axis=0)

# 中位数
np.median(nums)

# 均方差
np.var(nums)

# 标准差
np.std(nums)

# 协方差
a = [[1, 2], [4, 7]]
b = [[7, 16], [17, 8]]
c = np.cov(a, b)

# 条件判断
a = np.array([[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 10]])

print(a > 6)
'''
[[False False False False False],
[False  True  True  True  True]]
'''

# 一次性赋予 多个值
a[range(2), [1, 3]] = [100, 100]

'''
array([[  1, 100,   3,   4,   5],
       [  6,   7,   8, 100,  10]])
'''
b = a[a > 6]
# [ 7  8  9 10]

a[a > 6] = 0
'''
[[1 2 3 4 5],
 [6 0 0 0 0]]
'''

# 保存读取
np.save('path.npy', arr)
np.savetxt('path.txt', arr, delimiter=' ')
np.load('path.npy')
np.loadtxt("path.txt", delimiter=' ')

z = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
# None代表新增加一个维度，它有一个别称叫newaxis
z.shape
# numpy会把没写的维度自动补成:
# 等价于z[np.newaxis],z[np.newaxis, :, :],在开始追加一个维度
z[None].shape, z[None, ...].shape
# 等价于z[:, np.newaxis],z[:, np.newaxis, :]在第二个维度位置追加一个维度
z[:, None].shape
# 等价于z[:, :, np.newaxis]，在最后追加一个维度
z[:, :, None].shape, z[..., None].shape
# reshape 中-1值长度可调整一个维度
print(z.reshape(-1))  # 16
print(z.reshape(2, -1))  # 2,8
print(z.reshape(-1, 4))  # 4,4, 第一个4是自动调整来的

# 形状是 （a,1) * (1,b) -> (a,b), TensorFlow中有很多此类操作
# 与矩阵相乘dot的效果一样但实际是通过广播实现的
x = np.arange(5)[:, np.newaxis]
y = np.arange(9)[np.newaxis, :]
z = x * y
w = x.dot(y)
print(x.shape, y.shape, z.shape, w.shape)
print(w == z)

# 这样直接乘法会出错
np.arange(5) * np.arange(9)

t = np.array([[1, 2], [1, 2]])
print(t * t == t.dot(t))

'''
广播的原则
    数组维度不同，后缘维度的轴长相符
    数组维度相同，其中有个轴为1
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

# 重复元素
np.tile([1, 2], 14)
# 只扩充到最后一个维度
np.tile([[1, 2], [3, 4]], 14)
np.tile([1, 2], 14).reshape(2, 14)
