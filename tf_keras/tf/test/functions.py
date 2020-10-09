import tensorflow as tf
import numpy as np

'''
tensorflow与numpy的很大区别是很多操作都是从第二维度开始操作，第一维度认为是batch_size
'''

a = tf.constant(np.arange(8).reshape(1, 2, 2, 2))
b = tf.constant(np.arange(12).reshape(1, 2, 2, 3))
# b不参与运算，沿着bc拆开，得到每个i*j之间相乘，
tf.linalg.einsum('bijc,bijd->bcd', a, b)

#
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b1 = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
b2 = tf.constant([7, 8, 9, 10, 11, 12], shape=[2, 3])
# 逐个相乘 element wise
a * b2
tf.multiply(a, b2)
# 矩阵相乘
tf.matmul(a, b1)

# 删除1的维度
tf.squeeze(b, axis=0)

# 增加一个维度
tf.expand_dims(a, axis=1)

# 剪裁
tf.clip_by_value(a, clip_value_min=0, clip_value_max=1)

# 整数无法计算
tf.nn.tanh([1., 2., 3., 4., 5.])
tf.nn.softmax([[1., 2.], [2., 3.]])

tf.reduce_sum([[1., 2.], [2., 3.]])
tf.reduce_sum([[1., 2.], [2., 3.]], axis=1)

# logical_not 取反
tf.math.equal([0, 2, 0, 4, 0, 1], 0)
tf.math.logical_not(tf.math.equal([0, 2, 0, 4, 0, 1], 0))

# 多个序列相加
tf.add_n([[2, 3, 4], [4, 5, 6], [3, 4, 5]])

# 矩阵乘积 matmul transpose_b 后会将前面一样的维度合并，适用于batch_size等
tf.constant([[[[1, 2, 3]]]]).shape
tf.matmul([[[[1, 2, 3]]]], [[[[1, 2, 3]]]], transpose_b=True).shape
tf.matmul(tf.random.uniform((2, 3, 4, 5)), tf.random.uniform((2, 3, 4, 5)), transpose_b=True).shape
