import tensorflow as tf
import numpy as np

a = tf.constant(np.arange(8).reshape(1, 2, 2, 2))
b = tf.constant(np.arange(12).reshape(1, 2, 2, 3))
# b不参与运算，沿着bc拆开，得到每个i*j之间相乘，
tf.linalg.einsum('bijc,bijd->bcd', a, b)

#
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b1 = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
b2 = tf.constant([7, 8, 9, 10, 11, 12], shape=[2, 3])
# 逐个相乘
tf.multiply(a, b2)
# 矩阵相乘
tf.matmul(a, b1)
