import tensorflow as tf
from tensorflow_core import int32
import numpy as np

# 张量是不可变的
print(tf.add(1, 2))
print(tf.add([3, 8], [2, 5]))
print(tf.square(6))
print(tf.reduce_sum([7, 8, 9]))
print(tf.square(3) + tf.square(4))

# 形状和类型
x = tf.matmul([[3], [6]], [[2]])
print(x)
print(x.shape)
print(x.dtype)

# 测试GPU是否
x = tf.random.uniform([3, 3])
print('Is GPU available:')
print(tf.test.is_gpu_available())
print('Is the Tensor on gpu #0:')
print(x.device.endswith('GPU:0'))
