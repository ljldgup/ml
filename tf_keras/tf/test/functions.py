import tensorflow as tf
import numpy as np

a = tf.constant(np.arange(8).reshape(1, 2, 2, 2))
b = tf.constant(np.arange(12).reshape(1, 2, 2, 3))
# b不参与运算，沿着bc拆开，得到每个i*j之间相乘，
tf.linalg.einsum('bijc,bijd->bcd', a, b)
