from tensorflow import keras
import tensorflow as tf

model1 = keras.models.Sequential([
    keras.layers.Embedding(100, 5, input_shape=[4, ]),
    keras.layers.Dense(10, activation="relu"),
])

model1.summary()

model2 = keras.models.Sequential([
    keras.layers.Embedding(100, 5, input_shape=[4, ]),
    keras.layers.TimeDistributed(keras.layers.Dense(10, activation="relu")),
])

model2.summary()

# 餐数量使用TimeDistributed(Dense)和直接使用dense是完全相同的， 使用相同的参数输出也一样
# 从矩阵相乘的角度，去掉第一位batch_size，无论时间步拆不拆，dense层都是直接和后面几位相乘，应该是一样的
# Timedistribute 应该比较适合类似卷积涉及多个维度的
model2.set_weights(model1.get_weights())
t = tf.constant([[2, 3, 4, 5]])
model1.predict(t)
model2.predict(t)
model2.predict(t) == model1.predict(t)
