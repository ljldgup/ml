from sklearn.datasets import load_sample_image
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


def generate_time_series(batch_size, n_steps):
    # numpy数组也能通过这种方式解包
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # wave1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # + wave2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)  # +noise
    return series[..., np.newaxis].astype(np.float32)


n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

y_pred = X_valid[:, -1]
np.mean(keras.losses.mean_squared_error(y_valid, y_pred))

model = keras.models.Sequential([
    keras.layers.SimpleRNN(1, input_shape=[None, 1])
])

# deep rnn 结构，每层都return_sequences
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.SimpleRNN(1)
])

# TimeDistributed(keras.layers.Dense(10))
# 其实就是一个各个time step输入共享权重的dense layer,这里只有20*10 + 10=20个参数量

# 这里的rnn没有规定输入长度 ，20*20+20 + 20 =440 貌似 没有输入-20的权重w参数量，
# 这里的理解有误，rnn所有时间步之间的传播都是用同一组变换矩阵，所以可以是任意步数，参数不会变
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])

model = keras.models.Sequential([
    keras.layers.Conv1D(filters=20, kernel_size=4, strides=2,
                        padding="valid",
                        input_shape=[None, 1]),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])


# layerNorm在通道方向上归一化，主要对RNN作用明显；
class LNSimpleRNNCell(keras.layers.Layer):
    def __init__(self, units, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.state_size = units
        self.output_size = units
        self.simple_rnn_cell = keras.layers.SimpleRNNCell(units, activation=None)
        self.layer_norm = keras.layers.LayerNormalization()
        self.activation = keras.activations.get(activation)

    def call(self, inputs, states):
        outputs, new_states = self.simple_rnn_cell(inputs, states)
        norm_outputs = self.activation(self.layer_norm(outputs))
        return norm_outputs, [norm_outputs]
