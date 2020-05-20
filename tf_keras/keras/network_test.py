from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding, CuDNNGRU
import numpy as np


def embedding_test():
    '''
        模型将输入一个大小为 (batch, input_length) 的整数矩阵。
        输入中最大的整数（即词索引）不应该大于 999 （词汇表大小）
        现在 model.output_shape == (None, 10, 64)，其中 None 是 batch 的维度。
    '''
    model = Sequential()
    # model.add(Embedding(1000, 64, input_length=10))
    model.add(Embedding(160, 4))
    # input_array = np.random.randint(1000, size=(32, 10))
    input_array = np.random.randint(160, size=(5,4, 4))
    model.compile('rmsprop', 'mse')
    output_array = model.predict(input_array)
    print(output_array.shape)
    return model


def gru_test():
    '''
    使用return_sequences 返回所有time steps的输出
    不适用的时候只返回最后一次
    '''
    model = Sequential()
    model.add(CuDNNGRU(128))
    # model.add(CuDNNGRU(128, return_sequences=True))
    model.compile('rmsprop', 'mse')
    input_array = np.random.normal(size=(32, 10, 1))
    output_array = model.predict(input_array)
    print(output_array.shape)
    return model


if __name__ == '__main__':
    model = embedding_test()
