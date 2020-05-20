import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import GRU, Dense, Flatten, CuDNNGRU
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.layers import layers

from tf_keras.keras import tools


def get_data(fname):
    f = open(fname)
    data = f.read()
    f.close()
    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]
    print(header)
    print(len(lines))
    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values
    # 温度（单位：摄氏度
    # temp = float_data[:, 1]  ）
    # plt.plot(range(len(temp)), temp)

    return float_data


def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


def build_simple_model():
    model = Sequential()
    model.add(Flatten(input_shape=(lookback // step, float_data.shape[-1])))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    return model


def build_gru_model():
    model = Sequential()
    #CuDNNGRU 代替GRU, 提高速度
    model.add(CuDNNGRU(32, input_shape=(None, float_data.shape[-1])))
    model.add(Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    return model


def build_gru_model2():
    # 带有dropout，循环层堆叠（增加循环层层数）
    model = Sequential()
    model.add(layers.GRU(32,
                         dropout=0.1,
                         recurrent_dropout=0.5,
                         return_sequences=True,
                         input_shape=(None, float_data.shape[-1])))
    model.add(layers.GRU(64, activation='relu',
                         dropout=0.1,
                         recurrent_dropout=0.5))
    model.add(layers.Dense(1))


def evaluate_naive_method(value_steps, value_generator):
    batch_maes = []
    for step in range(value_steps):
        samples, targets = next(value_generator)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))


if __name__ == '__main__':
    lookback = 1440
    step = 6
    delay = 144
    batch_size = 128
    float_data = get_data('jena_climate_2009_2016.csv')

    train_gen = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=200000,
                          shuffle=True,
                          step=step,
                          batch_size=batch_size)
    val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=200001,
                        max_index=300000,
                        step=step,
                        batch_size=batch_size)
    test_gen = generator(float_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=300001,
                         max_index=None,
                         step=step,
                         batch_size=batch_size)
    val_steps = (300000 - 200001 - lookback) // batch_size
    test_steps = (len(float_data) - 300001 - lookback) // batch_size

    # model = build_simple_model()
    model = build_gru_model()
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=5,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)
    # 必须没训练之前调用
    # evaluate_naive_method(val_steps, val_gen)

    # 注意回归只有loss
    tools.plot_loss(history.history)
