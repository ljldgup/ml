from keras_preprocessing import sequence
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.layers import Embedding, SimpleRNN, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, CuDNNLSTM
from tensorflow.python.keras.optimizers import RMSprop

from tf_keras.keras import tools


def build_simple_rnn_model(max_features=10000):
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model


def build_LSTM_model(max_features=10000):
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(CuDNNLSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model


def build_cnn_1d_model(maxlen=500):
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(Conv1D(32, 7, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(32, 7, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer=RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model


if __name__ == '__main__':
    max_features = 10000
    maxlen = 500
    batch_size = 32
    print('Loading data...')
    (input_train, y_train), (input_test, y_test) = imdb.load_data(
        num_words=max_features)
    print(len(input_train), 'train sequences')
    print(len(input_test), 'test sequences')
    print('Pad sequences (samples x time)')
    input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
    input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
    print('input_train shape:', input_train.shape)
    print('input_test shape:', input_test.shape)
    # rnn训练速度极慢
    # model = build_simple_rnn_model()

    #CuDNNLSTM, 才能使用gpu提高速度，普通LSTM很慢
    model = build_LSTM_model()

    # 训练速度较快
    # model = build_cnn_1d_model()


    history = model.fit(input_train, y_train,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)
    tools.plot_loss(history.history)
    # tools.plot_accuracy(history.history)
