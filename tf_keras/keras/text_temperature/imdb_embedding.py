from tensorflow.keras.datasets import imdb
from tensorflow.python.keras import preprocessing, Sequential
from tensorflow.python.keras.layers import Embedding, Flatten, Dense

from tf_keras.keras import tools


def build_model(maxlen=20):
    model = Sequential()
    model.add(Embedding(10000, 8, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model


if __name__ == '__main__':
    max_features = 10000
    maxlen = 20
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=max_features)
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    model = build_model()
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_split=0.2)
    tools.plot_loss(history.history)
    # tools.plot_accuracy(history.history)
