import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.python.keras.utils import to_categorical

# pycharm不会将当前文件目录自动加入自己的sourse_path。右键make_directory as-->Sources Root将当前工作的文件夹加入source_path就可以了。
from tf_keras.keras import tools

(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.reuters.load_data(
    num_words=10000)


def decode_newswire(data):
    word_index = tf.keras.datasets.reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in
                                 train_data[0]])
    return decoded_newswire


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    # 这里应该可改成results[range(len(sequences)),sequences]=1
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


def create_model():
    # 对于这个例子，最好的损失函数是categorical_crossentropy（分类交叉熵）。它用于衡量两个概率分布之间的距离
    # 这里分类较多，所以中间的网络也较多,网络太小会导致信息被压缩
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    # one_hot_train_labels = to_one_hot(train_labels)
    # one_hot_test_labels = to_one_hot(test_labels)
    # to_categorical one hot code内置工具
    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)
    x_val, y_val, partial_x_train, partial_y_train = tools.create_data(x_train, one_hot_train_labels, 1000)
    model = create_model()
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=9,
                        batch_size=512,
                        validation_data=(x_val, y_val))

    # tools.plot_loss(history.history)
    tools.plot_accuracy(history.history)

    results = model.evaluate(x_test, one_hot_test_labels)
    predictions = model.predict(x_test)
    np.sum(predictions[0])
    np.argmax(predictions[0])
