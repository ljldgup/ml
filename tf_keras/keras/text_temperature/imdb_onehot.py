import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers

# pycharm不会将当前文件目录自动加入自己的sourse_path。右键make_directory as-->Sources Root将当前工作的文件夹加入source_path就可以了。
from tf_keras.keras import tools

(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(
    num_words=10000)


# 单词索引都不会超过10 000
# max([max(sequence) for sequence in train_data])

# 解码
def decode_review(data):
    word_index = tf.keras.datasets.imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    # 注意，索引减去了3，因为0、1、2 是为“padding”（ 填充）、“start of sequence”（序列开始）、“unknown”（未知词）分别保留的索引
    decoded_review = ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in data])
    return decoded_review


# 填充列表，使其具有相同的长度
# 对列表进行 one-hot 编码，将其转换为 0 和 1 组成的向量。
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def create_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    # 二分类问题，网络输出是一个概率值，那么最好使用binary_crossentropy（二元交叉熵）损失
    # 这并不是唯一可行的选择，比如你还可以使用mean_squared_error（均方误差）。但对于输出概率值的模型，交叉熵（crossentropy）往往是最好的选择。
    # 交叉熵是来自于信息论领域的概念，用于衡量概率分布之间的距离
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    # print(decode_review(train_data[0]))

    # 向量化
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    x_val, y_val, partial_x_train, partial_y_train = tools.create_data(x_train, y_train)

    model = create_model()
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))

    # 绘制训练损失和验证损失
    history_dict = history.history
    tools.plot_loss(history_dict)
    # tools.plot_accuracy(history_dict)

    # 预测
    model.predict(x_test)
