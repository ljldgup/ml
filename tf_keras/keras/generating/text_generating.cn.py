import random

import numpy as np
import tensorflow as tf
# 重置权重，使得权重分布差距不会太大，增强文本的随机性
from tensorflow.python.keras.layers import CuDNNLSTM, Dense, Embedding
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer

from tf_keras.tf.test.transformer_test import Transformer


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def reweight_distribution(original_distribution, temperature=0.5):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)


def data_generator(index_array):
    # 非精确batch_size
    round = 1000
    size = len(index_array) // round

    # 训练一小部分就可以
    for r in range(size // 50):
        t_array = index_array[r * size:r * size + size]
        sentences = []
        next_chars = []
        # i 到 i + maxlen - 1个字符, 对应训练目标 i + maxlen 处字符
        for i in range(0, len(t_array) - maxlen, 1):
            sentences.append(t_array[i: i + maxlen])
            next_chars.append(t_array[i + maxlen])
        yield np.array(sentences), np.array(next_chars)


def lstm_model():
    model = tf.keras.models.Sequential()
    model.add(Embedding(word_len, 64, input_length=maxlen))
    model.add(CuDNNLSTM(512, return_sequences=True))
    model.add(CuDNNLSTM(512))
    model.add(Dense(word_len, activation='softmax'))
    optimizer = tf.keras.optimizers.RMSprop(lr=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
    return model


def text_generate(model):
    for temperature in [0.01, 0.1, 1]:

        start_index = random.randint(0, len(text) - maxlen - 1)
        generated_index = index_array[start_index: start_index + maxlen].tolist()
        print('\n------ temperature:', temperature)
        for i in range(40):
            preds = model.predict(
                (np.array(generated_index[-maxlen:-1])[np.newaxis, :], np.array(generated_index[-1:])[np.newaxis, :]))[
                0]
            next_index = sample(preds.flatten(), temperature)
            generated_index.append(next_index)
        # print(le.inverse_transform("".join(le.inverse_transform(generated_index).tolist())))
        print(tokenizer_cn.sequences_to_texts([generated_index])[0].replace(' ', ''))


def transformer_model():
    # 最后输出的形状时 None,target,seq,target_vocab_size
    sample_transformer = Transformer(
        num_layers=4, d_model=64, num_heads=4, dff=128,
        input_vocab_size=word_len, target_vocab_size=word_len,
        pe_input=30, pe_target=1)
    input = tf.keras.layers.Input((30,))
    # target 决定了输出的序列长度，所以必须要加
    target = tf.keras.layers.Input((1,))

    out, _ = sample_transformer(input, target, training=True,
                                enc_padding_mask=None,
                                look_ahead_mask=None,
                                dec_padding_mask=None)
    final_out = tf.keras.layers.Softmax()(out)
    # 注意单个 输出，输出的时候不要用list
    model = tf.keras.Model([input, target], final_out)
    model.compile(optimizer=Adam(lr=1e-2),
                  loss='sparse_categorical_crossentropy')
    return model


if __name__ == '__main__':
    maxlen = 10
    path = r'127822.txt'
    text = open(path, encoding='utf-8').read().lower()
    text = text.replace('\n', '').replace('\u3000', '').replace(' ', '')
    text = text[20000:120000]
    # 使用Tokenizer代替label encoder
    tokenizer_cn = Tokenizer(char_level=True)
    tokenizer_cn.fit_on_texts([text])
    index_array = np.array(tokenizer_cn.texts_to_sequences([text])).flatten()
    word_len = len(tokenizer_cn.word_index) + 1
    '''
    # label encoder 编码
    text_array = np.array(list(text))
    le = LabelEncoder()
    index_array = le.fit_transform(np.array(text_array))
    '''

    # 使用dataset代替 自己的生成器
    dataset = tf.data.Dataset.from_tensor_slices(index_array)
    dataset = dataset.window(maxlen + 1, shift=1, drop_remainder=True)
    # 返回的是dateset，使用flat_map压平成为张量，31 batch_size的大小，正好每个窗口成为一组数据
    dataset = dataset.flat_map(lambda w: w.batch(maxlen + 1))

    # transformer 输入要两个懒得改了
    # dataset = dataset.map(lambda x: (x[:-1], x[-1]))
    dataset = dataset.map(lambda x: ((x[:-2], x[-2]), x[-1]))
    dataset = dataset.batch(128)
    '''
    x,y = next(iter(dataset))
    tokenizer_cn.sequences_to_texts(x.numpy())
    tokenizer_cn.sequences_to_texts(y.numpy()[None,:])
    for x, y in dataset:
        print(x.shape, y.shape)
    '''
    # transformer收敛效果很差，原因不明,可能是输出太小的缘故，导致decoder的通道太小，
    # 将decoder layer减小的到一层，损失明显下降。。单输出效果仍然一般，比之前好一点
    model = transformer_model()
    # model = lstm_model()
    history = model.fit(dataset, epochs=20, verbose=1)
    # 生成效果都很差
    text_generate(model)
