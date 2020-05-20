import random
import sys

import numpy as np
import tensorflow as tf
# 重置权重，使得权重分布差距不会太大，增强文本的随机性
from tensorflow.python.keras.layers import CuDNNLSTM, Dense


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


def vectorize(text):
    step = 3
    sentences = []
    next_chars = []
    round = len(chars)//10
    text = text.replace('\n', '').replace('\u3000', '')
    for r in range(len(text) // round):
        tmp_text = text[r * round:r * round + round]
        sentences = []
        next_chars = []
        # i 到 i + maxlen - 1个字符, 对应训练目标 i + maxlen 处字符
        for i in range(0, len(tmp_text) - maxlen, step):
            sentences.append(tmp_text[i: i + maxlen])
            next_chars.append(tmp_text[i + maxlen])

        # 使用中文这里直接会报内存错误，可以考虑用生成器
        # 但是实际上中文用one-hot编码效果估计不好，回头研究一下
        x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1

        yield x, y


def build_model():
    model = tf.keras.models.Sequential()
    model.add(CuDNNLSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars), activation='softmax'))
    optimizer = tf.keras.optimizers.RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model


def text_generate(model):
    for temperature in [0.2, 0.5, 1.2]:

        start_index = random.randint(0, len(text) - maxlen - 1)
        generated_text = text[start_index: start_index + maxlen]
        print('\n------ temperature:', temperature)
        sys.stdout.write(generated_text)
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]
            generated_text += next_char
            generated_text = generated_text[1:]
            sys.stdout.write(next_char)
        sys.stdout.flush()


if __name__ == '__main__':
    maxlen = 30
    path = r'1.txt'
    text = open(path, encoding='utf-8').read().lower()
    chars = sorted(list(set(text)))
    char_indices = dict((char, chars.index(char)) for char in chars)

    print('Corpus length:', len(text))
    model = build_model()
    for x, y in vectorize(text):

        model.train_on_batch(x, y)

    text_generate(model)
