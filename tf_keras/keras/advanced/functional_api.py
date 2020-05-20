from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import layers
from tensorflow.python.keras import Input
from tensorflow.python.keras.utils import to_categorical

import numpy as np


def simple_seq_model():
    '''
    #老用法
    seq_model = Sequential()
    seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
    seq_model.add(layers.Dense(32, activation='relu'))
    seq_model.add(layers.Dense(10, activation='softmax'))
    return seq_model
    '''

    # 函数式api
    input_tensor = Input(shape=(64,))
    x = layers.Dense(32, activation='relu')(input_tensor)
    x = layers.Dense(32, activation='relu')(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)
    # input_tensor 到output_tensor需要相连
    model = Model(input_tensor, output_tensor)
    return model


# 多输入模型
def multi_input_model():
    text_vocabulary_size = 10000
    question_vocabulary_size = 10000
    answer_vocabulary_size = 500
    text_input = Input(shape=(None,), dtype='int32', name='text')
    embedded_text = layers.Embedding(
        text_vocabulary_size, 64)(text_input)
    encoded_text = layers.CuDNNLSTM(32)(embedded_text)
    question_input = Input(shape=(None,),
                           dtype='int32',
                           name='question')
    embedded_question = layers.Embedding(
        question_vocabulary_size, 32)(question_input)
    encoded_question = layers.CuDNNLSTM(16)(embedded_question)

    # 注意这里将两个输入合并
    # 设axis=i，则沿着第i个下标变化的方向进行操作，-1代表倒数第一个加起来
    concatenated = layers.concatenate([encoded_text, encoded_question],
                                      axis=-1)
    answer = layers.Dense(answer_vocabulary_size,
                          activation='softmax')(concatenated)
    model = Model([text_input, question_input], answer)
    '''
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    num_samples = 1000
    max_length = 100
    text = np.random.randint(1, text_vocabulary_size,
                             size=(num_samples, max_length))
    question = np.random.randint(1, question_vocabulary_size,
                                 size=(num_samples, max_length))
    answers = np.random.randint(answer_vocabulary_size, size=(num_samples))
    answers = to_categorical(answers, answer_vocabulary_size)
    model.fit([text, question], answers, epochs=10, batch_size=128)
    model.fit({'text': text, 'question': question}, answers,
              epochs=10, batch_size=128)
    '''
    return model


def multi_output_model():
    vocabulary_size = 50000
    num_income_groups = 10
    posts_input = Input(shape=(None,), dtype='int32', name='posts')
    embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
    x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)

    # 两个输出
    age_prediction = layers.Dense(1, name='age')(x)
    income_prediction = layers.Dense(num_income_groups,
                                     activation='softmax',
                                     name='income')(x)
    gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)
    model = Model(posts_input,
                  [age_prediction, income_prediction, gender_prediction])

    '''
    #编译时选择多个损失标准
    model.compile(optimizer='rmsprop',
        loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])
    model.compile(optimizer='rmsprop',
        loss={'age': 'mse',
        'income': 'categorical_crossentropy',
        'gender': 'binary_crossentropy'})
    model.compile(optimizer='rmsprop',
        loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
        loss_weights=[0.25, 1., 10.])
    model.compile(optimizer='rmsprop',
        loss={'age': 'mse',
        'income': 'categorical_crossentropy',
        'gender': 'binary_crossentropy'},
        loss_weights={'age': 0.25,
        'income': 1.,
        'gender': 10.})
    model.fit(posts, [age_targets, income_targets, gender_targets],
        epochs=10, batch_size=64)   
    model.fit(posts, {'age': age_targets,
        'income': income_targets,
        'gender': gender_targets},
        epochs=10, batch_size=64)
    '''

    return model


# 残差连接
def residual_connection_model():
    x = Input(shape=(12, 4, 256))

    # padding 填充和之前size一样
    # 128 是通道数量，3是卷积核尺寸
    # 进行卷积后 通道数128将成为第三个维度，之前两个维度由于补全不变
    y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)

    # 使用1x1卷积核使得尺寸可以相加
    # strides 是卷积步长
    residual = layers.Conv2D(128, 1, strides=1, padding='same')(x)
    y = layers.add([y, residual])
    prediction = layers.Dense(1)(y)
    model = Model(x, prediction)
    return model


def shared_weight_model():
    lstm = layers.CuDNNLSTM(32)
    left_input = Input(shape=(None, 128))
    left_output = lstm(left_input)
    right_input = Input(shape=(None, 128))
    right_output = lstm(right_input)

    # 虽然也是用concatenate，但是是公用的一个lstm层，之前多输入是两个lstm
    merged = layers.concatenate([left_output, right_output], axis=-1)
    predictions = layers.Dense(1, activation='sigmoid')(merged)
    model = Model([left_input, right_input], predictions)
    # model.fit([left_data, right_data], targets)
    return model


if __name__ == '__main__':
    model1 = residual_connection_model()
    model1.summary()
    model2 = multi_input_model()
    model2.summary()
