import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
'''
全连接权重本身就可以看成注意力，这没什么意义
cnn是局部过滤器，也一样可以看成一种类似注意力的机制。
'''
if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    input = tf.keras.layers.Input(shape=(28, 28, 1))
    t = tf.keras.layers.SeparableConv2D(32, (2, 2), activation='relu', input_shape=(28, 28, 1))(input)
    t = tf.keras.layers.MaxPooling2D((2, 2))(t)
    t = tf.keras.layers.SeparableConv2D(64, (2, 2), activation='relu')(t)
    # h,w,c->h,w,1
    conv = tf.keras.layers.MaxPooling2D((2, 2))(t)
    # 对channel做attention，先用global averge层去掉位置信息
    channel_global_average = tf.keras.layers.GlobalAveragePooling2D()(conv)
    # 每个channel的权重
    attention_weights = tf.keras.layers.Dense(64, activation="relu", name="attention_vec")(channel_global_average)
    attention_weights = tf.keras.layers.Dense(64, activation="softmax", name="attention_vec")(channel_global_average)
    # h,w,c x h,w,1 -> h,w,c x h,w,1 每个位置乘以相应的注意权重
    t = tf.keras.layers.Multiply()([t, attention_weights])
    # 对空间求和，求和后效果很差
    # t = tf.keras.layers.Lambda(lambda x: K.sum(x, axis=2))(t)

    '''
    attention_weights = tf.keras.layers.Dense(64, activation="softmax", name="attention_vec")(t)
    t = tf.keras.layers.Multiply()([t, attention_weights])
    '''
    t = tf.keras.layers.Flatten()(t)
    t = tf.keras.layers.Dense(64, activation='relu')(t)
    output = tf.keras.layers.Dense(10, activation='softmax')(t)

    train_model = tf.keras.models.Model(input, output)
    visual_model = tf.keras.models.Model(input, [conv, attention_weights])

    train_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    train_model.fit(x_train.reshape(*x_train.shape, 1), y_train, batch_size=32, epochs=10,
                    validation_data=(x_test.reshape(*(x_test.shape), 1), y_test))

    layer_outputs = visual_model(x_test[0].reshape(1, *(x_test[0].shape), 1))
