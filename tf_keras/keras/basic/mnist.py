from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# pycharm不会将当前文件目录自动加入自己的sourse_path。右键make_directory as-->Sources Root将当前工作的文件夹加入source_path就可以了。
from tf_keras.keras import tools

# 屏蔽 INFO + WARNING 	输出 ERROR + FATAL
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 出现Failed to get convolution algorithm. This is probably because cuDNN fail


# Load and prepare the MNIST dataset. Convert the samples from integers to floating-point numbers:
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

'''

#绘制数据

fig, ax = plt.subplots(
    nrows=2,
    ncols=5,
    sharex=True,
    sharey=True, )

ax = ax.flatten()
for i in range(10):
    #i代表y输出
    img = x_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

'''


# plt.imshow(x_test[0], cmap='Greys', interpolation='nearest')


def simple_network():
    # Build the tf.keras.Sequential model by stacking layers. Choose an optimizer and loss function for training:
    model = tf.keras.models.Sequential([
        # Flatten layer & Dense layer
        # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到(Convolution)全连接层(Dense)的过渡
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        # 弃权
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # sparse_categorical_crossentropy多类对视损失： -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))
    # metrics 评估标准，可以用自定义函数，或者其他指标
    # Adam 是一种可以替代传统随机梯度下降过程的一阶优化算法，它能基于训练数据迭代地更新神经网络权重
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# 加入卷积层，速度较慢
# 消减了网络规模，卷积核大小，效果反而变好了。。。
def conv_network():
    model = tf.keras.models.Sequential()
    # 当使用Conv2D层作为第一层时，应提供input_shape参数。例如input_shape = (128,128,3)代表128*128的彩色RGB图像
    # 32,64,64这些称为深度
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

    # 注意通过summry可以看到Flatten本身就是有数量的，应该可以看成一层全连接
    # 全连接之间的参数数量巨大，conv2d似乎可以减小flatten时的数量，或许可以理解为信息的压缩
    model.add(tf.keras.layers.Flatten())

    # Dropout弃权，避免过拟合
    # model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(64, activation='relu'))

    # 输出层激活函数activation='softmax' 不加默认是线性，效果极差
    # 输出层10个神经元必须要加。。。之前忘记了,
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # categorical_crossentropy如果你有10个类别，每一个样本的标签应该是一个10维的向量，该向量在对应有值的索引位置为1其余为0。
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# 加入卷积层，速度较慢
# 消减了网络规模，卷积核大小，效果反而变好了。。。
def separable_conv2D_network():
    model = tf.keras.models.Sequential()
    # 当使用Conv2D层作为第一层时，应提供input_shape参数。例如input_shape = (128,128,3)代表128*128的彩色RGB图像
    # 32,64,64这些称为深度
    model.add(tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu'))

    # 全连接之间的参数数量巨大，conv2d似乎可以减小flatten时的数量，或许可以理解为信息的提取，压缩
    model.add(tf.keras.layers.Flatten())

    # Dropout弃权，避免过拟合
    # model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(64, activation='relu'))

    # 输出层激活函数activation='softmax' 不加默认是线性，效果极差
    # 输出层10个神经元必须要加。。。之前忘记了,
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # categorical_crossentropy如果你有10个类别，每一个样本的标签应该是一个10维的向量，该向量在对应有值的索引位置为1其余为0。
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# 自定义回调，当验证集准确率大于某个值停止
class StoppingAtAccuracy(tf.keras.callbacks.Callback):
    def __init__(self,value):
        self.value = value

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if logs and logs.get('val_accuracy') > self.value:
            self.model.stop_training = True


# 测试我的图，可能是不是手写和预处理的问题，无法正确预测
def test_my_pic(model, num: int):
    img = mpimg.imread('{}.bmp'.format(num))
    img = (1.0 - img[:, :, 0] / 255)
    print(img)
    ans = model.predict(img.reshape(1, 28, 28, 1))
    print(ans)
    print("input:{} output:{}".format(num, ans.argmax() + 1))
    return ans.argmax() + 1


if __name__ == '__main__':
    model1 = simple_network()
    model2 = conv_network()
    model3 = separable_conv2D_network()
    # Train and evaluate the model:

    model1.summary()
    model2.summary()
    model3.summary()
    # model1.fit(x_train, y_train, epochs=5)
    # model1.evaluate(x_test, y_test, verbose=2)

    '''
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(
            monitor='acc',
            patience=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            monitor='val_loss',
            save_best_only=True,
        )
    ]
    '''

    # batch_size 默认32
    model3.fit(x_train.reshape(list(x_train.shape) + [1]), y_train, batch_size=32, epochs=10,
               callbacks=[StoppingAtAccuracy(0.99)],
               validation_data=(x_test.reshape(list(x_test.shape) + [1]), y_test))

    model3.evaluate(x_test.reshape(list(x_test.shape) + [1]), y_test, verbose=2)

    # 由于Flatten(input_shape=(28, 28)) 强制规定了输入要28*28， 所以这里转成1*28*28，而不是724，1应该代表数据数量
    # 这里输出的是个1*10矩阵，其中第7个输出接近1，其他极小，这样的结果符合损失策略
    model3.predict(x_test[0].reshape(1, 28, 28, 1)).argmax()
    tools.plot_loss(model3.history.history)
    # tools.plot_accuracy(model2.history.history)
