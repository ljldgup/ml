import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

'''
VAE 不是将输入图像压缩成潜在空间中的固定编码，而是将图像转换为统计分布的参数，
即平均值和方差。本质上来说，这意味着我们假设输入图像是由统计过程生成的，在编码和解
码过程中应该考虑这一过程的随机性。然后，VAE 使用平均值和方差这两个参数来从分布中随
机采样一个元素，并将这个元素解码到原始输入
'''


class CustomVariationalLayer(keras.layers.Layer):
    # 该层应该是一个自定义的过滤器
    # 该层后续创建的时候输入 input_img, z_decoded
    # CustomVariationalLayer()([input_img, z_decoded])
    # 损失在vae_loss定义，并在call中调用
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)

        # 内容损失
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)

        # KL损失是X中个体X〜N（μ，σ²）与标准正态分布之间所有KL分支的总和
        # 这种损失鼓励编码器将所有编码（对于所有类型的输入，例如所有MNIST数字号）均匀地分布在潜在空间的中心周围。 如果它试图通过把它们聚集到特定的地区而远离原样本来“作弊”，将会受到惩罚
        kl_loss = -5e-4 * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon


if __name__ == '__main__':
    img_shape = (28, 28, 1)
    batch_size = 16
    latent_dim = 2

    input_img = keras.Input(shape=img_shape)
    x = layers.Conv2D(32, 3,
                      padding='same', activation='relu')(input_img)
    x = layers.Conv2D(64, 3,
                      padding='same', activation='relu',
                      strides=(2, 2))(x)
    x = layers.Conv2D(64, 3,
                      padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3,
                      padding='same', activation='relu')(x)

    shape_before_flattening = K.int_shape(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)

    # 输入图像最终被编码为这两个参数
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    z = layers.Lambda(sampling)([z_mean, z_log_var])

    decoder_input = layers.Input(K.int_shape(z)[1:])
    x = layers.Dense(np.prod(shape_before_flattening[1:]),
                     activation='relu')(decoder_input)
    x = layers.Reshape(shape_before_flattening[1:])(x)

    # 逆卷积操作
    x = layers.Conv2DTranspose(32, 3,
                               padding='same',
                               activation='relu',
                               strides=(2, 2))(x)
    x = layers.Conv2D(1, 3,
                      padding='same',
                      activation='sigmoid')(x)
    decoder = Model(decoder_input, x)
    z_decoded = decoder(z)

    # loss定义在CustomVariationalLayer 层， 由input_img, z_decoded计算得到
    y = CustomVariationalLayer()([input_img, z_decoded])

    vae = Model(input_img, y)
    vae.compile(optimizer='rmsprop', loss=None)
    (x_train, _), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    # 增加一个维度
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape + (1,))

    # 注意这里训练的是minst图像 到 z_mean, z_log_var 的编码及解码过程，所以y为None
    # loss 尤解码后的图像和输入图像计算得到
    vae.fit(x=x_train, y=None,
            shuffle=True,
            epochs=10,
            batch_size=batch_size,
            validation_data=(x_test, None))

    vae.summary()
    n = 15
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))

    # Φ(x) 为 累积概率密度函数，也即 cdf
    # Φ−1(x)，通过norm.ppf(x)
    # norm.ppf(x)  标准正太分布值 < norm.ppf(x) 的概率为x
    # 所以这里是一个二维5%-%95的标准正太分布的值的网格，用于解码成对应的数字图片
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])

            # np.tile 把z_sample重复batch_size次生成一个新array, reshape后作为decoder输入层的
            z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)

            # 注意这里是从decoder来predict的，不是从vae，decoder的输入层就是z_mean，z_log_var
            # 只使用了部分训练的模型作为预测结果
            x_decoded = decoder.predict(z_sample, batch_size=batch_size)

            # 这里用0还是别的一样，因为batch内输入都是一样
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
