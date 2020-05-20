import os

import tensorflow.keras as keras
from keras_preprocessing.image import array_to_img
from matplotlib import image
from tensorflow.keras import layers
import numpy as np

if __name__ == '__main__':

    latent_dim = 32
    height = 32
    width = 32
    channels = 3
    generator_input = keras.Input(shape=(latent_dim,))
    x = layers.Dense(128 * 16 * 16)(generator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((16, 16, 128))(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    # 生成一个大小为32×32 的单通道特征图（即CIFAR10 图像的形状）
    x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
    generator = keras.models.Model(generator_input, x)
    generator.summary()

    # discriminator 模型，它接收一张候选图像（真实的或合成的）作为输入，并将其划分到这两个类别之一：“生成图像”或“来自训练集的真实图像
    discriminator_input = layers.Input(shape=(height, width, channels))
    x = layers.Conv2D(128, 3)(discriminator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    # dropout 很重要
    x = layers.Dropout(0.4)(x)
    # 分类层
    x = layers.Dense(1, activation='sigmoid')(x)
    discriminator = keras.models.Model(discriminator_input, x)
    discriminator.summary()
    discriminator_optimizer = keras.optimizers.RMSprop(
        lr=0.0008,
        clipvalue=1.0,
        decay=1e-8)
    discriminator.compile(optimizer=discriminator_optimizer,
                          loss='binary_crossentropy')

    # 将判别器权重设置为不可训练
    # （仅应用于gan 模型）
    discriminator.trainable = False
    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.models.Model(gan_input, gan_output)
    gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
    gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

    (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()

    # 选择青蛙图像（类别编号为6）
    x_train = x_train[y_train.flatten() == 6]
    x_train = x_train.reshape((x_train.shape[0],) +
                              (height, width, channels)).astype('float32') / 255.
    iterations = 10000
    batch_size = 20
    save_dir = 'GAN'
    start = 0
    for step in range(iterations):

        # 在潜在空间中采样随机点
        random_latent_vectors = np.random.normal(size=(batch_size,
                                                       latent_dim))
        # 将这些点解码为虚假图像
        generated_images = generator.predict(random_latent_vectors)

        # 将这些虚假图像与真实图像合在一起
        stop = start + batch_size
        real_images = x_train[start: stop]
        combined_images = np.concatenate([generated_images, real_images])
        labels = np.concatenate([np.ones((batch_size, 1)),
                                 np.zeros((batch_size, 1))])
        # 向标签中添加随机噪声，这是一个很重要的技巧
        labels += 0.05 * np.random.random(labels.shape)

        # 训练判别器
        d_loss = discriminator.train_on_batch(combined_images, labels)
        random_latent_vectors = np.random.normal(size=(batch_size,
                                                       latent_dim))

        #合并标签，全部是“真实图像”（这是在撒谎）
        misleading_targets = np.zeros((batch_size, 1))

        # 通过gan 模型来训练生成器（ 此时冻结判别器权重）
        a_loss = gan.train_on_batch(random_latent_vectors,
                                    misleading_targets)
        start += batch_size
        if start > len(x_train) - batch_size:
            start = 0

        if step % 100 == 0:
            gan.save_weights('gan.h5')
            print('discriminator loss:', d_loss)
            print('adversarial loss:', a_loss)
            img = array_to_img(generated_images[0] * 255., scale=False)
            img.save(os.path.join(save_dir,
                                  'generated_frog' + str(step) + '.png'))
            img = array_to_img(real_images[0] * 255., scale=False)
            img.save(os.path.join(save_dir,
                                  'real_frog' + str(step) + '.png'))
