"""
加了dropout，数据增强以后模型不再过拟合，但梯度变慢，原书大约训练了100 epoch。。得到 86左右的正确率
而没有正则化的大约在10轮左右就过拟合，但梯度下降速度快
"""

import os

from tensorflow.keras import models, layers, optimizers
from tf_keras.keras.dogs_cats import preprocess
from tf_keras.keras import tools


def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    return model


if __name__ == '__main__':
    original_dataset_dir = r'D:\tmp\datesets\python_ml_data\data\kaggle_original_data'
    base_dir = r'D:\tmp\cats_and_dogs_small'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    preprocess.copy_files(base_dir, original_dataset_dir)

    model = build_model()
    model.summary()
    train_generator, validation_generator, test_generator = preprocess.get_image_generator(base_dir, enchenced=True)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=50)

    tools.plot_loss(history.history)
    tools.plot_accuracy(history.history)

    #用于绘制中间图像
    model.save('cats_and_dogs_small_2.h5')