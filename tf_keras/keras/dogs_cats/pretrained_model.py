# 卷积基输出保存成硬盘中的Numpy 数组，然后用这个数据作为输入，输入到独立的密集连接分类器

import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers

from tf_keras.keras import tools


# 调用conv_base 模型的predict 方法来从这些图像中提取特征，作为自定义模型的训练输入
# 提取的特征形状为(samples, 4, 4, 512)
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


def build_model(conv_base=None):
    model = models.Sequential()
    if conv_base:
        # 在底部添加 Dense 层来扩展已有模型（即 conv_base），并在输入数据上端到端地运行整个模型
        # 可以使用数据增强
        model.add(conv_base)
        model.add(layers.Dense(256, activation='relu'))
    else:
        # 卷积基输出保存成硬盘中的Numpy 数组，然后用这个数据作为输入，输入到独立的密集连接分类器
        model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model


if __name__ == '__main__':
    original_dataset_dir = r'D:\tmp\datesets\python_ml_data\data\kaggle_original_data'
    base_dir = r'D:\tmp\cats_and_dogs_small'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    datagen = ImageDataGenerator(rescale=1. / 255)
    batch_size = 20

    # include_top 指定模型最后是否包含密集连接分类器。默认情况下，这个密集连接分类器对应于ImageNet 的1000 个类别
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))
    conv_base.summary()

    # 这里输出的train future是已经通过VGG16计算输出的结果，所以后面build model没有传入VGG16模型
    # 这种模型训练快，但是不能使用数据增强，因为VGG16部分模型是冻结的
    train_features, train_labels = extract_features(train_dir, 2000)
    validation_features, validation_labels = extract_features(validation_dir, 1000)
    test_features, test_labels = extract_features(test_dir, 1000)
    train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
    test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

    model = build_model()
    history = model.fit(train_features, train_labels, epochs=30, batch_size=20,
                        validation_data=(validation_features, validation_labels))

    tools.plot_loss(history.history)
    tools.plot_accuracy(history.history)
