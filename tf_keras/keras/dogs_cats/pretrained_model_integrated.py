# VGG16卷积基参与运算

import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers, optimizers



from tf_keras.keras import tools
from tf_keras.keras.dogs_cats import preprocess


def build_model(conv_base):
    model = models.Sequential()
    # 在顶部添加 Dense 层来扩展已有模型（即 conv_base），并在输入数据上端到端地运行整个模型
    # 可以使用数据增强
    model.add(conv_base)
    # 不加flatten会默认把维度补齐，导致输出是（4,4,1）
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model


# 冻结直到某一层的所有层
# 微调卷积块5
def set_trainable(conv_base):
    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


if __name__ == '__main__':
    base_dir = r'D:\tmp\cats_and_dogs_small'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))
    model = build_model(conv_base)

    train_generator, validation_generator, test_generator = preprocess.get_image_generator(base_dir, enchenced=True)

    # 允许微调
    set_trainable(conv_base)
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=50)

    tools.plot_loss(history.history)
    tools.plot_accuracy(history.history)

    test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)

