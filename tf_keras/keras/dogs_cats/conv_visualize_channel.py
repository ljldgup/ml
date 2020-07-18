import random

from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing import image
import os

def plot_augment_image(train_dir):
    train_cats_dir = os.path.join(train_dir, 'cats')
    fnames = [os.path.join(train_cats_dir, fname) for
              fname in os.listdir(train_cats_dir)]
    img_path = fnames[random.randint(0, len(fnames)-1)]
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i % 4 == 0:
            break
    plt.show()


def create_img_tensor(test_dir):
    img_path = os.path.join(test_dir, r'cats\cat.{}.jpg'.format(random.randint(1501, 1999)))
    plt.imshow(plt.imread(img_path))

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    #plt.imshow(img_tensor[0])
    return img_tensor


def show_channel_image(model, img_tensor):
    # 生成一个输出前八层卷积层的模型
    layer_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                # 获取col * images_per_row + row的通道输出
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    return activations


if __name__ == '__main__':
    original_dataset_dir = r'D:\tmp\datesets\python_ml_data\data\kaggle_original_data'
    base_dir = r'D:\tmp\cats_and_dogs_small'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2,
                                 height_shift_range=0.2, shear_range=0.2,
                                 zoom_range=0.2, horizontal_flip=True,
                                 fill_mode='nearest')
    #plot_augment_image(train_dir)

    # 这里由于模型融合的问题前面vgg16读不到。。。使用了自己训练的卷积层
    model = models.load_model('cats_and_dogs_small_2.h5')
    img_tensor = create_img_tensor(test_dir)
    show_channel_image(model, img_tensor)
