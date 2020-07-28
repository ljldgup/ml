from sklearn.datasets import load_sample_image
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

# Load sample images
# 这里默认就是这两张图。。
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower])
batch_size, height, width, channels = images.shape
# Create 2 filters
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1  # vertical line
filters[3, :, :, 1] = 1  # horizontal line
outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")
plt.imshow(outputs[0, :, :, 1], cmap="gray")  # plot 1st image's 2nd
plt.show()

# 通道维度的max pool keras 未提供，可通过tf实现
output = tf.nn.max_pool(images, ksize=(1, 1, 1, 3), strides=(1, 1, 1, 3), padding="valid")

depth_pool = tf.keras.layers.Lambda(
    lambda x: tf.nn.max_pool(x, ksize=(1, 1, 1, 3), strides=(1, 1, 1, 3),
                             padding="valid"))
