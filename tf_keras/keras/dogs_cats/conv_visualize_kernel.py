from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow_core.python.keras import Model

# from tf_keras.keras.tools import use_proxy

# use_proxy()


model = VGG16(weights='imagenet', include_top=False)


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 通过梯度上升，使layer_name层卷积核filter_index通道输出最大，从而将其可视化
def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output

    # 临时生成一个模型用于输出所需要的层
    temp_model = Model(model.input, layer_output)

    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    input_img_data = tf.constant(input_img_data.astype('float32'))
    step = 1.
    for i in range(40):
        # print(i)
        # print(type(input_img_data))
        # print('----------------------------------------------------------------------------')
        with tf.GradientTape() as tape:
            tape.watch([input_img_data])
            output = temp_model(input_img_data)
            loss = K.mean(output[:, :, :, filter_index])
        grads = tape.gradient(loss, [input_img_data])[0]
        grads_value = grads / (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        # 这种写法不行
        # input_img_data += grads_value * step
        input_img_data += grads_value * step
    # 这里需要
    img = input_img_data[0].numpy()
    return deprocess_image(img)


layer_name = 'block1_conv1'
filter_index = 0
size = 64
margin = 5
results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        # print(filter_img)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

plt.figure(figsize=(20, 20))
plt.imshow(results.astype('int'))
