from matplotlib.image import imsave
from scipy.linalg import misc
from tensorflow.keras.applications import inception_v3
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import scipy
from tensorflow.python.keras.backend import image_data_format

# 使用了tf.gradients
tf.compat.v1.disable_eager_execution()


def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

# 梯度上升过程，用于求最大值，希望图片和原始图片有一定差距
def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x


def resize_img(img, size):
    img = np.copy(img)
    # 对高度和宽度缩放
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)


def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    imsave(fname, pil_img)


def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    # 增加一个0维度
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    if image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


if __name__ == '__main__':
    K.set_learning_phase(0)
    model = inception_v3.InceptionV3(weights='imagenet',
                                     include_top=False)
    # 可以改变,增删贡献层，效果会有区别
    layer_contributions = {
        'conv2d_40': 1.,
        'conv2d_60': 3.,
        'mixed8': 2.,
        'mixed10': 1.5,
    }
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # 源代码是K.variable, 但是似乎用的是tf.Variable, 使用assign_add相关函数后无法求梯度。。。。。、
    # 改成tf.constant后可以运行，constant，Variable，tensor含义有区别需要注意
    # constant一般定义输入，variable一般定义权重，tensor则是中间结果
    # loss = tf.constant(0.)

    # 将assign_add 改成 tf.add后可以运行
    # assign_add是改变了变量值，而tf.add是生成一个tensor，代表了一个表达式
    # 貌似只有用tensor,求梯度才有效果，用variable返回[None]
    loss = K.variable(0.)
    for layer_name in layer_contributions:
        coeff = layer_contributions[layer_name]
        activation = layer_dict[layer_name].output
        scaling = K.prod(K.cast(K.shape(activation), 'float32'))
        # 注意这个 K.sum, K.prod对应 tf.reduce_sum, tf.reduce_prod， 其他可以直接改
        loss = tf.add(loss, coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling)
        # loss += (coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling)

    dream = model.input
    # 求得是输入梯度
    grads = tf.gradients(loss, dream)[0]
    grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-7)
    outputs = [loss, grads]

    # K.function 与tf.function不同
    fetch_loss_and_grads = K.function([dream], outputs)

    # 梯度上升的步长, 缩放次数, 缩放比例,梯度上身次数

    #提高,减小步长值对应效果粗糙，精细
    step = 0.03
    num_octave = 3
    octave_scale = 1.4
    iterations = 20

    # 如果损失增大到大于10，我们要中断梯度上升过程，以避免得到丑陋的伪影
    max_loss = 10.

    base_image_path = '1.jpg'
    img = preprocess_image(base_image_path)

    # 准备一个由形状元组组成的列表，它定义了运行梯度上升的不同尺度
    original_shape = img.shape[1:3]
    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        # 缩放后的尺寸
        shape = tuple([int(dim / (octave_scale ** i))
                       for dim in original_shape])
        successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]

    original_img = np.copy(img)
    shrunk_original_img = resize_img(img, successive_shapes[0])
    for shape in successive_shapes:
        print('Processing image shape', shape)
        img = resize_img(img, shape)
        img = gradient_ascent(img,
                              iterations=iterations,
                              step=step,
                              max_loss=max_loss)
        upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
        # 由于缩放，在放大造成了图像内容损失，这里进行弥补
        same_size_original = resize_img(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img
        img += lost_detail
        shrunk_original_img = resize_img(original_img, shape)
        save_img(img, fname='dream_at_scale_' + str(shape) + '.png')
    save_img(img, fname='final_dream.png')
