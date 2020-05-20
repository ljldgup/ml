import time
import tensorflow as tf
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg19
from tensorflow.keras import backend as K
from matplotlib.image import imsave

# 使用了tf.gradients
tf.compat.v1.disable_eager_execution()


class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # BGR转RGB
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def content_loss(base, combination):
    return K.sum(K.square(combination - base))


def gram_matrix(x):
    # 这里输入的是风格矩阵和，
    # 将通道维度放到最前面
    # batch_flatten将nD张量转换为具有相同0维的2D张量,即把第一个通道维度后面的维度展平
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))

    # n维欧式空间中任意k个向量之间两两的内积所组成的矩阵，称为这k个向量的格拉姆矩阵(Gram matrix)
    # 而Gram计算的实际上是两两特征之间的相关性，哪两个特征是同时出现的，哪两个是此消彼长的等等，
    # 同时，Gram的对角线元素，还体现了每个特征在图像中出现的量，因此，Gram有助于把握整个图像的大体风格。
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, 1:, :img_width - 1, :])
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


if __name__ == '__main__':
    target_image_path = 'target.jpg'
    style_reference_image_path = 'style.jpg'
    width, height = image.load_img(target_image_path).size
    img_height = 400
    img_width = int(width * img_height / height)
    target_image = K.constant(preprocess_image(target_image_path))
    style_reference_image = K.constant(preprocess_image(style_reference_image_path))
    combination_image = K.placeholder((1, img_height, img_width, 3))
    input_tensor = K.concatenate([target_image,
                                  style_reference_image,
                                  combination_image], axis=0)
    model = vgg19.VGG19(input_tensor=input_tensor,
                        weights='imagenet',
                        include_top=False)
    print('Model loaded.')
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # 内容损失使用靠顶层的图， 风格损失使用相对
    content_layer = 'block5_conv2'
    style_layers = [
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1']
    total_variation_weight = 1e-4
    style_weight = 1.
    content_weight = 0.025

    # 定义损失表达
    loss = K.variable(0.)
    layer_features = outputs_dict[content_layer]

    # 这里定义的是几个样本之间的损失，不是之前只有一个样本的损失，风格损失也一样
    target_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = tf.add(loss, content_weight * content_loss(target_image_features,
                                                      combination_features))
    for layer_name in style_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        # 这里不知道为什么第一次会报duplicate node错误，第二次就不会，可能是有内容需要清空
        try:
            sl = style_loss(style_reference_features, combination_features)
        except:
            sl = style_loss(style_reference_features, combination_features)

        loss = tf.add(loss, (style_weight / len(style_layers)) * sl)
    loss = tf.add(loss, total_variation_weight * total_variation_loss(combination_image))

    grads = K.gradients(loss, combination_image)[0]
    fetch_loss_and_grads = K.function([combination_image], [loss, grads])

    evaluator = Evaluator()
    result_prefix = 'my_result'
    iterations = 4

    # 这里采用了target作为combination的初始化，并不是生成的target，实际上应该从白噪声开始更标准，但计算时间太长。
    x = preprocess_image(target_image_path)
    x = x.flatten()
    # x = np.zeros(img_height * img_width * 3)
    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time.time()

        # 注意输入的是evaluator，其参数是x，combination的输入
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                         x,
                                         fprime=evaluator.grads,
                                         maxfun=20)
        print('Current loss value:', min_val)
        img = x.copy().reshape((img_height, img_width, 3))
        img = deprocess_image(img)
        fname = result_prefix + '_at_iteration_%d.png' % i
        imsave(fname, img)
        print('Image saved as', fname)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
