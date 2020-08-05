import time

import tensorflow as tf

import numpy as np
from matplotlib import pyplot as plt

import IPython.display as display
import PIL.Image

from tensorflow.keras.preprocessing import image

# url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
url = 'origin6.jpg'


# Download an image and read it into a NumPy array.
def download(url, max_dim=None):
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)


def process_image(image_path, max_dim=None):
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)


# Normalize an image
def deprocess(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)


# Display an image
def show(img):
    plt.imshow(img)
    # display.display(PIL.Image.fromarray(np.array(img)))


# 用于在梯度带内计算损失
def calc_loss(img, model):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return tf.reduce_sum(losses)


# tf.Module 封装Autograph，避免变量被回收等问题
# 最终效果类似1.x里的tf.function
class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
                tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.int32),
                tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    # 这里传入所有参数都需要是tf格式的
    # 这里一次执行了steps 步，tf.function可以有循环，判断
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # This needs gradients relative to `img`
                # `GradientTape` only watches `tf.Variable`s by default
                tape.watch(img)
                loss = calc_loss(img, self.model)

            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)

            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

        return loss, img


def run_deep_dream_simple(img, steps=100, step_size=0.01):
    # Convert from uint8 to the range expected by the model.
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0

    # 这里不是特别明白为什么把这些转化成张量。。
    while steps_remaining:
        # 大于100步，执行100步，否则执行完
        if steps_remaining > 100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps

        # 这里因为数输入到tf.module，以autograph的形式运行，不是eager动态图，所以所有参数都转成张量
        loss, img = deepdream(img, run_steps, tf.constant(step_size))

        # display.clear_output(wait=True)
        # show(deprocess(img))
        print("Step {}, loss {}".format(step, loss))

    result = deprocess(img)
    # display.clear_output(wait=True)
    # show(result)

    return result


def random_roll(img, maxroll):
    # Randomly shift the image to avoid tiled boundaries.
    shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
    shift_down, shift_right = shift[0], shift[1]
    img_rolled = tf.roll(tf.roll(img, shift_right, axis=1), shift_down, axis=0)
    return shift_down, shift_right, img_rolled


# 分割图片分多次计算
class TiledGradients(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
                tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.int32),)
    )
    def __call__(self, img, tile_size=512):
        shift_down, shift_right, img_rolled = random_roll(img, tile_size)

        # Initialize the image gradients to zero.
        gradients = tf.zeros_like(img_rolled)

        # Skip the last tile, unless there's only one tile.
        xs = tf.range(0, img_rolled.shape[0], tile_size)[:-1]
        if not tf.cast(len(xs), bool):
            xs = tf.constant([0])
        ys = tf.range(0, img_rolled.shape[1], tile_size)[:-1]
        if not tf.cast(len(ys), bool):
            ys = tf.constant([0])

        for x in xs:
            for y in ys:
                # Calculate the gradients for this tile.
                with tf.GradientTape() as tape:
                    # This needs gradients relative to `img_rolled`.
                    # `GradientTape` only watches `tf.Variable`s by default.
                    tape.watch(img_rolled)

                    # Extract a tile out of the image.
                    img_tile = img_rolled[x:x + tile_size, y:y + tile_size]
                    loss = calc_loss(img_tile, self.model)

                # Update the image gradients for this tile.
                gradients = gradients + tape.gradient(loss, img_rolled)

        # Undo the random shift applied to the image and its gradients.
        gradients = tf.roll(tf.roll(gradients, -shift_right, axis=1), -shift_down, axis=0)

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8

        return gradients


def run_deep_dream_with_octaves(img, steps_per_octave=100, step_size=0.01,
                                octaves=range(-2, 3), octave_scale=1.3):
    base_shape = tf.shape(img)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # 对应网络需要的前处理
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    get_tiled_gradients = TiledGradients(dream_model)
    initial_shape = img.shape[:-1]
    img = tf.image.resize(img, initial_shape)
    for octave in octaves:
        # Scale the image based on the octave
        new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32) * (octave_scale ** octave)
        img = tf.image.resize(img, tf.cast(new_size, tf.int32))

        for step in range(steps_per_octave):
            gradients = get_tiled_gradients(img)
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

            if step % 10 == 0:
                # display.clear_output(wait=True)
                # show(deprocess(img))
                print("Octave {}, Step {}".format(octave, step))

    result = deprocess(img)
    return result


if __name__ == '__main__':

    # Downsizing the image makes it easier to work with.
    original_img = process_image('../../keras/generating/origin2.jpg', max_dim=500)
    # original_img = PIL.Image.open('origin1.jpg')
    # show(original_img)

    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

    # Maximize the activations of these layers
    names = ['mixed3', 'mixed5']
    layers = [base_model.get_layer(name).output for name in names]

    # Create the feature extraction model
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
    deepdream = DeepDream(dream_model)
    dream_img = run_deep_dream_simple(img=original_img, steps=100, step_size=0.01)
    plt.imsave('deep_dream_1.jpg', dream_img.numpy())

    # --------------------------------------------------
    start = time.time()

    OCTAVE_SCALE = 1.30

    img = tf.constant(np.array(original_img))
    base_shape = tf.shape(img)[:-1]
    float_base_shape = tf.cast(base_shape, tf.float32)
    # 根据OCTAVE_SCALE调整图片的大小，再提升
    for n in range(-2, 3):
        new_shape = tf.cast(float_base_shape * (OCTAVE_SCALE ** n), tf.int32)
        img = tf.image.resize(img, new_shape).numpy()
        img = run_deep_dream_simple(img=img, steps=50, step_size=0.01)

    # display.clear_output(wait=True)
    img = tf.image.resize(img, base_shape)
    img_add_detail = tf.image.convert_image_dtype(tf.cast(img, tf.float32) / 255.0, dtype=tf.uint8)
    # show(img)
    plt.imsave('deep_dream_2.jpg', img_add_detail.numpy())
    '''
    # 这里的图片先缩放在逐步放大，最后调回原来尺寸，原有内容实际上已经模糊（因为最后的尺寸比原图大所以是模糊，不然是像素化），但deepdream的纹路使其不明显
    # 可以通过将原图进行同样的操作后的得到的图与原图进行相减，将差值加到img的输出上来进行弥补，另一版有该操作
    original_img_octave = tf.constant(np.array(original_img))
    for n in range(-2, 3):
        new_shape = tf.cast(float_base_shape * (OCTAVE_SCALE ** n), tf.int32)
        original_img_octave = tf.image.resize(original_img_octave, new_shape).numpy()
    original_img_octave = tf.image.resize(original_img_octave, base_shape)
    lost_detail = tf.cast(tf.constant(np.array(original_img)), tf.float32) - original_img_octave
    # 添加细节后，会多一些轮廓噪点，效果不好
    img_add_detail = img + lost_detail
    img_add_detail = tf.image.convert_image_dtype(tf.cast(img_add_detail, tf.float32) / 255.0, dtype=tf.uint8)
    '''
    # show(img)
    plt.imsave('deep_dream_2_add_detail.jpg', img_add_detail.numpy())

    end = time.time()
    print(end - start)

    # 分割图片，然后计算run_deep_dream_with_octaves内部调用了random_roll进行分割
    shift_down, shift_right, img_rolled = random_roll(np.array(original_img), 512)
    # show(img_rolled)
    plt.imsave('deep_dream_img_rolled.jpg', img_rolled.numpy())

    img = run_deep_dream_with_octaves(img=original_img, step_size=0.01)

    # display.clear_output(wait=True)
    img = tf.image.resize(img, base_shape)
    img = tf.image.convert_image_dtype(img / 255.0, dtype=tf.uint8)
    # show(img)
    plt.imsave('deep_dream_3.jpg', img.numpy())
