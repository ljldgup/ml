import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 为了ax = fig.gca(projection='3d')
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf
import numpy as np

# 张量是不可变的
print(tf.add(1, 2))
print(tf.add([3, 8], [2, 5]))
print(tf.square(6))
print(tf.reduce_sum([7, 8, 9]))
print(tf.square(3) + tf.square(4))

# numpy()转为numpy数组
tf.constant([1, 2, 3, 4]).numpy()

# 形状和类型
x = tf.matmul([[3], [6]], [[2]])
print(x)
print(x.shape)
print(x.dtype)

# 测试GPU是否
x = tf.random.uniform([3, 3])
print('Is GPU available:')
print(tf.test.is_gpu_available())
print('Is the Tensor on gpu #0:')
print(x.device.endswith('GPU:0'))

t = tf.constant([[1., 2., 3.], [4., 5., 6.]])

t[:, 1:]

t[..., 1, tf.newaxis]

t + 10

tf.square(t)

t @ tf.transpose(t)


def unary_gradient():
    # -------------------一元梯度案例---------------------------
    print("一元梯度")
    x = tf.constant(value=3.0)
    with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
        tape.watch(x)
        y1 = 2 * x
        y2 = x * x + 2
        y3 = x * x + 2 * x

    # 梯度是一个方向，沿该方向下降最快
    # 一元函数实际上方向始终相同
    # 一阶导数
    dy1_dx = tape.gradient(target=y1, sources=x)
    dy2_dx = tape.gradient(target=y2, sources=x)
    dy3_dx = tape.gradient(target=y3, sources=x)
    print("dy1_dx:", dy1_dx)
    print("dy2_dx:", dy2_dx)
    print("dy3_dx:", dy3_dx)

    # # -------------------二元梯度案例---------------------------
    print("二元梯度")
    x = tf.constant(value=3.0)
    y = tf.constant(value=2.0)
    with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
        tape.watch([x, y])
        z1 = x * x * y + x * y
    # 一阶导数
    dz1_dx = tape.gradient(target=z1, sources=x)
    dz1_dy = tape.gradient(target=z1, sources=y)
    dz1_d = tape.gradient(target=z1, sources=[x, y])
    print("dz1_dx:", dz1_dx)
    print("dz1_dy:", dz1_dy)
    print("dz1_d:", dz1_d)
    print("type of dz1_d:", type(dz1_d))


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def himmelblau_gradient():
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    print('x,y range:', x.shape, y.shape)
    X, Y = np.meshgrid(x, y)
    print('X,Y maps:', X.shape, Y.shape)
    Z = himmelblau([X, Y])

    fig = plt.figure('himmelblau')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.view_init(60, -30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

    '''
    解
    f(3.0,2.0)=0.0
    f(−2.805118,3.131312)=0.0
    f(−3.779310,−3.283186)=0.0
    f(3.584428,−1.848126)=0.0
    '''

    # 根据初值不同解不同
    # [1., 0.], [-4, 0.], [-3, 0.]
    x = tf.constant([4., 0.])

    for step in range(200):
        with tf.GradientTape() as tape:
            tape.watch([x])
            y = himmelblau(x)

        # 梯度实际上就是导数
        grads = tape.gradient(y, [x])[0]
        x -= 0.01 * grads

        if step % 20 == 0:
            print('step {}: x = {}, f(x) = {}'
                  .format(step, x.numpy(), y.numpy()))


if __name__ == '__main__':
    unary_gradient()
    print("-" * 50)
    himmelblau_gradient()
