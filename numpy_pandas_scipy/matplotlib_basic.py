import numpy as np
from matplotlib import pyplot as plt

# 直线
x = np.arange(1, 11)
y1 = 2 * x + 5
# numpy x*x只是矩阵中的对应元素相乘，等同于np.multiply()，不是矩阵相乘, 不是矩阵相乘是np.dot
y2 = 2 * x * x - 4
plt.title("菜鸟教程 - 测试")
plt.plot(x, y1)
plt.plot(x, y2)
plt.show()

# 正弦曲线
# 计算正弦曲线上点的 x 和 y 坐标
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)
plt.title("sine wave form")
# 使用 matplotlib 来绘制点
plt.plot(x, y)
plt.show()

# subplot() 函数允许你在同一图中绘制不同的东西。
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
# 建立 subplot 网格，高为 2，宽为 1
# 激活第一个 subplot
plt.subplot(2, 1, 1)
# 绘制第一个图像
plt.plot(x, y_sin)
plt.title('Sine')
# 将第二个 subplot 激活，并绘制第二个图像
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')
# x轴范围
plt.xlim(0, 20)
# 展示图像
plt.show()

x = np.random.randint(0, 10, (4, 10))
plt.bar(np.arange(len(x)), x)
plt.show()

plt.barh(np.arange(len(x)), x)
plt.show()


# numpy.histogram()
# numpy.histogram() 函数是数据的频率分布的图形表示。 水平尺寸相等的矩形对应于类间隔，称为 bin，变量 height 对应于频率。

a = np.array([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27])

plt.hist(a, bins=[0, 20, 40, 60, 80, 100])
#  20 指定分格子，density=True 累计, cumulative=True 0-1 概率分布
plt.hist(a, 20, density=True, cumulative=True)
plt.title("histogram")
plt.show()

# 第二维度会被分开统计
b = np.random.random(size=(100, 8))
plt.hist(b, 4, density=True, cumulative=True)
plt.show()

# 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# (这种方法是通过X，Y和Z构建一个矩阵，XYZ各取一个点，这里有四个点）
X = [1, 1, 2, 2]
Y = [3, 4, 4, 3]
Z = [1, 2, 1, 1]

# 绘制曲面 (四个点可以确定一个四面体)
ax.plot_trisurf(X, Y, Z)
# 绘制散点图
ax.scatter(X, Y, Z)
plt.show()

# 多图
x = np.random.randint(-100, 100, 100)
fig = plt.figure()
ax1 = fig.add_subplot(221)  # 2*2的图形 在第一个位置
ax1.plot(x.cumsum())
ax2 = fig.add_subplot(222)
ax2.hist(np.random.randint(1, 100, 100), bins=20)
ax3 = fig.add_subplot(223)
ax3.scatter(np.random.randint(1, 100, 100), np.random.randint(1, 100, 100), s=75, alpha=.5)
ax3 = fig.add_subplot(224)
ax3.barh(np.log(abs(x.cumsum())))
plt.show()

# linspace的间隔数量是101-1
x = np.linspace(0, 10, 101)
# 直接使用plt.subplots, axes是个numpy数组，形状相同与subplots输入形状相同
fig, axes = plt.subplots(2, 2)
axes[0][0].plot(x, x * x)
axes[0][1].plot(x, (10 - x) * (10 - x))
axes[1][0].plot(x, 100 - x * x)
axes[1][1].plot(x, 100 - (10 - x) * (10 - x))
# 调整图像之间的间距
# plt.subplots_adjust(right=0, left=0, top=0, bottom=0)
plt.subplots_adjust(wspace=0, hspace=0)
# x轴刻度范围
axes[0][0].get_xlim()
axes[0][0].set_xlim((-0.5, 20.5))

# 设置刻度
axes[1][0].set_xticks([5, 10])
# 图例
axes[1][1].legend(loc='best')
# 文字
axes[0][1].text(5, 50, 'Hello world!', family='monospace', fontsize=10)

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
