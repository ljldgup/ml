import numpy as np
from sklearn.datasets import make_blobs


def gen_random_points(data_size):
    return np.random.randint(100, size=(data_size, 2))


def gen_cycle_points(data_size):
    return np.c_[np.sin(np.linspace(0, 2 * np.pi * (data_size - 1) / data_size, data_size)),
                 np.cos(np.linspace(0, 2 * np.pi * (data_size - 1) / data_size, data_size))]


def get_max_divider(num):
    sqrt_num = np.sqrt(num)
    for i in reversed(range(int(sqrt_num) + 1)):
        if divmod(num, i)[1] == 0:
            return i


def gen_rectangle(data_size):
    line_num = get_max_divider(data_size)
    x = np.c_[[np.linspace(1, line_num, line_num) for i in range(data_size // line_num)]][..., np.newaxis]
    y = np.c_[[np.linspace(1, data_size // line_num, data_size // line_num) for i in range(line_num)]][..., np.newaxis]
    return np.concatenate([x, y.swapaxes(0, 1)], axis=2).reshape(-1, 2)


def gen_blob(data_size):
    points, label = make_blobs(n_samples=data_size, cluster_std=4, n_features=2, centers=4)
    return points


def add_noise(points):
    return points + 3 * points.std()


type_map = {1: gen_random_points, 2: gen_cycle_points, 3: gen_blob, 4: gen_rectangle}


def gen_points(type: int, data_size: int, with_noise=False):
    points = type_map[type](data_size)
    if with_noise:
        points = add_noise(points)
    return points


def gen_dependency(data_size, dependency_size=2):
    if dependency_size == 0:
        return None
    return np.stack([np.random.choice(np.arange(data_size), size=2, replace=False) for i in range(dependency_size)])


def gen_importance(data_size, importance_size=2, fill_num=2):
    importance_matrix = np.zeros(shape=(data_size, importance_size))
    if fill_num:
        position = np.array(
            [np.random.choice(np.arange(data_size), size=(importance_size), replace=False) for i in range(fill_num)])
        importance_value = np.random.choice(np.arange(1, 10), size=(fill_num, importance_size))
        importance_matrix[position, np.arange(importance_size)] = importance_value
    return importance_matrix


def gen_head_position(data_size):
    return np.random.randint(data_size)


def get_test_data():
    distance = np.array([[0, 10, 25, 24, 17, 8],
                         [10, 0, 10, 26, 19, 22],
                         [25, 10, 0, 11, 20, 26],
                         [24, 16, 11, 0, 5, 13],
                         [17, 26, 20, 5, 0, 6],
                         [8, 22, 26, 13, 6, 0]])
    # 最小路径
    # distance = np.array([[0., 10., 20., 15., 21.],
    #                      [10., 0., 17., 19., 22.],
    #                      [20., 17., 0., 5., 11.],
    #                      [15., 22., 5., 0., 6.],
    #                      [21., 22., 11., 6., 0.]])

    points = np.random.random(size=(len(distance), 2))

    dependency = np.array([[2, 4]]) - 1
    # dependency = None
    # 无重要度，
    # importance = np.zeros(shape=(len(distance), 2))
    importance = np.array([[0, 0, 10, 0, 5, 0], [0, 0, 0, 5, 0, 0]]).swapaxes(0, 1)
    # head_position = None
    head_position = 0

    return points, distance, dependency, importance, head_position
