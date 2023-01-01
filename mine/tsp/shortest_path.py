from typing import Iterable

import numpy as np
from queue import PriorityQueue


# 为了方便判断，这里返回的路径不包含首尾
def dijkstra(distance: np.ndarray, source: int):
    v_queue = PriorityQueue()
    path_dict = {}
    result = set()

    for i in range(len(distance)):
        # 位置放在钱
        path_dict[i] = (distance[source][i], i, [])
        v_queue.put_nowait(path_dict[i])

    while not v_queue.empty():

        path_distance, target, path = v_queue.get_nowait()
        if target in result:
            continue

        path_dict[target] = (path_distance, target, path)
        result.add(target)

        if len(result) == len(distance):
            break

        for i in filter(lambda v: v not in result, range(len(distance))):
            # print(i, path.target)
            if path_distance + distance[target][i] < path_dict[i][0]:
                v_queue.put_nowait((path_distance + distance[target][i], i, path + [target]))

    return path_dict


# 为了方便判断，这里返回的路径不包含首尾
def floyd(distance: np.ndarray):
    points_size = len(distance)
    shortest_weight = distance
    line_row_num = np.arange(len(distance))
    ones = np.ones_like(line_row_num)

    shortest_paths = [[[] for i in range(points_size)] for j in range(points_size)]
    for k in range(0, len(distance)):
        new_paths = shortest_paths.copy()
        path_with_k = shortest_weight[line_row_num[:, np.newaxis], (ones * k)[np.newaxis, :]] + \
                      shortest_weight[(ones * k)[:, np.newaxis], line_row_num[np.newaxis, :]]

        greater = path_with_k < shortest_weight
        shortest_weight = np.where(greater, path_with_k, shortest_weight)

        for index in np.argwhere(greater):
            new_paths[index[0]][index[1]] = shortest_paths[index[0]][k] + [k] + shortest_paths[k][index[1]]
        shortest_paths = new_paths
    return shortest_weight, shortest_paths


def get_real_path(mapped_path, shortest_paths):

    data_size = len(mapped_path)
    real_path = []

    for i in range(data_size - 1):
        real_path.append(mapped_path[i])
        if shortest_paths[mapped_path[i]][mapped_path[i + 1]]:
            real_path.append(shortest_paths[mapped_path[i]][mapped_path[i + 1]])
    real_path.append(mapped_path[i + 1])
    return real_path


def add_one(object):
    if isinstance(object, Iterable):
        return tuple(map(add_one, object))
    return object + 1


if __name__ == '__main__':
    distance = np.array([[0, 10, np.inf, 24, np.inf, np.inf, np.inf, 10, ],
                         [10, 0, 10, np.inf, 19, 22, 7, np.inf, ],
                         [np.inf, 10, 0, 11, 20, np.inf, np.inf, np.inf, ],
                         [24, np.inf, 11, 0, 5, 13, 10, np.inf, ],
                         [np.inf, 26, 20, 5, 0, 6, np.inf, 5, ],
                         [np.inf, 22, np.inf, 13, 6, 0, np.inf, np.inf, ],
                         [np.inf, 7, np.inf, 10, np.inf, np.inf, np.inf, np.inf, ],
                         [10, np.inf, np.inf, np.inf, 5, np.inf, np.inf, np.inf, ]])
    # distance = np.random.randint(10, size=(100, 100))
    path = dijkstra(distance, 5)
    for p in sorted(path.values(), key=lambda p: p[1]):
        print(p)
    shortest_weight, shortest_paths = floyd(distance)


    calculated_line = np.array([1, 2, 4, 5, 6]) - 1

    calculated_distance = shortest_weight[calculated_line[:, np.newaxis], calculated_line[np.newaxis, :]]
    line_map = {i: calculated_line[i] for i in range(len(calculated_line))}
    best_path =[0, 1, 2, 3, 4]
    mapped_path = tuple(map(lambda i: line_map[i], best_path))
    real_path = get_real_path(mapped_path, shortest_paths)

    add_one(real_path)
