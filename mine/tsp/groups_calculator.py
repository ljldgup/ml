from functools import reduce
from typing import List

import numpy
import numpy as np
from collections import defaultdict


def gen_groups(importance: numpy.ndarray, dependency: numpy.ndarray, head=None):
    # 固定的头部不依赖其他位置点
    assert type(dependency) != numpy.ndarray or np.all(dependency[:, 1] != head)

    data_size, importance_size = importance.shape

    data_group = np.zeros(shape=data_size)
    # 按顺序，每个对象所属的组
    if type(importance) == np.ndarray:
        importance_cof = np.power(10, np.arange(importance_size)[::-1])
        new_importance = (importance * importance_cof[np.newaxis, np.newaxis, :]).sum(axis=-1).reshape(-1)

        importance_order = new_importance.argsort()[::-1]
        sorted_new_importance = new_importance[importance_order]

        group_end = np.argwhere(sorted_new_importance[1:] != sorted_new_importance[:-1]) + 1
        # 每组在排好序的的起止坐标
        group_index = np.stack([np.append(0, group_end), np.append(group_end, data_size)]).swapaxes(0, 1)

        for i in range(len(group_index)):
            st, ed = group_index[i]
            data_group[importance_order[st:ed]] = i
    # 头部分组
    if head != None:
        data_group[head] = -1

    return adjust_by_dependency(data_group, dependency)


def adjust_by_dependency(data_group: numpy.ndarray, dependency: numpy.ndarray):
    dependency_size, _ = dependency.shape
    # 拓扑排序
    graph = {}
    # 为了避免遍历时出错，不用defaultdict
    for i in np.unique(dependency):
        graph[i] = set()

    for depended, depend in dependency:
        graph[depend].add(depended)

    topology_order = get_topology_order(np.unique(dependency), graph)
    # 过滤出有依赖的
    topology_order = tuple(filter(lambda i: graph[i], topology_order))

    for v in topology_order:
        max_depend_group = max(map(lambda i: data_group[i], graph[v]))
        if max_depend_group + 1 > data_group[v]:
            # 因为重要度的限制，只有两种可能，1.前置同一个组，2.前置后面一个组, 这一点很重要，否则复杂度爆炸
            data_group[v] = max_depend_group + np.random.randint(2)
    return adjust_group_internal_order(data_group, graph)


def adjust_group_internal_order(data_group: np.ndarray, graph: dict):
    dependency_groups = {int(i): set() for i in np.unique(data_group)}
    # 领接链表都放进去
    for k, v in graph.items():
        dependency_groups[int(data_group[k])].add(k)
        for i in v:
            dependency_groups[int(data_group[i])].add(i)

    groups_dict = defaultdict(list)
    for i in range(len(data_group)):
        if i not in dependency_groups[data_group[i]]:
            groups_dict[data_group[i]].append(i)

    # 同一组内的依赖相关项按照拓扑排序随机插入
    for i in dependency_groups:
        dependency_list = dependency_groups[i]
        independent_list = groups_dict[i]
        old_in_len = len(independent_list)
        topology_order = get_topology_order(dependency_list, graph)
        insert_position = np.random.choice(np.arange(len(independent_list) + len(dependency_list)),
                                           size=len(dependency_list), replace=False)
        insert_position.sort()
        for i in range(len(dependency_list)):
            independent_list.insert(insert_position[i], topology_order[i])
        assert len(independent_list) == old_in_len + len(dependency_list)

    sorted_group_list = tuple(map(lambda i: groups_dict[i], sorted(groups_dict.keys())))
    assert sum(map(len, sorted_group_list), 0) == len(data_group)
    return sorted_group_list


def get_topology_order(points: set, graph: dict):
    searched = set()
    topology_order = []

    def dfs(g: dict):
        random_start = np.random.permutation(tuple(filter(lambda i: i not in searched, g.keys())))
        # 这里是惰性的，filter 不是一次性生效, searched生效会有影响
        for v in random_start:
            dfs_visit(g, v)

    def dfs_visit(g: dict, v: int):
        if v in searched:
            return
        searched.add(v)
        for i in filter(lambda j: j not in searched, g[v]):
            dfs_visit(g, i)
        if v in points:
            topology_order.append(v)

    dfs(graph)

    return topology_order


if __name__ == '__main__':
    data_size = 10
    importance_size = 2

    importance = np.zeros(shape=[data_size, importance_size])
    data_position = np.random.choice(np.arange(data_size), size=(data_size // 3, importance_size), replace=False)
    importance_value = np.random.choice(np.arange(10), size=(data_size // 3, importance_size))
    importance[data_position, np.arange(importance_size)[np.newaxis, :]] = importance_value

    dependency_size = 4
    dependency = np.stack([np.random.choice(np.arange(data_size), size=(2), replace=False)
                           for i in range(dependency_size)])

    group_list = gen_groups(importance, dependency)
