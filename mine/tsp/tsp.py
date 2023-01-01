import numpy as np
from matplotlib import pyplot as plt
from itertools import permutations
from functools import reduce
from sklearn.preprocessing import minmax_scale

# 穷举
from data_generator import gen_points, gen_dependency, gen_importance, gen_head_position, get_test_data
from groups_calculator import gen_groups
from shortest_path import floyd
from plot_tools import plot_points, plot_path, plot_direct_path, new_plot


# 可以用于暴力求解，或者对遗传算法的解集求最优解
def exhaustive(populations):
    # 这里如果赋予随机值，可能导致一个不满足要求的解
    best_population = None
    best_distance = None

    if data_size > 10:
        return best_population, best_distance
    for population in populations:
        population = np.array(population)
        cur_distance = distance[population[..., :-1], population[..., 1:]].sum()
        if not dependency_satisfied(population):
            continue
        if dependency_satisfied(population) and head_satisfied(population):
            if best_population is None \
                    or compare_importance(population, best_population) > 0 \
                    or (compare_importance(population, best_population) == 0 and cur_distance < best_distance):
                print("best", population, cur_distance)
                best_population = population
                best_distance = cur_distance
    return best_population, best_distance


def head_satisfied(population: np.ndarray):
    if head_position == None:
        return True

    return population[0] == head_position


def dependency_satisfied(population: np.ndarray):
    if type(dependency) != np.ndarray:
        return True

    order = np.ones_like(population)
    order[population] = np.arange(data_size)
    return np.all(order[dependency[:, 0]] - order[dependency[:, 1]] <= 0)


def compare_importance(population1: np.ndarray, population2: np.ndarray):
    importance1 = new_importance[population1].tolist()
    importance2 = new_importance[population2].tolist()
    # python 原生的list会依次比较
    if importance1 == importance2:
        return 0
    elif importance1 > importance2:
        return 1
    else:
        return -1


def get_distance_matrix():
    data_size, _ = points.shape
    # 将坐标重复data_size次， 变成data_size*data_size*2矩阵，第1个维度代表第几个点
    points_repeated = points.repeat(data_size, axis=0).reshape(data_size, data_size, 2)
    # 这里没法直接转置，因为最后一个维度放的是坐标不操作, 转置后第2个维度代表第几个点
    points_repeated_trans = points_repeated.swapaxes(0, 1)
    # 第1 2个维度分别确定了两段的点，进而计算出距离
    delta = points_repeated - points_repeated_trans
    delta_x, delta_y = delta[..., 0], delta[..., 1]
    # distance[i,j]为点i,j距离
    return np.sqrt(np.square(delta_x) + np.square(delta_y))


def get_order_matrix(populations: np.ndarray):
    population_size, data_size = populations.shape
    orders = np.empty_like(populations)
    lines = np.ones_like(populations).cumsum(axis=0) - 1
    origin_order_matrix = np.array([np.arange(data_size) for i in range(len(populations))])
    orders[lines, populations] = origin_order_matrix
    return orders


def get_importance_cost(populations: np.ndarray):
    population_size, data_size = populations.shape
    # 利用了广播
    populations_importance = new_importance[populations]
    data_index = np.broadcast_to(np.arange(data_size), (data_size, data_size))
    importance_delta = populations_importance[:, data_index] - populations_importance[:, data_index.T]

    # 由于是差值左下和右下相反，将右下角设为0
    left_down_index = np.argwhere(data_index < data_index.T)
    importance_delta[:, left_down_index[:, 0], left_down_index[:, 1]] = 0
    importance_delta[importance_delta < 0] = 0
    return importance_delta.sum(axis=(1, 2))


def get_dependency_cost(populations: np.ndarray, orders: np.ndarray):
    if type(dependency) == np.ndarray:
        # 依赖关系判断
        depended_order = orders[:, depended]
        depend_order = orders[:, depend]
        # 直接将布尔值作为整形来用，如果依赖者顺序小于被依赖者，带来1的成本
        return (depend_order < depended_order).sum(axis=1)
    else:
        return np.zeros(shape=len(populations))


def get_path_cost(populations: np.ndarray, orders: np.ndarray):
    path_cost = distance[populations[..., :-1], populations[..., 1:]]
    return path_cost.sum(axis=1)


def get_cost(populations: np.ndarray, orders: np.ndarray):
    path_cost = get_path_cost(populations, orders)
    dependency_cost = get_dependency_cost(populations, orders)
    importance_cost = get_importance_cost(populations)
    return path_cost, dependency_cost, importance_cost


def validate(populations: np.ndarray, orders: np.ndarray):
    population_size, data_size = populations.shape
    if CHECK_ORDER:
        assert np.all(populations[np.arange(population_size)[:, np.newaxis], orders] == np.arange(data_size))
    if head_position != None:
        assert np.all(populations[:, 0] == head_position)


def choose_survivor(populations: np.ndarray, orders: np.ndarray, cost: np.ndarray, dependency_cost: np.ndarray):
    population_size, data_size = populations.shape
    # 丢弃前置不满足部分
    populations = populations[dependency_cost == 0]
    cost = cost[dependency_cost == 0]
    orders = orders[dependency_cost == 0]

    if len(populations) == 0:
        populations = init_population()
        orders = get_order_matrix(populations)
        return populations, orders

    rest_population_size = len(populations)
    # 加0.1避免到达最优值以后全为0，出现nan
    adjusted_cost = (1.0 * cost.max() - cost + 0.001) / (cost.max() - cost.min() + 0.001) * 4
    adjusted_cost = np.exp(adjusted_cost)
    probability = (adjusted_cost / adjusted_cost.sum())
    chosen_index = np.random.choice(np.arange(rest_population_size), population_size, p=probability.tolist())
    return populations[chosen_index], orders[chosen_index]


def cross_survivors(populations: np.ndarray, orders: np.ndarray):
    population_size, data_size = populations.shape
    # replace=False无重复抽样
    chosen_index = np.random.choice(np.arange(population_size), size=int(population_size * cross_prob), replace=False)
    cross_num = len(chosen_index) // 2
    p1_line_num = chosen_index[:cross_num]
    p2_line_num = chosen_index[cross_num:cross_num * 2]

    length_choices = np.arange(1, data_size - 1)
    prob = get_length_prob(length_choices)
    cur_cross_length = np.random.choice(length_choices, p=prob)
    cross_start_position = np.random.randint(data_size - cur_cross_length, size=cross_num)

    for i in range(cur_cross_length):
        swap_single_gene(cross_start_position + i, populations, orders, p1_line_num, p2_line_num)


def swap_single_gene(cross_position, populations, orders, p1_line_num, p2_line_num):
    p1_gene = populations[p1_line_num, cross_position]
    p2_gene = populations[p2_line_num, cross_position]
    # 交换位置
    p1_swap_position = orders[p1_line_num, p2_gene]
    p2_swap_position = orders[p2_line_num, p1_gene]
    # 这里赋值不能用多次[]得到的结果赋值，ndarray做索引得到的结果，再次选择赋值，对原始对象貌似不能生效
    # 交换基因
    populations[p1_line_num, cross_position], populations[p1_line_num, p1_swap_position] = \
        populations[p1_line_num, p1_swap_position], populations[p1_line_num, cross_position]
    populations[p2_line_num, cross_position], populations[p2_line_num, p2_swap_position] = \
        populations[p2_line_num, p2_swap_position], populations[p2_line_num, cross_position]
    # 调整顺序矩阵
    # 这里如果p1_gene 和p2_gene存在重叠基因，赋值将造成问题，原因在于两个赋值变量指向位置，最终造成重复
    orders[p1_line_num, p1_gene], orders[p1_line_num, p2_gene] = orders[p1_line_num, p2_gene], orders[
        p1_line_num, p1_gene]
    orders[p2_line_num, p2_gene], orders[p2_line_num, p1_gene] = orders[p2_line_num, p1_gene], orders[
        p2_line_num, p2_gene]


def mutate(populations: np.ndarray, orders: np.ndarray, mutate_prob: float):
    chosen_index = np.random.choice(np.arange(population_size), size=int(population_size * mutate_prob), replace=False)
    interval = len(chosen_index) // len(mutate_func)
    for i in range(len(mutate_func)):
        mutate_func[i](populations, orders, chosen_index[i * interval:(i + 1) * interval])


def reverse_mutate(populations: np.ndarray, orders: np.ndarray, chosen_index: np.ndarray):
    population_size, data_size = populations.shape
    if data_size == 1:
        return

    mutate_num = len(chosen_index)
    chosen_index = chosen_index[..., np.newaxis]
    length_choices = np.arange(2, data_size + 1)
    prob = get_length_prob(length_choices)
    mutate_length = np.random.choice(length_choices, p=prob)

    mutate_start_position = np.random.randint(0, data_size - mutate_length + 1, size=mutate_num)
    # 广播
    mutate_position = mutate_start_position[:, np.newaxis] + np.arange(mutate_length)

    # 逆转顺序，理论上后逆转也可以，因为基因的位置范围没变
    mutate_gen = populations[chosen_index, mutate_position]
    orders[chosen_index, mutate_gen] = orders[chosen_index, mutate_gen[:, ::-1]]

    # 逆转基因
    populations[chosen_index, mutate_position] = populations[chosen_index, mutate_position[:, ::-1]]


def swap_mutate(populations: np.ndarray, orders: np.ndarray, chosen_index: np.ndarray):
    population_size, data_size = populations.shape
    mutate_num = len(chosen_index)
    chosen_index = chosen_index[:, np.newaxis]
    swap_left_position = np.random.choice(np.arange(data_size - 1), size=(mutate_num, 1))
    swap_position = np.concatenate([swap_left_position, swap_left_position + 1], axis=1)
    # 逆转顺序，理论上后逆转也可以，因为基因的位置范围没变
    mutate_gen = populations[chosen_index, swap_position]
    orders[chosen_index, mutate_gen] = orders[chosen_index, mutate_gen[:, ::-1]]

    populations[chosen_index, swap_position] = populations[chosen_index, swap_position[:, ::-1]]


def permutation_mutate(populations: np.ndarray, orders: np.ndarray, chosen_index: np.ndarray):
    population_size, data_size = populations.shape
    if data_size == 1:
        return

    mutate_num = len(chosen_index)
    chosen_index = chosen_index[..., np.newaxis]
    mutate_length = np.random.choice(np.arange(2, data_size + 1))
    mutate_range = np.arange(mutate_length)
    mutate_start_position = np.random.randint(0, data_size - mutate_length + 1, size=mutate_num)

    mutate_position = mutate_start_position[:, np.newaxis] + mutate_range
    permutation_order = np.array([np.random.permutation(mutate_range) for i in range(mutate_num)])

    mutate_gen = populations[chosen_index, mutate_position]
    orders[chosen_index, mutate_gen[np.arange(mutate_num)[:, np.newaxis], permutation_order]] = orders[
        chosen_index, mutate_gen]
    populations[chosen_index, mutate_position] = populations[
        chosen_index, mutate_position[np.arange(mutate_num)[:, np.newaxis], permutation_order]]


def init_population():
    # 直接使用分组产生种群
    return np.array(tuple(map(lambda groups: sum(groups, []),
                              [gen_groups(importance, dependency, head_position) for _ in range(population_size)])))
    # return np.array([reduce(np.append, map(np.random.permutation, groups_list)) for i in range(population_size)])


# 长度越短，稳定性越高，给与更高的概率
def get_length_prob(choices):
    prob = 1 / np.square(choices)
    return prob / prob.sum()


# 遗传算法
def ga(iter_times=800):
    populations = init_population()
    orders = get_order_matrix(populations)
    validate(populations, orders)

    score_history = []
    population_history = []

    plot_interval = 1 + iter_times // 10

    for i in range(iter_times):
        path_cost, dependency_cost, importance_cost = get_cost(populations, orders)
        cost = 10 * minmax_scale(importance_cost) + minmax_scale(path_cost)
        populations, orders = choose_survivor(populations, orders, cost, dependency_cost)

        # 保存记录
        if divmod(i, plot_interval)[1] == 0:
            path_cost, dependency_cost, importance_cost = get_cost(populations, orders)
            cost = 10 * minmax_scale(importance_cost) + minmax_scale(
                path_cost - path_cost.min())
            # zh
            min_index = cost.argmin()
            score_history.append(
                [i, cost.min(), path_cost[min_index], dependency_cost[min_index], importance_cost[min_index]])
            population_history.append(populations[cost.argmin()])
            print("ga", populations[cost.argmin()])
            validate(populations, orders)

        cross_survivors(populations, orders)
        for st, ed in group_index.swapaxes(0, 1):
            if ed - st > 1:
                mutate(populations[:, st:ed], orders, mutate_prob)

    return populations, np.array(score_history), np.array(population_history)


if __name__ == '__main__':

    CHECK_ORDER = True
    data_size = 9
    population_size = 500
    cross_prob = 0.4
    mutate_prob = 0.1

    importance_size = 2
    importance_fill_num = 2
    dependency_size = 5
    mutate_func = [reverse_mutate, swap_mutate]

    head_position = gen_head_position(data_size)
    points = gen_points(3, data_size, with_noise=True)
    dependency = gen_dependency(data_size, dependency_size)
    importance = gen_importance(data_size, importance_size, importance_fill_num)
    distance = get_distance_matrix()
    # head_position = None
    #
    # points, distance, dependency, importance, head_position = get_test_data()
    # data_size = len(points)

    paths = None
    if np.any(distance == np.inf):
        old_distance = distance
        distance, paths = floyd(distance)

    if type(dependency) == np.ndarray:
        if head_position != None:
            dependency = dependency[dependency[:, 1] != head_position]
        depended = dependency[:, 0]
        depend = dependency[:, 1]
    importance_cof = np.power(10, np.arange(importance_size)[::-1])
    new_importance = (importance * importance_cof[np.newaxis, np.newaxis, :]).sum(axis=-1).reshape(-1)

    # 即使不在组内变异，分组初始化也能对收敛有好处
    # groups_list = gen_groups(importance, dependency, head_position)
    # 不分组
    groups_list = [[head_position], tuple(filter(lambda i: i != head_position, range(data_size)))]
    group_end_index = np.cumsum(tuple(map(len, groups_list)))
    group_index = np.stack([np.append(0, group_end_index[:-1]), group_end_index])
    # 归一化，数据过大，可能会有损性能
    # points = points / points.max()

    plt.plot(points[:, 0], points[:, 1])

    final_populations, final_score_history, final_population_history = ga(iter_times=400)

    for idx in range(0, len(final_population_history), len(final_population_history) // 10):
        g = final_population_history[idx]
    plot_direct_path(g, points)
    # plt.title(g)

    new_plot(final_score_history[:, 0], final_score_history[:, 2])

    # 获得ga算法的最优解
    ga_best_population, ga_best_distance = exhaustive(final_population_history)

    ga_path_cost, ga_dependency_cost, ga_importance_cost = get_cost(ga_best_population.reshape(1, -1),
                                                                    get_order_matrix(ga_best_population.reshape(1, -1)))

    # 穷举以及一些场景验证
    # 如果头部没有满足说明没找到满足条件的解！！！
    best_population, best_distance = exhaustive(permutations(tuple(range(data_size))))
    path_cost, dependency_cost, importance_cost = get_cost(best_population.reshape(1, -1),
                                                           get_order_matrix(best_population.reshape(1, -1)))

    np.concatenate([np.arange(data_size)[:, np.newaxis], importance], axis=1)
