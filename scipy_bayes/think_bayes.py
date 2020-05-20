import scipy.stats as st
import numpy as np
from matplotlib import pyplot as plt


def mm():
    # 两袋各取一颗，黄，绿， 求黄色来自1994的概率
    # m&m problem

    color_1994 = ('brown', 'yellow', 'red', 'green', 'orange', 'yellow_brown')
    color_1996 = ('blue', 'green', 'orange', 'yellow', 'red', 'brown')
    p_1994 = (.3, .2, .2, .1, .1, .1)
    p_1996 = (.24, .2, .16, .14, .13, .13)

    # 把所有颜色转为对应的数字保存在字典中
    color = set(color_1996).union(set(color_1994))
    color_dict = {i: c for c, i in enumerate(color)}
    color_1996 = [color_dict[c] for c in color_1996]
    color_1994 = [color_dict[c] for c in color_1994]
    mm_1994 = st.rv_discrete(values=(color_1994, p_1994))
    mm_1996 = st.rv_discrete(values=(color_1996, p_1996))

    # s为取到一黄一绿的可能性
    # p(黄=1994|s) = p(s|黄=1994) * p(黄=1994)/p(s)= p(s，黄=1994)/p(s), 这里
    p1 = mm_1994.pmf(color_dict['yellow']) * mm_1996.pmf(color_dict['green'])
    p2 = mm_1996.pmf(color_dict['yellow']) * mm_1994.pmf(color_dict['green'])
    p = p1 / (p1 + p2)
    print(p)


def monty_hall():
    # 选择A门，打开BC门，
    # 打开B门后没车，车在A，B，C的概率
    door = ('A', 'B', 'C')
    p_door = (1 / 3, 1 / 3, 1 / 3)
    door_dict = {i: c for c, i in enumerate(door)}
    door = [door_dict[d] for d in door]

    monty = st.rv_discrete(values=(door, p_door))
    # 车在A门后面，随机打开BC门概率，1 / 2，如果每次优先选B门则，概率为1
    p_A_H = 1 / (len(monty.xk) - 1) * monty.pmf(door_dict['A'])
    p_B_H = 0
    p_C_H = 1 * monty.pmf(door_dict['A'])

    # 通常不求标准化常量P（D）,因为他是固定不变的,直接把所有情况求出来后归一化
    print('A:', p_A_H / (p_A_H + p_B_H + p_C_H))
    print('B:', p_B_H / (p_A_H + p_B_H + p_C_H))
    print('C:', p_C_H / (p_A_H + p_B_H + p_C_H))


# P(H|D) = P(D|H)*P(H)/P(D)
# 通常求所有h，d的P(D|H)*P(H)求和归一化得到P(H|D)，P(D)为标准化常量，不求
class bayes:
    # 输入为np.array
    def __init__(self, P_H_items, P_H_prob):
        # 先验概率P(H)
        self.P_H_items = P_H_items
        # 后验概率P(H|D), 初始D为空
        self.P_H_D_prob = P_H_prob / sum(P_H_prob)

    # 后验概率（条件概率） P(D|H), 每次返回所有h对应d的概率
    def likely_hood(self, d):
        pass

    # 新增时间d
    def updata_h(self, D):
        for d in D:
            self.P_H_D_prob *= self.likely_hood(d)

        # 归一化
        self.P_H_D_prob /= sum(self.P_H_D_prob)

    # cdf 质量分布
    def cdf(self, pct):
        cdf = self.P_H_D_prob.cumsum()
        return self.P_H_items[cdf > pct][0]

    # 使用当前H的分布概率来预测下一次实验，出现D的可能性
    def predict_next(self, D):
        next_distribution = np.array(list(
            map(lambda d: sum(self.P_H_D_prob * self.likely_hood(d)), D)
        ))
        next_distribution /= sum(next_distribution)
        return next_distribution


class train(bayes):
    def __init__(self, P_H_items, P_H_prob):
        bayes.__init__(self, P_H_items, P_H_prob)

    def likely_hood(self, d):
        return np.array(list(
            map(lambda h: 0 if d > h else 1 / h, self.P_H_items)
        ))


class coin(bayes):
    def __init__(self, P_H_items, P_H_prob):
        bayes.__init__(self, P_H_items, P_H_prob)

    def likely_hood(self, d):
        return st.binom.pmf(d, n=250, p=self.P_H_items)


class hockey(bayes):
    def __init__(self, P_H_items, P_H_prob):
        bayes.__init__(self, P_H_items, P_H_prob)

    def likely_hood(self, d):
        return st.poisson.pmf(d, self.P_H_items)


def train_test():
    # 火车问题
    n = 1000
    train_test = train(np.linspace(1, 400, 400), np.ones(400))
    # 注意这里的updata_h, get_P_H_D,虽然是调用的父类，但是用的变量都是子类

    # 看到的火车号码
    train_test.updata_h([60, 112, 3, 24])
    # 90%置信区间

    x = train_test.P_H_items
    y = train_test.P_H_D_prob

    fig1 = plt.figure()
    plt.title(
        "train max number 5%-95% interval ({},{})".format(train_test.cdf(0.05), train_test.cdf(0.95)))
    plt.plot(x, y)
    plt.show()

    # 下一辆车看到号码的可能性，由于最大车号肯定小于112，所以112一下是等分布
    fig2 = plt.figure()
    plt.title('next train number prediction')
    plt.plot(np.linspace(1, 400, 400), train_test.predict_next(np.linspace(1, 400, 400)))
    plt.show()


# 二项分布, 扔200次硬币， 正反各%50，st.binom.pmf(i, n=200, p=0.5)
def euro_coin():
    '''
    p = []
    for i in range(201):
        p.append(st.binom.pmf(i, n=250, p=0.5))
    plt.title("5%-95% interval ({},{})".format(st.binom.ppf(0.05, n=200, p=0.5), st.binom.ppf(0.95, n=200, p=0.5)))
    plt.plot(np.array(list(range(200))), np.array(p))
    plt.show()
    '''
    print("先验二项分布")
    print("均匀分布出现 140, 110可能性:", 2 * st.binom.cdf(110, n=250, p=0.5))
    print("均匀分布 5%-95% 置信区间 ({},{})".format(st.binom.ppf(0.05, n=250, p=0.5), st.binom.ppf(0.95, n=250, p=0.5)))

    coin_test = coin(np.linspace(0, 1, 100), np.ones(100))
    coin_test.updata_h([140])

    print("后验概率")
    # 使用置信区间来判断均匀分布是否可能
    print("均匀百分比的置信分布 5%-95%可能性:({:.2f},{:.2f})".format(coin_test.cdf(0.05), coin_test.cdf(0.95)))
    print("均匀百分比的置信分布 50% 处可能性:{:.2f}".format(coin_test.cdf(0.5)))

    print('使用beta分布')
    print("均匀百分比的置信分布 5%-95%可能性:({0[0]:.2f},{0[1]:.2f})".format(st.beta.ppf([0.05, 0.95], 140, 110)))
    print("均匀百分比的置信分布 50% 处可能性:{:.2f}".format(st.beta.ppf(0.5, 140, 110)))
    print("均匀百分比的置信分布 50% 处可能性:{:.2f}".format(st.beta.ppf(0.5, 140, 110)))


def hockey_test():
    bear = hockey(np.linspace(0, 10, 110), np.ones(110))
    canada = hockey(np.linspace(0, 10, 110), np.ones(110))
    bear.updata_h([0, 1, 8, 3])
    canada.updata_h([3, 1, 2, 0])

    fig1 = plt.figure()
    plt.title('poisson lambda distribution')
    plt.plot(bear.P_H_items, bear.P_H_D_prob, label='bear')
    plt.plot(bear.P_H_items, canada.P_H_D_prob, label='canada')
    plt.show()

    fig2 = plt.figure()
    plt.title('next match score distribution')
    plt.plot(np.linspace(0, 10, 11), bear.predict_next(np.linspace(0, 10, 11)), label='bear')
    plt.plot(np.linspace(0, 10, 11), canada.predict_next(np.linspace(0, 10, 11)), label='canada')
    plt.show()

if __name__ == '__main__':
    # mm()
    # monty_hall()
    # train_test()
    # euro_coin()
    hockey_test()