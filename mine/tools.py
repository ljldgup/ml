from matplotlib import pyplot as plt


def muti_scatter(x_cols, y_col, data):
    for j in range((len(x_cols) - 1) // 12 + 1):
        fig, axes = plt.subplots(3, 4)
        for i in range(12):
            if j * 12 + i == len(x_cols):
                break
            axes[i // 4][i % 4].scatter(data[x_cols[j * 12 + i]], data[y_col], s=5)
            axes[i // 4][i % 4].set_title(x_cols[j * 12 + i], fontsize=8, color='b')
            # x轴不显示
            axes[i // 4][i % 4].xaxis.set_ticks([])
            axes[i // 4][i % 4].yaxis.set_ticks([])
