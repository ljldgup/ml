from matplotlib import pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

def muti_scatter(x_cols, y_col, data):
    for j in range((len(x_cols) - 1) // 12 + 1):
        fig, axes = plt.subplots(3, 4)
        for i in range(12):
            if j * 12 + i == len(x_cols):
                break
            # 数据中有nan的时候scatter会出错
            axes[i // 4][i % 4].scatter(data[x_cols[j * 12 + i]][~data[x_cols[j * 12 + i]].isna()], data[y_col][~data[x_cols[j * 12 + i]].isna()], s=5)
            axes[i // 4][i % 4].set_title(x_cols[j * 12 + i], fontsize=8, color='b')
            # x轴不显示
            axes[i // 4][i % 4].xaxis.set_ticks([])
            axes[i // 4][i % 4].yaxis.set_ticks([])

def regressors_test(X, y):
    regressors = [LinearSVR(C=1), SVR(kernel="rbf", C=1),
                  KNeighborsRegressor(n_neighbors=6),
                  RandomForestRegressor(),
                  GradientBoostingRegressor()]

    for regressor in regressors:
        print(regressor.__class__)
        regressor.fit(X, y)
        y_pred = regressor.predict(X)
        ans = np.c_[y, y_pred, y_pred - y]
        print(mean_squared_error(ans[:, 0], ans[:, 1]))
        print(cross_val_score(regressor, X, y, cv=3, scoring="neg_mean_squared_error"))
    return regressors


def classifier_test(X, y):
    classifiers = [SGDClassifier(random_state=42), LogisticRegression(), LinearSVC(C=1), SVC(kernel="rbf", C=1),
                   KNeighborsClassifier(n_neighbors=6),
                   RandomForestClassifier(),
                   # 这里调整最大深度后，精度会提高，但交叉验证变差了
                   GradientBoostingClassifier()]

    for classifier in classifiers:
        print(classifier)
        classifier.fit(X, y)
        print(classifier.score(X, y))
        print(cross_val_score(classifier, X, y, cv=2, scoring="accuracy"))
        print('\n--------------------------------------------------------------------------\n\n')
    return classifiers