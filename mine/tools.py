from matplotlib import pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

import xgboost as xgb
import lightgbm as lgb

import seaborn as sns

def muti_scatter(x_cols, y_col, data, type):
    for j in range((len(x_cols) - 1) // 12 + 1):
        fig, axes = plt.subplots(3, 4)
        for i in range(12):
            if j * 12 + i == len(x_cols):
                break
            if type == 's':
                # 数据中有nan的时候scatter会出错
                axes[i // 4][i % 4].scatter(data[x_cols[j * 12 + i]][~data[x_cols[j * 12 + i]].isna()],
                                            data[y_col][~data[x_cols[j * 12 + i]].isna()], s=5)

            elif type == 'b':
                if len(data[x_cols[j * 12 + i]].unique()) < 30:
                    sns.boxplot(data[x_cols[j * 12 + i]],data[y_col],ax = axes[i // 4][i % 4])

            axes[i // 4][i % 4].set_title(x_cols[j * 12 + i], fontsize=8, color='b')
            # x轴不显示
            axes[i // 4][i % 4].xaxis.set_ticks([])
            axes[i // 4][i % 4].yaxis.set_ticks([])


def rmsle_cv(model, x, y):
    n_folds = 5
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x)
    # cross_val_score 等用的是拷贝的model，原model不会有影响
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)


def regressors_test(X, y):
    regressors = [  make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1)),
        make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)),
        LinearSVR(),
        KNeighborsRegressor(n_neighbors=8),
        RandomForestRegressor(),
        # 这个huber loss选择对结果有很大影响
        GradientBoostingRegressor(learning_rate=0.05,
                                  n_estimators=900, max_depth=4, max_features=14, max_leaf_nodes=10,
                                  loss='huber', min_samples_split=150),

    ]
    '''
        #网络上已经调好参数的xgboost,LGBMRegressor
      GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                 max_depth=4, max_features='sqrt',
                                 min_samples_leaf=15, min_samples_split=10,
                                 loss='huber', random_state=5)
                                                     
      xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                       learning_rate=0.05, max_depth=3,
                       min_child_weight=1.7817, n_estimators=2200,
                       reg_alpha=0.4640, reg_lambda=0.8571,
                       subsample=0.5213, silent=1,
                       random_state=7, nthread=-1),
      lgb.LGBMRegressor(objective='regression', num_leaves=5,
                        learning_rate=0.05, n_estimators=720,
                        max_bin=55, bagging_fraction=0.8,
                        bagging_freq=5, feature_fraction=0.2319,
                        feature_fraction_seed=9, bagging_seed=9,
                        min_data_in_leaf=6, min_sum_hessian_in_leaf=1)
    '''

    all_scores = []
    for regressor in regressors:
        print(regressor.__class__)
        score = rmsle_cv(regressor, X, y)
        all_scores.append(score.mean())
        print(score)
        regressor.fit(X, y)
        y_pred = regressor.predict(X)
        ans = np.c_[y, y_pred, y_pred - y]
        print(np.sqrt(mean_absolute_error(ans[:, 0], ans[:, 1])))
    all_scores = np.array(all_scores)
    print("所有分类器得分均值:", all_scores.mean())
    print("每个得分类器得分均值:", all_scores)

    return regressors


def classifier_test(X, y):
    classifiers = [SGDClassifier(random_state=42), LogisticRegression(), LinearSVC(C=1), SVC(kernel="rbf", C=1),
                   KNeighborsClassifier(n_neighbors=6),
                   RandomForestClassifier(),
                   # 这里调整最大深度后,精度会提高,但交叉验证变差了
                   GradientBoostingClassifier()]
    all_scores = []
    for classifier in classifiers:
        print(classifier)
        score = cross_val_score(classifier, X, y, cv=3, scoring="accuracy")
        print(score)
        classifier.fit(X, y)
        print(classifier.score(X, y))
        all_scores.append(score.mean())
        print('\n--------------------------------------------------------------------------\n\n')
    all_scores = np.array(all_scores)
    print(all_scores)
    print("所有分类器得分 均值:", all_scores.mean())
    return classifiers


def custom_kfold(model, x, y, score_func):
    print(model, ' k折验证')
    model_ = clone(model)
    kfold = KFold(n_splits=3, shuffle=True, random_state=24)
    ans = []
    for train_idx, test_idx in kfold.split(x, y):
        model_.fit(x[train_idx], y[train_idx])
        # evaluate the model
        scores = score_func(model_.predict(x[test_idx]), y[test_idx])
        ans.append(scores)
    print(ans)
    return ans


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # 这里内部用了kfold，外部用交叉验证会出错，原因不明。。
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # out_of_fold_predictions 保留了每个模型对训练集的预测值,作为第二层模型的训练值
        # 这里没有测试集
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # 注意返回值才有meta_model_
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)
