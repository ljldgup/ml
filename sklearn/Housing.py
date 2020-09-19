import os
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from scipy.stats import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


# 自定义转换器Transformer
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):  # no *args or**kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


HOUSING_PATH = '.'


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# 文件较大放在百度网盘
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def test(housing):
    # 查看数据的基本情况，可以结合可视化
    housing.info()
    housing.describe()
    # 数据呈现长尾分布
    # housing.hist(bins=50, figsize=(20, 15))
    train_set, test_set = train_test_split(housing, test_size=0.2,
                                           random_state=42)

    # 收入转化为标签
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    # housing["income_cat"].hist()

    # 分割数据
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing,
                                               housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()

    '''
    # 注意这里绘图时通过x,y指定对应列
    # housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    # 半径s人口，颜色c房价
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"] / 100, label="population", figsize=(10, 7),
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
                 )
    plt.legend()
    '''

    # 相关性
    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)

    attributes = ["median_house_value", "median_income", "total_rooms",
                  "housing_median_age"]
    # scatter_matrix(housing[attributes], figsize=(12, 8))
    # housing.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    # 数据清洗
    # NA 插入数据
    housing.dropna(subset=["total_bedrooms"])  # option 1
    housing.drop("total_bedrooms", axis=1)  # option 2
    median = housing["total_bedrooms"].median()  # option 3
    housing["total_bedrooms"].fillna(median, inplace=True)

    # sklearn提供的填充空值方法
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    imputer.statistics_
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                              index=housing_num.index)

    # 编码，OrdinalEncoder相当于LabelEncoder的矩阵版，能对多个特征同时编码
    housing_cat = housing[["ocean_proximity"]]
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)


    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing.values)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    housing_num_tr = num_pipeline.fit_transform(housing_num)

    # ColumnTransformer对pandas每列操作
    # 注意这里list(housing_num) 返回的是housing_num的列名
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    # 注意这里传入了dataframe的列名（属性名），使Transformer进行相应的转换
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    # 这里返回的onehot+连续值使用的是非稀疏举证, 将每个类编都转换成了0/1两种值的列，
    # 可以分开转换然后用sparse.hstack组装起来，titanic有用过
    housing_prepared = full_pipeline.fit_transform(housing)

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    print("Predictions:", lin_reg.predict(some_data_prepared))
    print("Labels:", list(some_labels))

    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)


    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    # 这里的损失是0，因为没有测试集，决策树严重的过拟合了
    print(tree_rmse)

    # 采用K折验证，决策树的表现不如之前,cv分成几份
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    print(tree_rmse_scores)

    # cross_val_score的评估标准是与正常loss相反
    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                                 scoring="neg_mean_squared_error", cv=10)

    lin_rmse_scores = np.sqrt(-lin_scores)
    display_scores(lin_rmse_scores)

    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing_prepared, housing_labels)
    forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                    scoring="neg_mean_squared_error", cv=10)

    forest_rmse_scores = np.sqrt(-forest_scores)
    display_scores(forest_rmse_scores)

    # 超参数的自动搜索
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)

    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    final_model = grid_search.best_estimator_
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(final_rmse)

    # 95置信区间
    confidence = 0.95
    squared_errors = (final_predictions - y_test) ** 2
    np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc=squared_errors.mean(),
                             scale=stats.sem(squared_errors)))


if __name__ == '__main__':
    housing = load_housing_data()
    corr = housing.corr().sort_values(by='median_house_value', ascending=False).round(2)
    plt.imshow(corr, interpolation='nearest', cmap=plt.cm.hot)