import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb
import time

st_time = time.clock()
"""
目前使用3个月预测目前来看效果最好
使用了预测时间段leakage的特征后，效果提升了约20%。。。 非常夸张。。
online的数据对于预测用处不大，提取少量重要特征即可
"""
# used_interval consume_interval是用来统计的，7月份并不能取得这个数据
drop_columns = ['Unnamed: 0', 'Date', 'Discount_rate', 'Merchant_id', 'used', 'received_date', 'date',
                'used_days', 'filter_date', 'last_consume_date', 'last_use_date', 'last_received_date',
                'coupon_last_received_date', 'value_rank', 'rate_rank']
# input_feature_0 = pd.read_csv('2m_samples_2016-03-01~2016-04-01.csv').drop(columns=drop_columns)
input_feature_1 = pd.read_csv('3m_samples_2016-04-01~2016-05-01.csv').drop(columns=drop_columns)
input_feature_2 = pd.read_csv('3m_samples_2016-05-01~2016-06-01.csv').drop(columns=drop_columns)
input_feature_3 = pd.read_csv('3m_samples_2016-06-01~2016-07-01.csv').drop(columns=drop_columns)

# 补全为-1 比0效果好
train_input = pd.concat([input_feature_1, input_feature_2, input_feature_3], ignore_index=True)
train_input = train_input.replace([np.inf, -np.inf], np.nan).fillna(-1)

predict_samples = pd.read_csv('3m_samples_2016-07-01~2016-08-01.csv').drop(columns=drop_columns)
predict_samples = predict_samples.replace([np.inf, -np.inf], np.nan).fillna(-1)

'''
# 能预测概率分数较高的是GBT
classifiers = [LogisticRegression(), RandomForestClassifier(),
               GradientBoostingClassifier()]
all_scores = []
for classifier in classifiers:
    print(classifier)
    classifier.fit(train_input.drop(columns=['target', 'Coupon_id', 'User_id','Date_received']).values, train_input['target'].values)
    all_scores.append(roc_auc_score(input_feature_3['target'].values,
                                    classifier.predict(input_feature_3.drop(columns=['target', 'Coupon_id']).values)))
print(all_scores)
'''

'''
# 基准分数
roc_auc_score(input_feature_3['target'].values,
              np.random.randn(*input_feature_3['target'].values.shape))

input_feature_3['pred'] = classifier.predict_proba(
    input_feature_3.drop(columns=['target', 'Coupon_id', 'User_id','Date_received']).values)[:, 1]

roc_auc_score(input_feature_3['target'].values, input_feature_3['pred'].values)
'''
'''
t = input_feature_2.sample(50000)
classifier = GradientBoostingClassifier(n_estimators=200, max_depth=4)
sc1 = cross_val_score(classifier,
                      t.drop(columns=['target', 'Coupon_id', 'User_id', 'Date_received']).values,
                      t['target'].values, cv=3, scoring="roc_auc")

t = input_feature_3.sample(50000)
sc2 = cross_val_score(classifier,
                      input_feature_3.drop(columns=['target', 'Coupon_id', 'User_id', 'Date_received']).values,
                      input_feature_3['target'].values, cv=3, scoring="roc_auc")
'''
'''
t = train_input
# param_test1 = {'learning_rate': [0.1, 0.14, 0.18, 0.22, 0.26, 0.30]}  # 0.2
param_test1 = {'n_estimators': range(300, 901, 150)}  # 400

# param_test1 = {'max_depth': range(4, 7, 1)}  # 'max_depth': 5, 'min_samples_split': 600
# param_test1 = {'min_samples_leaf': range(100, 600, 100)}
# param_test1 = {'max_leaf_nodes': [4, 8, 12, 16], 'max_features': [8, 16, 24, 32]} # {'max_features': 8, 'max_leaf_nodes': 12}
# param_test1 = {'min_samples_leaf': range(500, 800, 100), 'min_samples_split': [600, 9000, 1200, 1500]}
# 这个参数有明显提升{'min_samples_leaf': 500, 'min_samples_split': 1500} 0.84830209385559


gsearch1 = GridSearchCV(
    estimator=GradientBoostingClassifier(learning_rate=0.2),
    param_grid=param_test1, scoring='roc_auc', iid=False, cv=3, verbose=2, n_jobs=3)
gsearch1.fit(t.drop(columns=['target', 'Coupon_id', 'User_id', 'Date_received']).values, t['target'].values)
print(gsearch1.best_params_, gsearch1.best_score_)




# 尝试删除 重要度为0的特征效果反而变差了
classifier = joblib.load('3m_gdbt.model')
columns = predict_samples.drop(columns=['target', 'Coupon_id', 'User_id', 'Date_received']).columns
columns[classifier.feature_importances_.argsort()[::-1]]
drop_columns = columns[classifier.feature_importances_ == 0].to_list() + ['target', 'Coupon_id', 'User_id',
                                                                'Date_received']

'''

drop_t_columns = ['target', 'Coupon_id', 'User_id', 'Date_received']

classifier = GradientBoostingClassifier(learning_rate=0.12, n_estimators=400, max_depth=5,
                                        min_samples_split=1500, max_features=10, min_samples_leaf=500,
                                        max_leaf_nodes=12)
'''

classifier = model = xgb.XGBClassifier(max_depth=5, learning_rate=0.12, n_estimators=400,
                                       min_child_weight=200, max_delta_step=0, subsample=0.8,
                                       colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.2,
                                       scale_pos_weight=10, eval_metric='auc', seed=1440, nthread=3)
'''
classifier.fit(train_input.drop(columns=drop_t_columns).values, train_input['target'].values)
predict_samples['Probability'] = classifier.predict_proba(
    predict_samples.drop(columns=drop_t_columns).values)[:, 1]

predict_samples[['User_id', 'Coupon_id', 'Date_received']] = predict_samples[
    ['User_id', 'Coupon_id', 'Date_received']].astype(int)
predict_samples[['User_id', 'Coupon_id', 'Date_received', 'Probability']].to_csv('submission.csv', index=None)
ans = classifier.predict_proba(train_input.drop(columns=drop_t_columns).values)
print(roc_auc_score(train_input['target'].values, ans[:, 1]))

columns = predict_samples.drop(columns=['target', 'Coupon_id', 'User_id', 'Date_received']).columns
print(columns[classifier.feature_importances_.argsort()[::-1]][:40])

joblib.dump(classifier, '3m_gdbt.model')

'''
# 模型融合，略微提高，0.6,0.3,0.1效果最佳
classifiers = [joblib.load('3m_gdbt.model'), joblib.load('4m_gdbt.model')]
predict_samples = [
    pd.read_csv('3m_samples_2016-07-01~2016-08-01.csv').drop(columns=drop_columns).replace([np.inf, -np.inf], np.nan).fillna(-1),
    pd.read_csv('4m_samples_2016-07-01~2016-08-01.csv').drop(columns=drop_columns).replace([np.inf, -np.inf], np.nan).fillna(-1),
]
for i in range(len(classifiers)):
    predict_samples[i]['Probability'] = classifiers[i].predict_proba(
        predict_samples[i].drop(columns=['target', 'Coupon_id', 'User_id', 'Date_received']).values)[:, 1]

t = pd.read_csv('3m_samples_2016-07-01~2016-08-01.csv').drop(columns=drop_columns)
t['Probability'] = 0.65 * predict_samples[0]['Probability'] + 0.35 * predict_samples[1][
    'Probability'] # + 0.1 * predict_samples[2]['Probability']
t[['User_id', 'Coupon_id', 'Date_received']] = predict_samples[0][
    ['User_id', 'Coupon_id', 'Date_received']].astype(int)
t[['User_id', 'Coupon_id', 'Date_received', 'Probability']].to_csv('submission.csv', index=None)
'''
print(time.clock() - st_time)
