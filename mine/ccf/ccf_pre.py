import pandas as pd
import numpy as np
import seaborn as sns
from pandas.tseries.offsets import Day

from mine.tools import classifier_test

'''
用户之间无关，但商户和优惠券时有关的
用户的动作频率要考虑
'''
'''
# 用户O2O线下优惠券使用预测样本
offline_test_revised = pd.read_csv('ccf_offline_stage1_test_revised.csv')
# 用户线下消费和优惠券领取行为
offline = pd.read_csv('ccf_offline_stage1_train.csv')
#  用户线上点击/消费和优惠券领取行为
online = pd.read_csv('ccf_online_stage1_train.csv')
# 选手提交文件字段
'''
# 测试用少量数据就够了
offline_ori = pd.read_csv('ccf_offline_stage1_train.csv', nrows=10000)
online_ori = pd.read_csv('ccf_online_stage1_train.csv', nrows=10000)
offline_pre_ori = pd.read_csv('ccf_offline_stage1_test_revised.csv', nrows=10000)
offline_ori = pd.concat([offline_ori, offline_pre_ori], ignore_index=True)
'''
# 线下数据分成两块以防数据泄露影响交叉试验
# 数据时按照用户id排的, 一个用户的一组行为被排在一起
online_ori.head(20)
online_ori.shape

# 平均每个用户大约有十几个行为。
online_ori['User_id'].unique().shape
online_ori['Merchant_id'].unique().shape
# 大部分用户行为集中在20次以下
# online_ori.groupby(['User_id']).count()['Merchant_id'].hist(bins=20)

offline_ori.shape
offline_ori['User_id'].unique().shape
offline_ori['Merchant_id'].unique().shape


# 大部分用户行为集中在10次以下
# offline_ori.groupby(['User_id']).count()['Merchant_id'].hist(bins=20)

# 每月消费统计
# date_cat = pd.cut(offline_ori['Date'], range(20160100, 20160701, 100), labels=[1, 2, 3, 4, 5, 6])
# pd.value_counts(date_cat).sort_index().plot.bar()

def shared_id_generator():
    for id in online_ori['User_id'].unique():
        if id in offline_ori['User_id']:
            yield id


# 查看 用户在线上线下消费行为
t = shared_id_generator()
id = next(t)
offline_ori[offline_ori['User_id'] == id]
online_ori[online_ori['User_id'] == id]
'''
###############################################################
# offline,线下主要统计用户特征，因为商户和优惠券和线上交叉不大

# 提取打折率和满减额度
# offline没有限价优惠fixed
offline_ori['coupon_used'] = (~offline_ori['Date_received'].isna()) & (~offline_ori['Date'].isna())

# 被使用率7%
# offline_ori['coupon_used'].sum()/offline_ori['Date_received'].count()

t = offline_ori[~offline_ori['Date_received'].isna()]['Date_received'].astype(int).astype(str)
offline_ori['received_date'] = pd.to_datetime(t.str[:4] + '-' + t.str[4:6] + '-' + t.str[6:])
t = offline_ori['Date'][~offline_ori['Date'].isna()].astype(int).astype(str)

offline_ori['date'] = pd.to_datetime(t.str[:4] + '-' + t.str[4:6] + '-' + t.str[6:])
offline_ori['coupon_used_days'] = (offline_ori['date'] - offline_ori['received_date']).dt.days
offline_ori['target'] = offline_ori['coupon_used_days'] < 15

# 通过将消费，领取优惠券作为行为合并成为一列，来统计上一次的消费间隔，
offline_consume = offline_ori[~offline_ori['date'].isna()][['User_id', 'Coupon_id', 'date']]
offline_coupon_receive = offline_ori[~offline_ori['received_date'].isna()][['User_id', 'Coupon_id', 'received_date']]

offline_consume['ori_index'] = offline_consume.index
# action 0 领取优惠券，1普通消费， 2使用优惠券消费
offline_consume['action'] = np.where(offline_consume['Coupon_id'].isna(), 1, 2)
offline_consume['consume_date'] = offline_consume['date']
offline_consume['coupon_used_date'] = offline_consume['date'][offline_consume['action'] == 2]
offline_consume = offline_consume.rename(
    columns={"date": "action_date"}, errors="raise")

offline_coupon_receive['ori_index'] = offline_coupon_receive.index
offline_coupon_receive['action'] = 0
offline_coupon_receive['consume_date'] = pd.NaT
offline_coupon_receive = offline_coupon_receive.rename(
    columns={"received_date": "action_date"}, errors="raise")

offline_action = pd.concat([offline_consume, offline_coupon_receive], ignore_index=True)
offline_action = offline_action.sort_values(['action_date', 'action'])
offline_action['last_consume_date'] = offline_action.groupby('User_id')[
    'consume_date'].transform(lambda x: x.cummax(skipna=False).shift(1))
offline_action['consume_interval'] = (offline_action['action_date'] - offline_action['last_consume_date']).dt.days

offline_action['last_use_date'] = offline_action.groupby('User_id')[
    'coupon_used_date'].transform(lambda x: x.cummax(skipna=False).shift(1))
offline_action['coupon_interval'] = (offline_action['action_date'] - offline_action['last_use_date']).dt.days

# 主要是接收优惠券距离上次消费，使用优惠券的日期
offline_action = offline_action[offline_action['action'] == 0].set_index('ori_index')
offline_ori['last_consume_date'] = offline_action['last_consume_date']
offline_ori['last_use_date'] = offline_action['last_use_date']
offline_ori['consume_interval'] = offline_action['consume_interval']
offline_ori['coupon_interval'] = offline_action['coupon_interval']

offline_ori['coupon_used'] = (~offline_ori['Date_received'].isna()) & (~offline_ori['Date'].isna())
t = offline_ori[~offline_ori['Date_received'].isna()]['Date_received'].astype(int).astype(str)
offline_ori['received_date'] = pd.to_datetime(t.str[:4] + '-' + t.str[4:6] + '-' + t.str[6:])
t = offline_ori['Date'][~offline_ori['Date'].isna()].astype(int).astype(str)
offline_ori['date'] = pd.to_datetime(t.str[:4] + '-' + t.str[4:6] + '-' + t.str[6:])
offline_ori['coupon_used_days'] = (offline_ori['date'] - offline_ori['received_date']).dt.days

online_ori['coupon_used'] = (~online_ori['Date_received'].isna()) & (~online_ori['Date'].isna())
t = online_ori[~online_ori['Date_received'].isna()]['Date_received'].astype(int).astype(str)
online_ori['received_date'] = pd.to_datetime(t.str[:4] + '-' + t.str[4:6] + '-' + t.str[6:])
t = online_ori['Date'][~online_ori['Date'].isna()].astype(int).astype(str)
online_ori['date'] = pd.to_datetime(t.str[:4] + '-' + t.str[4:6] + '-' + t.str[6:])
online_ori['coupon_used_days'] = (online_ori['date'] - online_ori['received_date']).dt.days

'''
offline['last_use_date'] = offline[offline['coupon_used']].sort_values('date').groupby('User_id')[
    'date'].transform(
    lambda x: x.cummax(skipna=False).shift(1))
offline['use_interval'] = (offline['date'] - offline['last_use_date']).dt.days
offline['last_consume_date'] = offline[~offline['date'].isna()].sort_values('date').groupby('User_id')[
    'date'].transform(
    lambda x: x.cummax(skipna=False).shift(1))
offline['consume_interval'] = (offline['date'] - offline['last_consume_date']).dt.days

# 测试用
tt = offline[offline['coupon_used']][offline['User_id'].isin([803811, 6784896, 3664850])][
    ['User_id', 'date', 'received_date', 'last_use_date', 'use_interval']]
tt.sort_values('date').groupby('User_id').apply(
    lambda x: print(x['User_id'].iloc[0], ':\n', x.drop(columns=['User_id'])))
    
tt = offline[~offline['date'].isna()][offline['User_id'].isin([803811, 6784896, 3664850])][
    ['User_id', 'date', 'received_date','last_consume_date','consume_interval']]
tt.sort_values('date').groupby('User_id').apply(
    lambda x: print(x['User_id'].iloc[0], ':\n', x.drop(columns=['User_id']))) 
'''
offline = offline_ori[
    (offline_ori['date'] >= pd.to_datetime('2016-01-01')) &
    (offline_ori['date'] <= pd.to_datetime('2016-04-01'))]
online = online_ori[
    (online_ori['date'] >= pd.to_datetime('2016-01-01')) &
    (online_ori['date'] <= pd.to_datetime('2016-04-01'))]

used = offline[offline['coupon_used']][
    ['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Date_received', 'date']]
# 打折率0.95,0.8
rate = used[used['Discount_rate'].str.startswith('0')]
# 满减
value = used[used['Discount_rate'].str.contains(':')]
value[['required_value', 'discount_value']] = value['Discount_rate'].str.split(':', 2, expand=True).astype(int)
value['rate'] = value['discount_value'] / value['required_value']

# 这种切片赋值是可以的
used['rate'] = 1 - rate['Discount_rate'].astype(np.float)
used['rate'][used['Discount_rate'].str.contains(':')] = value['rate']
used[['required_value', 'discount_value']] = value[['required_value', 'discount_value']]
used[['required_value', 'discount_value']] = used[['required_value', 'discount_value']].fillna(0)

offline_user = offline[['User_id', 'Merchant_id', 'Date_received', 'Date']].groupby(
    'User_id').count()
offline_user.columns = [
    'merchant_count', 'coupon_count', 'consume_count']

# Date_received，Date均不为空，优惠券被使用，这里是布尔值，使用sum
offline_user['coupon_used_count'] = offline.groupby('User_id')['coupon_used'].sum()
offline_user['coupon_used_rate'] = offline_user['coupon_used_count'] / offline_user['coupon_count']
# offline_user['last_coupon_used'] = used.groupby('User_id')['date'].last()
offline_user['use_interval_mean'] = 182 / offline_user['coupon_used_count']
offline_user['consume_interval_mean'] = 182 / offline_user['consume_count']
offline_user['distance_mean'] = offline.groupby('User_id')['Distance'].mean()
offline_user['distance_max'] = offline.groupby('User_id')['Distance'].max()
offline_user['coupon_used_rate_min'] = used.groupby('User_id')['rate'].min()
offline_user['coupon_used_rate_mean'] = used.groupby('User_id')['rate'].mean()
offline_user['coupon_used_value_min'] = used.groupby('User_id')['required_value'].min()
offline_user['coupon_used_value_mean'] = used.groupby('User_id')['required_value'].mean()
offline_user['coupon_used_days_mean'] = offline.groupby('User_id')['coupon_used_days'].mean()
offline_user['coupon_used_days_min'] = offline.groupby('User_id')['coupon_used_days'].min()
offline_user['coupon_used_days_max'] = offline.groupby('User_id')['coupon_used_days'].max()

offline_merchant = offline[['User_id', 'Merchant_id', 'Date_received', 'Date']].groupby(
    'Merchant_id').count()
offline_merchant.columns = [
    'user_count', 'coupon_count', 'consume_count']

offline_merchant['coupon_used_count'] = offline.groupby('Merchant_id')['coupon_used'].sum()
offline_merchant['coupon_used_rate'] = offline_merchant['coupon_used_count'] / offline_merchant['coupon_count']
offline_merchant['use_interval_mean'] = 182 / offline_merchant['coupon_used_count']
offline_merchant['consume_interval_mean'] = 182 / offline_merchant['consume_count']
offline_merchant['coupon_used_rate_min'] = used.groupby('Merchant_id')['rate'].min()
offline_merchant['coupon_used_rate_mean'] = used.groupby('Merchant_id')['rate'].mean()
offline_merchant['coupon_used_value_min'] = used.groupby('Merchant_id')['required_value'].min()
offline_merchant['coupon_used_value_mean'] = used.groupby('Merchant_id')['required_value'].mean()
offline_merchant['coupon_used_days_mean'] = offline.groupby('Merchant_id')['coupon_used_days'].mean()
offline_merchant['coupon_used_days_min'] = offline.groupby('Merchant_id')['coupon_used_days'].min()
offline_merchant['coupon_used_days_max'] = offline.groupby('Merchant_id')['coupon_used_days'].max()

offline_coupon = offline[['Coupon_id', 'User_id', 'Date_received']].groupby('Coupon_id').count()
offline_coupon.columns = ['user_count', 'received_count']
offline_coupon['coupon_used_count'] = offline.groupby('Coupon_id')['coupon_used'].sum()
offline_coupon['coupon_used_rate'] = offline_coupon['coupon_used_count'] / offline_coupon['received_count']

# 用户,和商户交叉统计,采用两列groupby
offline_user_merchant = offline.groupby(['User_id', 'Merchant_id'])['Date_received', 'Date'].count()
offline_user_merchant.columns = ['received_coupon_count', 'consume_count']
offline_user_merchant['coupon_used'] = used.groupby(['User_id', 'Merchant_id'])['Date_received'].count()
offline_user_merchant['rate_min'] = used.groupby(['User_id', 'Merchant_id'])['rate'].min()
offline_user_merchant['required_value_max'] = used.groupby(['User_id', 'Merchant_id'])['required_value'].max()

##########################################################
# online

'''
#online 数据过多，直接用次数估算
online['last_use_date'] = online[online['coupon_used']].sort_values('date').groupby('User_id')[
    'date'].transform(
    lambda x: x.cummax(skipna=False).shift(1))
online['use_interval'] = (online['date'] - online['last_use_date']).dt.days
online['last_consume_date'] = online[~online['date'].isna()].sort_values('date').groupby('User_id')[
    'date'].transform(
    lambda x: x.cummax(skipna=False).shift(1))
online['consume_interval'] = (online['date'] - online['last_consume_date']).dt.days


tt = online[online['coupon_used']][online['User_id'].isin([367140, 2815379, 5733964])][
    ['User_id', 'date', 'received_date', 'last_use_date', 'use_interval']]
tt.sort_values('date').groupby('User_id').apply(
    lambda x: print(x['User_id'].iloc[0], ':\n', x.drop(columns=['User_id'])))

# 注意这里出现了n个日期相同的顺序会有点乱，但最终结果仍是对的
tt = online[~online['date'].isna()][online['User_id'].isin([367140, 2815379, 5733964])][
    ['User_id', 'date', 'received_date','last_consume_date','consume_interval']]
tt.sort_values('date').groupby('User_id').apply(
    lambda x: print(x['User_id'].iloc[0], ':\n', x.drop(columns=['User_id']))) 
'''

used = online[online['coupon_used']][
    ['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Date_received', 'date']]
rate = used[used['Discount_rate'].str.startswith('0')]
value = used[used['Discount_rate'].str.contains(':')]
value[['required_value', 'discount_value']] = value['Discount_rate'].str.split(':', 2, expand=True).astype(int)
value['rate'] = value['discount_value'] / value['required_value']
used['rate'] = 1 - rate['Discount_rate'].astype(np.float)

# 这种切片赋值是可以的
used['rate'][used['Discount_rate'].str.contains(':')] = value['rate']
used[['required_value', 'discount_value']] = value[['required_value', 'discount_value']]
used[['required_value', 'discount_value']] = used[['required_value', 'discount_value']].fillna(0)

online_user = online[['User_id', 'Coupon_id', 'Date_received', 'Date']].groupby(
    'User_id').count()
online_user.columns = [
    'merchant_count', 'coupon_count', 'consume_count']
online_user['coupon_used_count'] = online.groupby('User_id')['coupon_used'].sum()
online_user['coupon_used_rate'] = online_user['coupon_used_count'] / online_user['coupon_count']
# online 数据过多，直接用次数估算
online_user['coupon_use_interval_mean'] = 182 / online_user['coupon_used_count']
online_user['consume_interval_mean'] = 182 / online_user['consume_count']
online_user['coupon_used_rate_min'] = used.groupby('User_id')['rate'].min()
online_user['coupon_used_rate_mean'] = used.groupby('User_id')['rate'].mean()
online_user['coupon_used_value_min'] = used.groupby('User_id')['required_value'].min()
online_user['coupon_used_value_mean'] = used.groupby('User_id')['required_value'].mean()
online_user['coupon_used_days_mean'] = online.groupby('User_id')['coupon_used_days'].mean()
online_user['coupon_used_days_min'] = online.groupby('User_id')['coupon_used_days'].min()
online_user['coupon_used_days_max'] = online.groupby('User_id')['coupon_used_days'].max()
online_user['merchant_click_count'] = online[online['Action'] == 0].groupby('User_id')['Action'].count()
online_user['coupon_fixed'] = used[used['Discount_rate'] == 'fixed'].groupby(
    'User_id')['required_value'].count()


# 经过测试一次group 返回多列比多次groupby更慢。。没必要一次group返回多列
# 多次group没有直接dataframe统计快，尽量大面积操作
def user_feature_gen(x):
    offline_off_num = x['Date_received'].count()
    offline_ccf_used = ((x['Date_received'] != np.nan) & (x['Date'] != np.nan)).sum()
    offline_merchant_num = x['Merchant_id'].count()
    return offline_off_num, offline_ccf_used, offline_merchant_num


# 目标值产生
offline_train = offline_ori[
    (offline_ori['date'] >= pd.to_datetime('2016-04-01')) &
    (offline_ori['date'] <= pd.to_datetime('2016-05-01'))]
train_index = offline_train[~offline_train['Date_received'].isna()]
train_index['target'] = 1
train_index['target'][train_index['coupon_used_days'].isna()] = 0
train_index['target'][train_index['coupon_used_days'] > 15] = 0

for feature, name in zip([offline_user, online_user, offline_merchant, offline_user_merchant],
                         ['offline_user', 'online_user', 'offline_merchant', 'offline_user_merchant']):
    feature.columns = map(lambda x: name + '_' + x, feature.columns)

'''
# x特征
feature0 = offline_user.loc[train_index['User_id']]
feature1 = online_user.loc[train_index['User_id']]
feature2 = online_merchant.loc[train_index['Merchant_id']]
# 双重索引没有找到特别好的loc提取办法，用merge可以实现,所以统一用merge
'''

train_index = pd.merge(train_index, offline_user.reset_index(), how='left', on=['User_id'])
train_index = pd.merge(train_index, online_user.reset_index(), how='left', on=['User_id'])
train_index = pd.merge(train_index, offline_merchant.reset_index(), how='left', on=['Merchant_id'])
train_index = pd.merge(train_index, offline_coupon.reset_index(), how='left', on=['Coupon_id'])
train_index = pd.merge(train_index, offline_user_merchant.reset_index(), how='left', on=['User_id', 'Merchant_id'])
drop_columns = ['Coupon_id', 'Date', 'Date_received', 'Discount_rate',
       'Merchant_id', 'User_id', 'coupon_used', 'received_date', 'date',
       'coupon_used_days', 'last_consume_date', 'last_use_date']
train_index = train_index.drop(columns=drop_columns)
train_index = train_index.replace([np.inf, -np.inf], np.nan)
train_index = train_index.fillna(0)

classifiers = classifier_test(train_index.drop(columns=['target']).values, train_index['target'].values)

# pd.read_csv('sample_submission.csv')
# offline_test_ori = pd.read_csv('ccf_offline_stage1_test_revised.csv', nrows=500000)
