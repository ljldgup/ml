import pandas as pd
import numpy as np
import seaborn as sns
from pandas.tseries.offsets import Day

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
# pd.read_csv('sample_submission.csv')
'''
# 测试用少量数据就够了
offline_test_revised = pd.read_csv('ccf_offline_stage1_test_revised.csv', nrows=100000)
offline = pd.read_csv('ccf_offline_stage1_train.csv', nrows=100000)
online = pd.read_csv('ccf_online_stage1_train.csv', nrows=100000)

# 数据时按照用户id排的, 一个用户的一组行为被排在一起
online.head(20)
online.shape

# 平均每个用户大约有十几个行为。
online['User_id'].unique().shape
online['Merchant_id'].unique().shape
# 大部分用户行为集中在20次以下
# online.groupby(['User_id']).count()['Merchant_id'].hist(bins=20)

offline.shape
offline['User_id'].unique().shape
offline['Merchant_id'].unique().shape


# 大部分用户行为集中在10次以下
# offline.groupby(['User_id']).count()['Merchant_id'].hist(bins=20)

# 每月消费统计
# date_cat = pd.cut(offline['Date'], range(20160100, 20160701, 100), labels=[1, 2, 3, 4, 5, 6])
# pd.value_counts(date_cat).sort_index().plot.bar()

def shared_id_generator():
    for id in online['User_id'].unique():
        if id in offline['User_id']:
            yield id


# 查看 用户在线上线下消费行为
t = shared_id_generator()
id = next(t)
offline[offline['User_id'] == id]
online[online['User_id'] == id]

###############################################################
# offline,线下主要统计用户特征，因为商户和优惠券和线上交叉不大

# 提取打折率和满减额度
# offline没有限价优惠fixed
offline['coupon_used'] = (~offline['Date_received'].isna()) & (~offline['Date'].isna())
used = offline[offline['coupon_used']][
    ['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Date_received', 'Date']]
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

t = offline[~offline['Date_received'].isna()]['Date_received'].astype(int).astype(str)
offline['received_date'] = pd.to_datetime(t.str[:4] + '-' + t.str[4:6] + '-' + t.str[6:])
t = offline['Date'][~offline['Date'].isna()].astype(int).astype(str)
offline['date'] = pd.to_datetime(t.str[:4] + '-' + t.str[4:6] + '-' + t.str[6:])
offline['coupon_used_days'] = (offline['date'] - offline['received_date']).dt.days

offline_user = offline[['User_id', 'Merchant_id', 'Date_received', 'Date']].groupby(
    'User_id').count()
offline_user.columns = [
    'merchant_count', 'coupon_count', 'consume_count']

# Date_received，Date均不为空，优惠券被使用，这里是布尔值，使用sum
offline_user['coupon_used'] = offline.groupby('User_id')['coupon_used'].sum()
offline_user['coupon_used_interval'] = 182 / offline_user['coupon_used']
offline_user['Distance_mean'] = offline.groupby('User_id')['Distance'].mean()
offline_user['Distance_max'] = offline.groupby('User_id')['Distance'].max()
offline_user['coupon_used_rate_min'] = used.groupby('User_id')['rate'].min()
offline_user['coupon_used_rate_mean'] = used.groupby('User_id')['rate'].mean()
offline_user['coupon_used_value_min'] = used.groupby('User_id')['required_value'].min()
offline_user['coupon_used_value_mean'] = used.groupby('User_id')['required_value'].mean()
offline_user['coupon_used_days_mean'] = offline.groupby('User_id')['coupon_used_days'].mean()
offline_user['coupon_used_days_min'] = offline.groupby('User_id')['coupon_used_days'].min()
offline_user['coupon_used_days_max'] = offline.groupby('User_id')['coupon_used_days'].max()

##########################################################
# online
online['coupon_used'] = (~online['Date_received'].isna()) & (~online['Date'].isna())
used = online[online['coupon_used']][['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Date_received', 'Date']]
rate = used[used['Discount_rate'].str.startswith('0')]
value = used[used['Discount_rate'].str.contains(':')]
value[['required_value', 'discount_value']] = value['Discount_rate'].str.split(':', 2, expand=True).astype(int)
value['rate'] = value['discount_value'] / value['required_value']
used['rate'] = 1 - rate['Discount_rate'].astype(np.float)
# 这种切片赋值是可以的
used['rate'][used['Discount_rate'].str.contains(':')] = value['rate']
used[['required_value', 'discount_value']] = value[['required_value', 'discount_value']]
used[['required_value', 'discount_value']] = used[['required_value', 'discount_value']].fillna(0)

t = online[~online['Date_received'].isna()]['Date_received'].astype(int).astype(str)
online['received_date'] = pd.to_datetime(t.str[:4] + '-' + t.str[4:6] + '-' + t.str[6:])
t = online['Date'][~online['Date'].isna()].astype(int).astype(str)
online['date'] = pd.to_datetime(t.str[:4] + '-' + t.str[4:6] + '-' + t.str[6:])
online['coupon_used_days'] = (online['date'] - online['received_date']).dt.days

online_user = online[['User_id', 'Coupon_id', 'Date_received', 'Date']].groupby(
    'User_id').count()
online_user.columns = [
    'merchant_count', 'coupon_count', 'consume_count']
online_user['coupon_used'] = online.groupby('User_id')['coupon_used'].sum()
online_user['coupon_used_interval'] = 182 / offline_user['coupon_used']
online_user['coupon_used_rate_min'] = used.groupby('User_id')['rate'].min()
online_user['coupon_used_rate_mean'] = used.groupby('User_id')['rate'].mean()
online_user['coupon_used_value_min'] = used.groupby('User_id')['required_value'].min()
online_user['coupon_used_value_mean'] = used.groupby('User_id')['required_value'].mean()
online_user['coupon_used_days_mean'] = online.groupby('User_id')['coupon_used_days'].mean()
online_user['coupon_used_days_min'] = online.groupby('User_id')['coupon_used_days'].min()
online_user['coupon_used_days_max'] = online.groupby('User_id')['coupon_used_days'].max()

online_user['coupon_used_fixed_count'] = used[used['Discount_rate'] == 'fixed'].groupby(
    'User_id')['required_value'].count()
online_merchant = online[['User_id', 'Merchant_id', 'Date_received', 'Date']].groupby(
    'Merchant_id').count()
online_merchant.columns = [
    'user_count', 'coupon_count', 'consume_count']

online_merchant['coupon_received'] = online.groupby('Merchant_id')['Date_received'].count()
online_merchant['coupon_used'] = online.groupby('Merchant_id')['coupon_used'].sum()
online_merchant['coupon_used_rate_min'] = used.groupby('Merchant_id')['rate'].min()
online_merchant['coupon_used_rate_mean'] = used.groupby('Merchant_id')['rate'].mean()
online_merchant['coupon_used_value_min'] = used.groupby('Merchant_id')['required_value'].min()
online_merchant['coupon_used_value_mean'] = used.groupby('Merchant_id')['required_value'].mean()
online_merchant['coupon_used_days_mean'] = online.groupby('Merchant_id')['coupon_used_days'].mean()
online_merchant['coupon_used_days_min'] = online.groupby('Merchant_id')['coupon_used_days'].min()
online_merchant['coupon_used_days_max'] = online.groupby('Merchant_id')['coupon_used_days'].max()

# 用户,和商户交叉统计,采用两列groupby
online_user_merchant = online.groupby(['User_id', 'Merchant_id'])['Date_received', 'Date'].count()
online_user_merchant.columns = ['received_coupon_count', 'consume_count']
online_user_merchant['coupon_used_count'] = used.groupby(['User_id', 'Merchant_id'])['Date_received'].count()
online_user_merchant['rate_min'] = used.groupby(['User_id', 'Merchant_id'])['rate'].min()
online_user_merchant['required_value_max'] = used.groupby(['User_id', 'Merchant_id'])['required_value'].max()


# 经过测试一次group 返回多列比多次groupby更慢。。没必要一次group返回多列
# 多次group没有直接dataframe统计快，尽量大面积操作
def user_feature_gen(x):
    offline_off_num = x['Date_received'].count()
    offline_ccf_used = ((x['Date_received'] != np.nan) & (x['Date'] != np.nan)).sum()
    offline_merchant_num = x['Merchant_id'].count()
    return offline_off_num, offline_ccf_used, offline_merchant_num


# 目标值产生
train_index = online[~online['Date_received'].isna()]
train_index['target'] = 1
train_index['target'][train_index['coupon_used_days'].isna()] = 0
train_index['target'][train_index['coupon_used_days'] > 15] = 0

for feature, name in zip([offline_user, online_user, online_merchant, online_user_merchant],
                         ['offline_user', 'online_user', 'online_merchant', 'online_user']):
    feature.columns = map(lambda x: name + '_' + x, feature.columns)

'''
# x特征
feature0 = offline_user.loc[train_index['User_id']]
feature1 = online_user.loc[train_index['User_id']]
feature2 = online_merchant.loc[train_index['Merchant_id']]
# 双重缩影没有找到特别好的loc提取办法，用merge可以实现,所以统一用merge
'''

train_index = pd.merge(train_index, offline_user.reset_index(), how='left', on=['User_id'])
train_index = pd.merge(train_index, online_user.reset_index(), how='left', on=['User_id'])
train_index = pd.merge(train_index, online_merchant.reset_index(), how='left', on=['Merchant_id'])
train_index = pd.merge(train_index, online_user_merchant.reset_index(), how='left', on=['User_id', 'Merchant_id'])
train_index = train_index.drop(columns=['User_id', 'Merchant_id', 'Action', 'Coupon_id', 'Discount_rate',
                                        'Date_received', 'Date', 'coupon_used', 'received_date', 'date'])
