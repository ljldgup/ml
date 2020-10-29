import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthBegin

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
'''
# 测试用少量数据就够了 )  #
offline_ori = pd.read_csv('ccf_offline_stage1_train.csv')  # , nrows=10000)
online_ori = pd.read_csv('ccf_online_stage1_train.csv')  # , nrows=10000)
offline_pre_ori = pd.read_csv('ccf_offline_stage1_test_revised.csv')  # , nrows=10000)
offline_ori = pd.concat([offline_ori, offline_pre_ori], ignore_index=True)


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

'''
offline_ori = pd.read_csv('ccf_offline_stage1_train.csv')  # , nrows=10000)
online_ori = pd.read_csv('ccf_online_stage1_train.csv')  # , nrows=10000)
offline_pre_ori = pd.read_csv('ccf_offline_stage1_test_revised.csv')  # , nrows=10000)
offline_ori = pd.concat([offline_ori, offline_pre_ori], ignore_index=True)
###############################################################
# offline,线下主要统计用户特征，因为商户和优惠券和线上交叉不大

# 提取打折率和满减额度
# offline没有限价优惠fixed
offline_ori['used'] = (~offline_ori['Date_received'].isna()) & (~offline_ori['Date'].isna())

# 被使用率7%
# offline_ori['used'].sum()/offline_ori['Date_received'].count()

t = offline_ori[~offline_ori['Date_received'].isna()]['Date_received'].astype(int).astype(str)
offline_ori['received_date'] = pd.to_datetime(t.str[:4] + '-' + t.str[4:6] + '-' + t.str[6:])
offline_ori['weekday'] = offline_ori['received_date'].dt.weekday
offline_ori['mouthday'] = offline_ori['received_date'].dt.day
t = offline_ori['Date'][~offline_ori['Date'].isna()].astype(int).astype(str)
offline_ori['date'] = pd.to_datetime(t.str[:4] + '-' + t.str[4:6] + '-' + t.str[6:])
offline_ori['used_days'] = (offline_ori['date'] - offline_ori['received_date']).dt.days
offline_ori['target'] = offline_ori['used_days'] < 15

offline_ori['filter_date'] = pd.to_datetime(np.where(offline_ori['received_date'].isna(), offline_ori['date'],
                                                     offline_ori['received_date']))

# 通过将消费，领取优惠券作为行为合并成为一列，来统计上一次的消费间隔，
offline_consume = offline_ori[~offline_ori['date'].isna()][['User_id', 'Coupon_id', 'date']]
offline_coupon_receive = offline_ori[~offline_ori['received_date'].isna()][['User_id', 'Coupon_id', 'received_date']]

offline_consume['ori_index'] = offline_consume.index
# action 0 领取优惠券，1普通消费， 2使用优惠券消费
offline_consume['action'] = np.where(offline_consume['Coupon_id'].isna(), 1, 2)
offline_consume['consume_date'] = offline_consume['date']
offline_consume['used_date'] = offline_consume['date'][offline_consume['action'] == 2]
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
    'used_date'].transform(lambda x: x.cummax(skipna=False).shift(1))
offline_action['used_interval'] = (offline_action['action_date'] - offline_action['last_use_date']).dt.days

# 主要是接收优惠券距离上次消费，使用优惠券的日期
offline_action = offline_action[offline_action['action'] == 0].set_index('ori_index')
offline_ori['last_consume_date'] = offline_action['last_consume_date']
offline_ori['last_use_date'] = offline_action['last_use_date']
offline_ori['consume_interval'] = offline_action['consume_interval']
offline_ori['used_interval'] = offline_action['used_interval']

# 优惠券转换
value = offline_ori[offline_ori['Discount_rate'].str.contains(':').fillna(False)]
offline_ori[['required_value', 'discount_value']] = value['Discount_rate'].str.split(':', 2, expand=True).astype(int)
offline_ori['rate'] = offline_ori['discount_value'] / offline_ori['required_value']
rate_index = offline_ori['Discount_rate'].str.startswith('0').fillna(False)
rate = offline_ori['Discount_rate'][rate_index]
# 是指价格减让与原价的百分率 如折扣率20%，成交价格=原价×（1-20%）。
offline_ori['rate'][rate_index] = 1 - rate.astype(float)

offline_ori['coupon_type'][offline_ori['Discount_rate'].str.startswith('0').fillna(False)] = 1
offline_ori['coupon_type'][offline_ori['Discount_rate'].str.contains(':').fillna(False)] = 2

# 统计消费券使用距领取过了多少天
offline_ori['used'] = (~offline_ori['Date_received'].isna()) & (~offline_ori['Date'].isna())
t = offline_ori[~offline_ori['Date_received'].isna()]['Date_received'].astype(int).astype(str)
offline_ori['received_date'] = pd.to_datetime(t.str[:4] + '-' + t.str[4:6] + '-' + t.str[6:])
t = offline_ori['Date'][~offline_ori['Date'].isna()].astype(int).astype(str)
offline_ori['date'] = pd.to_datetime(t.str[:4] + '-' + t.str[4:6] + '-' + t.str[6:])
offline_ori['used_days'] = (offline_ori['date'] - offline_ori['received_date']).dt.days

online_ori['used'] = (~online_ori['Date_received'].isna()) & (~online_ori['Date'].isna())
t = online_ori[~online_ori['Date_received'].isna()]['Date_received'].astype(int).astype(str)
online_ori['received_date'] = pd.to_datetime(t.str[:4] + '-' + t.str[4:6] + '-' + t.str[6:])
t = online_ori['Date'][~online_ori['Date'].isna()].astype(int).astype(str)
online_ori['date'] = pd.to_datetime(t.str[:4] + '-' + t.str[4:6] + '-' + t.str[6:])
online_ori['used_days'] = (online_ori['date'] - online_ori['received_date']).dt.days
online_ori['filter_date'] = pd.to_datetime(np.where(online_ori['received_date'].isna(), online_ori['date'],
                                                    online_ori['received_date']))

value = online_ori[online_ori['Discount_rate'].str.contains(':').fillna(False)]
online_ori[['required_value', 'discount_value']] = value['Discount_rate'].str.split(':', 2, expand=True).astype(int)
online_ori['rate'] = online_ori['discount_value'] / online_ori['required_value']
rate_index = online_ori['Discount_rate'].str.startswith('0').fillna(False)
rate = online_ori['Discount_rate'][rate_index]
online_ori['rate'][rate_index] = 1 - rate.astype(float)
online_ori['coupon_type]=np.na
offline_ori.to_csv('offline_pre.csv')
online_ori.to_csv('online_pre.csv')
'''
# 直接读取之前计算好的
offline_ori = pd.read_csv('offline_pre.csv')
online_ori = pd.read_csv('online_pre.csv')
offline_ori['filter_date'] = pd.to_datetime(offline_ori['filter_date'])
online_ori['filter_date'] = pd.to_datetime(online_ori['filter_date'])
offline_ori['received_date'] = pd.to_datetime(offline_ori['received_date'])
online_ori['received_date'] = pd.to_datetime(online_ori['received_date'])
offline_ori['date'] = pd.to_datetime(offline_ori['date'])
online_ori['date'] = pd.to_datetime(online_ori['date'])

'''
# 测试用
tt = offline[offline['used']][offline['User_id'].isin([803811, 6784896, 3664850])][
    ['User_id', 'date', 'received_date', 'last_received_date', 'received_interval']]
tt.sort_values('date').groupby('User_id').apply(
    lambda x: print(x['User_id'].iloc[0], ':\n', x.drop(columns=['User_id'])))
    
'''
'''
# 需要的价格越高，核销率越低
(offline_ori.groupby('required_value')['used'].sum()/offline_ori.groupby('required_value')['Coupon_id'].count()).plot.bar()
# 线上几乎没有任何区别
(online_ori.groupby('required_value')['used'].sum()/online_ori.groupby('required_value')['Coupon_id'].count()).plot.bar()


'''


# 经过测试一次group 返回多列比多次groupby更慢。。没必要一次group返回多列
# 多次group没有直接dataframe统计快，尽量大面积操作
def user_feature_gen(x):
    offline_off_num = x['Date_received'].count()
    offline_ccf_used = ((x['Date_received'] != np.nan) & (x['Date'] != np.nan)).sum()
    offline_merchant_num = x['Merchant_id'].count()
    return offline_off_num, offline_ccf_used, offline_merchant_num


####################################################################################
# 这里用于统计信息,这里是统计的区间，按filter_date统计，预测样本要按照received_date分
# filter_date优先使用Date_received，保证不会统计到预测的时间窗口造成
# 参与统计的月份
mouth_num = 3
#  for start_date in (,):
for i in range(7 - mouth_num):
    # 这里如果把预测月份也统计进去就是leakage
    offline = offline_ori[
        (offline_ori['filter_date'] >= pd.to_datetime('2016-01-01') + MonthBegin(i)) &
        (offline_ori['filter_date'] < pd.to_datetime('2016-01-01') + MonthBegin(i + mouth_num))]
    online = online_ori[
        (online_ori['filter_date'] >= pd.to_datetime('2016-01-01') + MonthBegin(i)) &
        (online_ori['filter_date'] < pd.to_datetime('2016-01-01') + MonthBegin(i + mouth_num))]

    offline_user = offline.groupby('User_id')[['Merchant_id', 'Date_received', 'Date']].count()
    offline_user.columns = ['action_count', 'received_count', 'consume_count']

    # Date_received，Date均不为空，优惠券被使用，这里是布尔值，使用sum
    offline_user['used_count'] = offline.groupby('User_id')['used'].sum()
    offline_user['unused_count'] = offline_user['received_count'] - offline_user['used_count']
    offline_user['used_rate'] = offline_user['used_count'] / offline_user['received_count']
    # 200及以上核销率较低，单独计算其核销率
    coupon_200_received_count = offline[offline['required_value'] >= 200].groupby('User_id')['Date_received'].count()
    coupon_200_used_count = offline[offline['required_value'] >= 200].groupby('User_id')['used'].sum()
    offline_user['>=200_received_rate'] = coupon_200_received_count / offline_user['received_count']
    offline_user['>=200_used_rate'] = coupon_200_used_count / coupon_200_received_count
    offline_user['consume_with_coupon_rate'] = offline_user['used_count'] / offline_user['consume_count']
    offline_user['consume_rate'] = offline_user['consume_count'] / offline_user['merchant_count']
    offline_user['merchant_nunique'] = offline.groupby('User_id')['Merchant_id'].nunique()
    offline_user['each_merchant_used_count'] = offline_user['used_count'] / offline_user['merchant_nunique']
    offline_user['each_merchant_consume_count'] = offline_user['consume_count'] / offline_user['merchant_nunique']
    # offline_user['used_interval'] = 30*mouth_num / offline_user['used_count']
    # offline_user['received_interval'] = 30 * mouth_num / offline_user['received_count']
    # offline_user['consume_interval'] = 30*mouth_num / offline_user['received_count']
    # offline_user['consume_interval_max'] = offline[~offline['Date'].isna()].groupby('User_id')['consume_interval'].max()
    # offline_user['consume_interval_min'] = offline[~offline['Date'].isna()].groupby('User_id')['consume_interval'].min()
    # offline_user['consume_interval_mean'] = offline[~offline['Date'].isna()].groupby('User_id')['consume_interval'].mean()
    # offline_user['used_interval'] = offline[offline['used']].groupby('User_id')['used_interval'].max()
    # offline_user['used_interval'] = offline[offline['used']].groupby('User_id')['used_interval'].min()
    # offline_user['used_interval'] = offline[offline['used']].groupby('User_id')['used_interval'].mean()
    # offline_user['distance_mean'] = offline[offline['used']].groupby('User_id')['Distance'].mean()
    offline_user['distance_max'] = offline[offline['used']].groupby('User_id')['Distance'].max()
    offline_user['distance_min'] = offline[offline['used']].groupby('User_id')['Distance'].min()
    offline_user['used_rate_min'] = offline[offline['used']].groupby('User_id')['rate'].min()
    offline_user['used_rate_max'] = offline[offline['used']].groupby('User_id')['rate'].max()
    offline_user['used_rate_mean'] = offline[offline['used']].groupby('User_id')['rate'].mean()
    offline_user['used_value_min'] = offline[offline['used']].groupby('User_id')['required_value'].min()
    offline_user['used_value_max'] = offline[offline['used']].groupby('User_id')['required_value'].max()
    offline_user['used_value_mean'] = offline[offline['used']].groupby('User_id')['required_value'].mean()
    offline_user['used_days_mean'] = offline[offline['used']].groupby('User_id')['used_days'].mean()
    # offline_user['used_days_min'] = offline[offline['used']].groupby('User_id')['used_days'].min()
    # offline_user['used_days_max'] = offline[offline['used']].groupby('User_id')['used_days'].max()
    offline_user['used_days<15_count'] = offline[offline['used_days'] < 15].groupby('User_id')['used_days'].count()
    offline_user['used_days<15_rate'] = offline_user['used_days<15_count'] / offline_user['used_count']

    offline_merchant = offline.groupby('Merchant_id')[['User_id', 'Date_received', 'Date']].count()
    offline_merchant.columns = ['user_count', 'received_count', 'consume_count']

    offline_merchant['used_count'] = offline.groupby('Merchant_id')['used'].sum()
    offline_merchant['unused_count'] = offline_merchant['received_count'] - offline_merchant['used_count']
    offline_merchant['used_rate'] = offline_merchant['used_count'] / offline_merchant['received_count']
    offline_merchant['consume_with_coupon_rate'] = offline_merchant['used_count'] / offline_merchant[
        'consume_count']
    offline_merchant['user_nunique'] = offline.groupby('Merchant_id')['User_id'].nunique()
    offline_merchant['each_user_consume_count'] = offline_merchant['consume_count'] / offline_merchant['user_nunique']
    offline_merchant['coupon_nunique'] = offline.groupby('Merchant_id')['Coupon_id'].nunique()
    offline_merchant['each_coupon_used_count'] = offline_merchant['used_count'] / offline_merchant['coupon_nunique']
    offline_merchant['used_interval'] = 30 * mouth_num / offline_user['used_count']
    # offline_merchant['received_interval'] = 30 * mouth_num / offline_user['received_count']
    # offline_merchant['consume_interval'] = offline[~offline['Date'].isna()].groupby('Merchant_id')[
    #    'consume_interval'].mean()
    offline_merchant['used_rate_max'] = offline[offline['used']].groupby('Merchant_id')['rate'].max()
    offline_merchant['used_rate_min'] = offline[offline['used']].groupby('Merchant_id')['rate'].min()
    offline_merchant['used_rate_mean'] = offline[offline['used']].groupby('Merchant_id')['rate'].mean()
    offline_merchant['used_value_max'] = offline[offline['used']].groupby('Merchant_id')['required_value'].max()
    offline_merchant['used_value_min'] = offline[offline['used']].groupby('Merchant_id')['required_value'].min()
    offline_merchant['used_value_mean'] = offline[offline['used']].groupby('Merchant_id')['required_value'].mean()
    offline_merchant['distance_mean'] = offline[offline['used']].groupby('Merchant_id')['Distance'].mean()
    offline_merchant['distance_max'] = offline[offline['used']].groupby('Merchant_id')['Distance'].max()
    offline_merchant['distance_min'] = offline[offline['used']].groupby('Merchant_id')['Distance'].min()
    offline_merchant['used_days_mean'] = offline[offline['used']].groupby('Merchant_id')['used_days'].mean()

    offline_coupon = offline.groupby('Coupon_id')[['User_id', 'Date_received']].count()
    offline_coupon.columns = ['user_count', 'received_count']
    offline_coupon['used_count'] = offline.groupby('Coupon_id')['used'].sum()
    offline_coupon['unused_count'] = offline_coupon['received_count'] - offline_coupon['used_count']
    offline_coupon['used_rate'] = offline_coupon['used_count'] / offline_coupon['received_count']
    offline_coupon['used_interval'] = 30 * mouth_num / offline_coupon['used_count']
    offline_coupon['used_days<15_count'] = offline[offline['used_days'] < 15].groupby('Coupon_id')['used_days'].count()
    offline_coupon['used_days<15_rate'] = offline_coupon['used_days<15_count'] / offline_coupon['used_count']

    # 用户,和商户交叉统计,采用两列groupby
    offline_user_merchant = offline.groupby(['User_id', 'Merchant_id'])['Date_received', 'Date'].count()
    offline_user_merchant.columns = ['received_count', 'consume_count']
    offline_user_merchant['used_count'] = offline.groupby(['User_id', 'Merchant_id'])['used'].sum()
    offline_user_merchant['used_rate'] = offline_user_merchant['used_count'] / offline_user_merchant['received_count']
    offline_user_merchant['unused_count'] = offline_user_merchant['received_count'] - offline_user_merchant[
        'used_count']
    offline_user_merchant['consume_with_coupon_rate'] = offline_user_merchant['used_count'] / offline_user_merchant[
        'consume_count']
    offline_user_merchant['value_min'] = offline[offline['used']].groupby(['User_id', 'Merchant_id'])[
        'required_value'].min()
    offline_user_merchant['value_mean'] = offline[offline['used']].groupby(['User_id', 'Merchant_id'])[
        'required_value'].mean()
    offline_user_merchant['rate_max'] = offline[offline['used']].groupby(['User_id', 'Merchant_id'])['rate'].max()
    offline_user_merchant['used_interval'] = 30 * mouth_num / offline_user_merchant['used_count']
    offline_user_merchant['coupon_receive_interval'] = 30 * mouth_num / offline_user_merchant[
        'received_count']
    offline_user_merchant['consume_interval'] = 30 * mouth_num / offline_user_merchant['consume_count']
    offline_user_coupon = offline[['Coupon_id', 'User_id', 'Date_received']].groupby(['User_id', 'Coupon_id']).count()
    offline_user_coupon.columns = ['received_count']
    offline_user_coupon['used_count'] = offline.groupby(['User_id', 'Coupon_id'])['used'].sum()
    offline_user_coupon['unused_count'] = offline_user_coupon['received_count'] - offline_user_coupon['used_count']
    offline_user_coupon['used_rate'] = offline_user_coupon['used_count'] / offline_user_coupon[
        'received_count']
    offline_user_coupon['used_interval'] = 30 * mouth_num / offline_user_coupon['used_count']
    offline_user_coupon['coupon_received_interval'] = 30 * mouth_num / offline_user_coupon['received_count']
    ##########################################################
    # online 的数据作用不大
    # 这里有date代表是有操作，不一定消费，有可能是点击 receive后直接使用action为1
    online_user = online.groupby('User_id')[['Date_received', 'Action']].count()
    online_user.columns = ['received_count', 'action_count']
    online_user['merchant_nunique'] = online.groupby('User_id')['Merchant_id'].nunique()
    online_user['used_count'] = online.groupby('User_id')['used'].sum()
    online_user['unused_count'] = online_user['received_count'] - online_user['used_count']
    online_user['used_rate'] = online_user['used_count'] / online_user['received_count']
    online_user['consume_count'] = online[online['Action'] == 1].groupby('User_id')['Action'].count()
    online_user['click_count'] = online[online['Action'] == 0].groupby('User_id')['Action'].count()
    online_user['consume_rate'] = online_user['consume_count'] / online_user['action_count']
    online_user['click_rate'] = online_user['click_count'] / online_user['action_count']
    offline_unconsume_count = offline_user['action_count'] - offline_user['consume_count']
    online_unconsume_count = online_user['action_count'] - online_user['consume_count']
    online_user['offline_not_consume_rate'] = offline_unconsume_count / (
            offline_unconsume_count + online_unconsume_count)
    online_user['offline_received_rate'] = offline_user['receive_count'] / (
            online_user['receive_count'] + offline_user['receive_count'])
    online_user['offline_used_rate'] = offline_user['used_count'] / (
            online_user['used_count'] + offline_user['used_count'])

    # 训练样本要按照接受优惠券的时间,不能与统计特征重叠，否则存在泄漏
    input_feature = offline_ori[
        (offline_ori['received_date'] >= pd.to_datetime('2016-01-01') + MonthBegin(i + mouth_num)) &
        (offline_ori['received_date'] < pd.to_datetime('2016-01-01') + MonthBegin(i + mouth_num + 1))]
    input_feature = input_feature[~input_feature['Date_received'].isna()]
    input_feature['target'] = 1
    input_feature['target'][input_feature['used_days'].isna()] = 0
    input_feature['target'][input_feature['used_days'] > 15] = 0

    input_feature['last_received_date'] = input_feature.sort_values('received_date').groupby('User_id')[
        'received_date'].transform(
        lambda x: x.cummax(skipna=False).shift(1))

    input_feature['received'] = 1
    input_feature['received_times'] = input_feature.sort_values('received_date').groupby('User_id')[
        'received'].transform(
        lambda x: x.cumsum(skipna=False).shift(1))

    input_feature['reversed_received_times'] = input_feature.sort_values('received_date').groupby('User_id')[
        'received_times'].transform(
        lambda x: x[::-1])

    input_feature['received_interval'] = (input_feature['received_date'] - input_feature['last_received_date']).dt.days
    leakage_user = pd.DataFrame({'received_count': input_feature.groupby('User_id')[['Date_received']].count()})
    leakage_user['merchant_nunique'] = input_feature.groupby('User_id')['Merchant_id'].nunique()
    leakage_user['coupon_nunique'] = input_feature.groupby('User_id')['Merchant_id'].nunique()

    leakage_merchant = pd.DataFrame({'received_count': input_feature.groupby('Merchant_id')['Date_received'].count()})
    leakage_merchant['user_nunique'] = input_feature.groupby('Merchant_id')['User_id'].nunique()
    leakage_merchant['coupon_nunique'] = input_feature.groupby('Merchant_id')['Coupon_id'].nunique()

    leakage_user_merchant = pd.DataFrame(
        {'received_count': input_feature.groupby(['User_id', 'Merchant_id'])['Date_received'].count()})

    leakage_user_coupon = pd.DataFrame(
        {'received_count': input_feature.groupby(['User_id', 'Coupon_id'])['Date_received'].count()})

    leakage_merchant_coupon = pd.DataFrame(
        {'received_count': input_feature.groupby(['Merchant_id', 'Coupon_id'])['Date_received'].count()})

    leakage_user_date = pd.DataFrame(
        {'received_count': input_feature.groupby(['User_id', 'Date_received'])['Merchant_id'].count()})

    leakage_user_coupon_date = pd.DataFrame(
        {'received_count': input_feature.groupby(['User_id', 'Coupon_id', 'Date_received'])['Merchant_id'].count()})

    for feature, name in zip(
            [offline_user, online_user, offline_merchant, offline_user_merchant, offline_coupon,
             offline_user_coupon],
            ['offline_user', 'online_user', 'offline_merchant', 'offline_user_merchant', 'offline_coupon',
             'offline_user_coupon']):
        feature.columns = map(lambda x: name + '_' + x, feature.columns)

    for feature, name in zip(
            [leakage_user, leakage_merchant, leakage_user_merchant, leakage_user_coupon,
             leakage_merchant_coupon, leakage_user_date, leakage_user_coupon_date],
            ['leakage_user', 'leakage_merchant', 'leakage_user_merchant', 'leakage_user_coupon',
             'leakage_merchant_coupon', 'leakage_user_date', 'leakage_user_coupon_date']):
        feature.columns = map(lambda x: name + '_' + x, feature.columns)

    '''
    # x特征
    feature0 = offline_user.loc[train_index['User_id']]
    feature1 = online_user.loc[train_index['User_id']]
    feature2 = online_merchant.loc[train_index['Merchant_id']]
    # 双重索引没有找到特别好的loc提取办法，用merge可以实现,所以统一用merge
    '''

    input_feature = pd.merge(input_feature, offline_user.reset_index(), how='left', on=['User_id'])
    input_feature = pd.merge(input_feature, online_user.reset_index(), how='left', on=['User_id'])
    input_feature = pd.merge(input_feature, offline_merchant.reset_index(), how='left', on=['Merchant_id'])
    input_feature = pd.merge(input_feature, offline_coupon.reset_index(), how='left', on=['Coupon_id'])
    input_feature = pd.merge(input_feature, offline_user_merchant.reset_index(), how='left',
                             on=['User_id', 'Merchant_id'])
    input_feature = pd.merge(input_feature, offline_user_coupon.reset_index(), how='left',
                             on=['User_id', 'Coupon_id'])
    # 用户在当前商家操作占所有商家的比例
    input_feature['offline_merchant_user_consume_rate'] = input_feature[
                                                              'offline_user_merchant_consume_count'] / input_feature[
                                                              'offline_user_consume_count']
    input_feature['offline_user_used_merchant_rate'] = input_feature[
                                                           'offline_user_merchant_used_count'] / input_feature[
                                                           'offline_user_used_count']
    input_feature['offline_user_unused_merchant_rate'] = input_feature[
                                                             'offline_user_merchant_unused_count'] / input_feature[
                                                             'offline_user_unused_count']
    input_feature['offline_merchant_used_user_rate'] = input_feature[
                                                           'offline_user_merchant_used_count'] / input_feature[
                                                           'offline_merchant_used_count']
    input_feature['offline_merchant_unused_user_rate'] = input_feature[
                                                             'offline_user_merchant_unused_count'] / input_feature[
                                                             'offline_merchant_unused_count']
    input_feature['offline_user_used_coupon_rate'] = input_feature[
                                                         'offline_user_coupon_used_count'] / input_feature[
                                                         'offline_user_used_count']

    input_feature = pd.merge(input_feature, leakage_user.reset_index(), how='left', on=['User_id'])
    input_feature = pd.merge(input_feature, leakage_merchant.reset_index(), how='left', on=['Merchant_id'])
    input_feature = pd.merge(input_feature, leakage_user_merchant.reset_index(), how='left',
                             on=['User_id', 'Merchant_id'])
    input_feature = pd.merge(input_feature, leakage_user_coupon.reset_index(), how='left', on=['User_id', 'Coupon_id'])
    input_feature = pd.merge(input_feature, leakage_merchant_coupon.reset_index(), how='left',
                             on=['Merchant_id', 'Coupon_id'])
    input_feature = pd.merge(input_feature, leakage_user_date.reset_index(), how='left',
                             on=['User_id', 'Date_received'])
    input_feature = pd.merge(input_feature, leakage_user_coupon_date.reset_index(), how='left',
                             on=['User_id', 'Coupon_id', 'Date_received'])

    coupon_rate_columns = input_feature.columns[input_feature.columns.str.contains('rate_m')]
    for c in coupon_rate_columns:
        input_feature[c + '-rate'] = input_feature[c] - input_feature['rate']
    coupon_value_columns = input_feature.columns[input_feature.columns.str.contains('value_m')]
    for c in coupon_value_columns:
        input_feature[c + '-value'] = input_feature[c] - input_feature['required_value']
    coupon_distance_columns = input_feature.columns[input_feature.columns.str.contains('distance_m')]
    for c in coupon_distance_columns:
        input_feature[c + '-distance'] = input_feature[c] - input_feature['Distance']
    st_date = (pd.to_datetime('2016-01-01') + MonthBegin(i + mouth_num)).strftime('%Y-%m-%d')
    ed_date = (pd.to_datetime('2016-01-01') + MonthBegin(i + mouth_num + 1)).strftime('%Y-%m-%d')
    input_feature.to_csv('{}m_samples_{}~{}.csv'.format(mouth_num, st_date, ed_date))

    '''
    t = offline.groupby('User_id')['Coupon_id'].count()
    t[(t < 20) & (t > 15)].head(1)
    offline[offline['User_id'] == 308925][['Merchant_id', 'Coupon_id', 'Date_received', 'Discount_rate', 'Date']]
    offline_user.loc[7726]
    offline[(offline['User_id'] == 7726) & (offline['Merchant_id'] == 1080)][
        ['Merchant_id', 'Coupon_id', 'Date_received', 'Discount_rate', 'Date']]
    offline_user_merchant.loc[(7726, 1080)]
    offline_user_coupon.loc[(7726, 4275)]
    '''
