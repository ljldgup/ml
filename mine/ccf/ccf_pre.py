import pandas as pd
import numpy as np
import seaborn as sns

'''
用户之间无关，但商户和优惠券时有关的
用户的动作频率要考虑
'''

# 用户O2O线下优惠券使用预测样本
offline_test_revised = pd.read_csv('ccf_offline_stage1_test_revised.csv')
# 用户线下消费和优惠券领取行为
offline = pd.read_csv('ccf_offline_stage1_train.csv')
#  用户线上点击/消费和优惠券领取行为
online = pd.read_csv('ccf_online_stage1_train.csv')
# 选手提交文件字段
# pd.read_csv('sample_submission.csv')

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


def shared_id_generator():
    for id in online['User_id'].unique():
        if id in offline['User_id']:
            yield id


# 查看 用户在线上线下消费行为
t = shared_id_generator()
id = next(t)
offline[offline['User_id'] == id]
online[online['User_id'] == id]

user_feature = pd.DataFrame(online.groupby('User_id')['Action'].count())
user_feature = user_feature.rename(columns={"Action": "online_action_count"})
user_feature['action_2_count'] = online[online['Action'] == 2].groupby('User_id')['Action'].count()



# 线下领取优惠券数量
user_feature['offline_off_num'] = offline.groupby('User_id').apply(lambda x: x['Date_received'].count())
# 线下实际使用优惠券
user_feature['offline_off_used'] = offline.groupby('User_id').apply(
    lambda x: x[(x['Date_received'] != np.nan) & (x['Date'] != np.nan)].count())

