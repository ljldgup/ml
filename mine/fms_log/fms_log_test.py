from collections import defaultdict
from datetime import date, datetime
from functools import reduce

from pyspark import SparkContext, SparkConf
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StringIndexer, VectorIndexer, CountVectorizer
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, array_contains, col, explode
from pyspark.sql.pandas.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructField, StringType, ArrayType, DateType, TimestampType
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.evaluation import ClusteringEvaluator

# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import OrdinalEncoder

'''
处理e6 文件信息
主要是通过udf 增加列和pandas_udf聚合
mapReduce的操作思想比较重要
传统的顺序操作，比较适合将数据集缩小后使用pandas等进行
'''

appName = 'fms_log_test'
master = 'local'
spark = SparkSession \
    .builder \
    .appName("appName") \
    .getOrCreate()
sc = spark.sparkContext


############################################################
# 将日志信息转化位表中一行
def log_to_row(line):
    if 'starting' in line:
        attr0 = line.split(' ----- ')
        return attr0[0], -1, -1, 'start', attr0[1]

    attr0 = line.split(' : ')
    attr1 = attr0[0].split('/')
    # 时间，线程，操作，日志等及, 具体日志
    # attr1[6]具体日志，在实际操作中应该变为类型
    return attr1[0], int(attr1[1]), int(attr1[2]), attr1[3], attr0[1]


# 读取文件转表
lines = sc.textFile("fms_log.txt")

# 去掉空行 以及特殊信息
info = lines.filter(lambda line: ' : ' in line or ' ----- ' in line).map(log_to_row)

df = spark.createDataFrame(
    info, "date_str string, thread_num int, operation int, level string, context string")


#######################################################################
def words_padding(x):
    return x.strftime("%b %d %H")


tokenizer = Tokenizer(inputCol="context", outputCol="context_words")
df = tokenizer.transform(df)


# indexer = StringIndexer(inputCol="context_words", outputCol="context_words_label")
# indexed = indexer.fit(df).transform(df)

#####################################################################
# 对日志内容进行编码，目前未使用，发现fp growth不用编码也可以使用
def context_process(x):
    return x.split(' ')[0]


# dataframe的映射需要用withColumn或select + udf,select 需要重命名
context_process = udf(context_process, StringType())
df = df.withColumn('one_context', context_process(df['context']))

indexer = StringIndexer(inputCol="one_context", outputCol="label_context")
indexed = indexer.fit(df).transform(df)


####################################################

# 字符串转时间
def str_2_date(x):
    return datetime.strptime(x, "%a %b %d %H:%M:%S %Y")


# 时间转月日分钟字符串，用于groupby统计
def date_2_min(x):
    return x.strftime("%b %d %H")


# dataframe的映射需要用withColumn或select + udf,select 需要重命名相当于rdd中的map
# TimestampType 能保留时分秒 ， DateType 会被转成 date类型
str_2_date = udf(str_2_date, TimestampType())
date_2_min = udf(date_2_min, StringType())

df = df.withColumn('date', str_2_date(df['date_str']))
# 这里发现str_2_date 输入已经是datetime 可能spark 进行了处理
df = df.withColumn('min_str', date_2_min(df['date']))

# 每分钟新开资源统计
df.filter(df['one_context'].startswith('Begin')).groupby(df['min_str']).count().show(10, False)

words = df.select('context_words').withColumn('words', explode(col('context_words')))
words_count = words.groupby('words').count().filter(col('count') > 20).select('words')
high_freq_words = set(map(lambda x: x.asDict()['words'], words_count.collect()))
sc.brao

def remove_low_freq_words(words):
    # print(words)
    rst = list(filter(lambda x: x in high_freq_words, words))
    # print(rst)
    return rst


remove_low_freq_words = udf(remove_low_freq_words, ArrayType(StringType()))

df = df.withColumn('high_freq_words', remove_low_freq_words(col('context_words')))
cv = CountVectorizer(inputCol="high_freq_words", outputCol="words_features", vocabSize=len(high_freq_words))
model = cv.fit(df)
df = model.transform(df)

kmeans = KMeans(featuresCol="words_features", predictionCol="kmeans_prediction").setK(100).setSeed(1)
model = kmeans.fit(df)
predictions = model.transform(df)
predictions.select(['context_words', 'high_freq_words', 'kmeans_prediction']).show(100,truncate=False)


#################################################################
# 按线程分组统计

# pandas_udf用于分组后的操作 GROUPED_AGG返回一条，GROUPED_MAP一对一
# pdf是pandas的dataframe
@pandas_udf("string ",
            PandasUDFType.GROUPED_AGG)
def collect_context(pdf):
    return pdf.map(lambda c: '{0[0]} -> '.format(c.split(' '))).sum()


@pandas_udf("timestamp",
            PandasUDFType.GROUPED_AGG)
def start_time(pdf):
    return pdf.iloc[0]


# 注意select根据条件选择，列名就是条件，过滤要用filter，
# 这里groupby用apply是对等转换，无法聚合需要用agg
thread_process = df.filter(df['thread_num'] > 0). \
    filter(~ df['context'].startswith('Bug')). \
    groupBy(df['thread_num']). \
    agg(collect_context(df['context']), start_time(df['date']))

# withColumnRenamed列重命名
thread_process = thread_process.withColumnRenamed('collect_context(context)', 'c_context'). \
    withColumnRenamed('start_time(date_str)', 'start_time')

thread_process.cache()
##############################################################
crash_time = df.filter(df['context'].startswith('start'))
crash_time.createOrReplaceTempView("crash_time")
thread_process.createOrReplaceTempView('thread_process')

"""
# spark联立没法用不等于，不知道怎么简洁的实现,估计只能用pandas,或者用udf映射
mysql 可以用limit 1
thread_crash_time = spark.sql('''
    select thread_num, c_context, start_time,
    (select date from crash_time where date > start_time limit 1) as c_time from thread_process
    ''')
"""


#######################################################
# 预处理
# 分割日志成位列表
def context_split(x):
    t = x.split(' -> ')[:-1]
    t.append('crash')
    return t


# dataframe的映射需要用withColumn或select + udf,select 需要重命名
context_split = udf(context_split, ArrayType(StringType()))

# spark 逻辑符号用 '&' for 'and', '|' for 'or', '~' for 'not'
# withColumn 新增一列相当与map
thread_process = thread_process.filter(~ thread_process['c_context'].endswith('Open,Contexts -> ')). \
    withColumn('l_context', context_split(col('c_context')))

thread_process.groupby('c_context').count().head(10)

# .withColumn('l_context', context_split(thread_process['c_context']))

'''
# 使用fpGrowth寻找频繁项
fpGrowth = FPGrowth(itemsCol="l_context", minSupport=0.5, minConfidence=0.6)
model = fpGrowth.fit(thread_process)

get_last_item = udf(lambda x: x[-1], StringType())

# 通过array_contains 筛选，col('items')代表自己当前这一列
model.freqItemsets.filter(array_contains(col('items'), 'crash')).show(truncate=False)

# 通过将items映射成string列，然后筛选，这样分拆过程后很明显更简单
model.freqItemsets.withColumn('last_item', get_last_item(col('items'))) \
    .filter(col('last_item').endswith('crash')) \
    .show(10)

# csv 无法保存 arrayType
# thread_process.write.mode('Overwrite').format('csv').save('fms')
'''

pd_df = df.toPandas()
pd_df = pd_df.set_index('date')
pd_df.resample('w').count()
pd_df.resample('d').count()

# 可以加数字表示几分钟
pd_df.resample('5min').count()
# 这里中间没有的时间也会算，所以很慢
pd_df.resample('min').apply(lambda x: x[x['context'].str.contains('Begin')]['context'].count())

'''
k mean 对日志文本分类
steps_logs = df.toPandas()
word_freq = defaultdict(int)
for words in steps_logs['context_words']:
    for word in words:
        word_freq[word] += 1

# 这里应该过滤高频词汇，因为低频词汇在别的地方不一样
high_freq_words = set(word for word in word_freq.keys() if word_freq[word] > 15)

# 删除低频词汇
for words in steps_logs['context_words']:
    # 补集
    deleted_words = set(words) - high_freq_words
    for word in deleted_words:
        words.remove(word)

# 最长序列
max_length = steps_logs['context_words'].map(len).max()


# 补全
def padding(x: list):
    x.extend(['None'] * (max_length - len(x)))
    return x


steps_logs['context_words'].map(padding)

oe = OrdinalEncoder()
words_vector = np.array(steps_logs['context_words'].values.tolist())
words_label_vector = oe.fit_transform(words_vector)



scores = []
for i in range(20, 60):
    kmeans = KMeans(n_clusters=i)
    y_pred = kmeans.fit_predict(words_label_vector)
    scores.append(silhouette_score(words_label_vector, y_pred))
pd.DataFrame(scores).plot.line()

kmeans = KMeans(n_clusters=30)
ans = kmeans.fit_predict(words_label_vector)
for i in set(ans):
    print(i)
    print(words_vector[ans==i][:8])
'''
