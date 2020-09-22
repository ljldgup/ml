from datetime import date, datetime
from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
# col 会报错，但实际上是能生效的
from pyspark.sql.functions import udf, array_contains, col
from pyspark.sql.pandas.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructField, StringType, ArrayType, DateType, TimestampType

import pandas as pd


# 将日志信息转化位表中一行
def log_to_row(line):
    if line.startswith('FileServer'):
        return 'start', 0, 0, 'start', 'start'
    attr0 = line.split(' : ')
    if len(attr0) < 2:
        return 'blank', -1, -1, 'blank', 'blank'

    attr1 = attr0[0].split('/')
    # 时间，线程，操作，日志等及, 具体日志
    # attr1[6]具体日志，在实际操作中应该变为类型
    return attr1[0], int(attr1[1]), int(attr1[2]), attr1[3], attr0[1]


appName = 'fms_log_test'
master = 'local'
spark = SparkSession \
    .builder \
    .appName("appName") \
    .getOrCreate()
sc = spark.sparkContext

############################################################
# 读取文件转表
lines = sc.textFile("fms_log.txt")
info = lines.map(log_to_row)

df = spark.createDataFrame(
    info, "date_str string, thread_num int, operation int, level string, context string")


#####################################################################
# 对日志内容进行编码
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

# 这里发现str_2_date 输入已经是datetime 可能spark 进行了处理
# df = df.withColumn('date', str_2_date(df['date_str']))
df = df.withColumn('min_str', str_2_date(df['date_str']))

# 每分钟新开资源统计
df.filter(df['one_context'].startswith('Begin')).groupby(df['min_str']).count().show(10, False)


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
    agg(collect_context(df['context']), start_time(df['date_str']))

# withColumnRenamed列重命名
thread_process = thread_process.withColumnRenamed('collect_context(context)', 'c_context'). \
    withColumnRenamed('start_time(date)', 'start_time')


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
thread_process = thread_process.filter(~ thread_process['c_context'].endswith('Open,Contexts -> ')) \
    # csv 无法保存 arrayType
# .withColumn('l_context', context_split(thread_process['c_context']))

# 使用fpGrowth寻找频繁项
fpGrowth = FPGrowth(itemsCol="l_context", minSupport=0.5, minConfidence=0.6)
model = fpGrowth.fit(thread_process.withColumn('l_context', context_split(thread_process['c_context'])))

get_last_item = udf(lambda x: x[-1], StringType())

# 通过array_contains 筛选，col('items')代表自己当前这一列
model.freqItemsets.filter(array_contains(col('items'), 'crash')).show(truncate=False)

# 通过将items映射成string列，然后筛选，这样分拆过程后很明显更简单
model.freqItemsets.withColumn('last_item',get_last_item(col('items'))).filter(col('last_item').endswith('crash')).show(10)

# thread_process.write.mode('Overwrite').format('csv').save('fms')
