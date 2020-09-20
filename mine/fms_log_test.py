from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField


def log_to_struct(line):
    if line.startswith('FileServer'):
        return ('start', 0, 0, 'start', 'start')
    attr0 = line.split(' : ')
    if len(attr0) < 2:
        return ('blank', -1, -1, 'blank', 'blank')

    attr1 = attr0[0].split(' ')
    attr2 = attr1[4].split('/')
    # 时间，线程，操作，日志等及, 具体日志
    # attr1[6]具体日志，在实际操作中应该变为类型
    return (' '.join([attr2[0], attr1[1], attr1[2], attr1[3]]),
            int(attr2[1]), int(attr2[2]), attr2[3], attr0[1])


appName = 'fms_log_test'
master = 'local'
spark = SparkSession \
    .builder \
    .appName("appName") \
    .getOrCreate()
sc = spark.sparkContext

lines = sc.textFile("fms_log.txt")
info = lines.map(log_to_struct)

df = spark.createDataFrame(
    info, "date_str string, thread_num int, operation int, level string, context string")

# 注意select根据条件选择，列名就是条件，过滤要用filter，
df.filter(df['thread_num'] > 0).\
    groupBy(df['thread_num']).count().write.mode('Overwrite').format('csv').save('fms')
