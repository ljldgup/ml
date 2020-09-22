
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python mllib example") \
    .getOrCreate()
sc = spark.sparkContext

# FPGrowth通过fp树寻找频繁项
from pyspark.ml.fpm import FPGrowth
df = spark.createDataFrame([
    (0, [1, 2, 5]),
    (1, [1, 2, 3, 5]),
    (2, [1, 2])
], ["id", "items"])

# minSupport 为百分比，乘以行数得到最小支持度，最小置信度minConfidence
fpGrowth = FPGrowth(itemsCol="items", minSupport=0.5, minConfidence=0.6)
model = fpGrowth.fit(df)

# 通过fp树统计的数据集出现的频繁程度
model.freqItemsets.show()

# antecedent -> consequent confidence置信度
model.associationRules.show()

# 预测
model.transform(df).show()

# PrefixSpan通过前缀和投影数据库查找频繁序列
from pyspark.ml.fpm import PrefixSpan
from pyspark import Row

# FPGrowth 统计的是频繁项， PrefixSpan 统计的是频繁序列， 这里每一行输入都是一个时间序列，使用parallelize创建不是createDataFrame
df = sc.parallelize([Row(sequence=[[1, 2], [3]]),
                     Row(sequence=[[1], [3, 2], [1, 2]]),
                     Row(sequence=[[1, 2], [5]]),
                     Row(sequence=[[6]])]).toDF()

# 阀值minSupport 0.5, 即最小4*0.5=2两次，maxLocalProjDBSize 最大单机投影数据库的项数
prefixSpan = PrefixSpan(minSupport=0.5, maxPatternLength=5,
                        maxLocalProjDBSize=32000000)

prefixSpan.findFrequentSequentialPatterns(df).show()
