from pyspark import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation, ChiSquareTest, Summarizer
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python mllib basic example") \
    .getOrCreate()
sc = spark.sparkContext

# 这里有稀疏矩阵和密集矩阵
data = [(Vectors.sparse(4, [(0, 1.0), (3, -2.0)]),),
        (Vectors.dense([4.0, 5.0, 0.0, 3.0]),),
        (Vectors.dense([6.0, 7.0, 0.0, 8.0]),),
        (Vectors.sparse(4, [(0, 9.0), (3, 1.0)]),)]
df = spark.createDataFrame(data, ["features"])

r1 = Correlation.corr(df, "features").head()
print("Pearson correlation matrix:\n" + str(r1[0]))

r2 = Correlation.corr(df, "features", "spearman").head()
print("Spearman correlation matrix:\n" + str(r2[0]))

#############################################################################
# 假设检验
data = [(0.0, Vectors.dense(0.5, 10.0)),
        (0.0, Vectors.dense(1.5, 20.0)),
        (1.0, Vectors.dense(1.5, 30.0)),
        (0.0, Vectors.dense(3.5, 30.0)),
        (0.0, Vectors.dense(3.5, 40.0)),
        (1.0, Vectors.dense(3.5, 40.0))]
df = spark.createDataFrame(data, ["label", "features"])

# 卡方检验，比较两个及两个以上样本率( 构成比）以及两个分类变量的关联性分析
# 其根本思想就是在于比较理论频数和实际频数的吻合程度或拟合优度问题。
r = ChiSquareTest.test(df, "features", "label").head()

# 这里输出的都是两项，代表两列features与label之间相关性假设检验
print("pValues: " + str(r.pValues))
print("degreesOfFreedom: " + str(r.degreesOfFreedom))
print("statistics: " + str(r.statistics))

df = sc.parallelize([Row(weight=1.0, features=Vectors.dense(1.0, 1.0, 1.0)),
                     Row(weight=0.0, features=Vectors.dense(1.0, 2.0, 3.0))]).toDF()

summarizer = Summarizer.metrics("mean", "count")

# 乘以weight之后的均值，计数
df.select(summarizer.summary(df.features, df.weight)).show(truncate=False)
df.select(summarizer.summary(df.features)).show(truncate=False)

df.select(Summarizer.mean(df.features, df.weight)).show(truncate=False)
df.select(Summarizer.mean(df.features)).show(truncate=False)
