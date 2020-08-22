from pyspark import Row
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext

spark = SparkSession \
    .builder \
    .appName("Python mllib example") \
    .getOrCreate()

sc = spark.sparkContext
ssc = StreamingContext(sc, 5)

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

assembler = VectorAssembler(inputCols=['X1', 'X2'], outputCol="features")


def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    # 最后一Y是label， 其余是feature
    return LabeledPoint(values[-1], values[:-1])


# 不能直接用dstream训练，使用foreachrdd传入函数进行训练，
def train_model(rdd):
    # dstream不会判断rdd是否为空
    if not rdd.isEmpty():
        assembler = VectorAssembler(inputCols=['X1', 'X2'], outputCol="features")
        df = spark.createDataFrame(rdd)
        training = assembler.transform(df)
        training.show(truncate=False)
        model = lr.fit(training)
        prediction = model.transform(training)
        prediction.select('label', 'features', 'prediction').show(truncate=False)


nums = ssc.socketTextStream("localhost", 9999)
# nums = nums.map(parsePoint)
nums = nums.map(lambda line: line.split(" "))

# 这里自己转换不行，需要用LabeledPoint
xy = nums.map(lambda p: Row(X1=float(p[0]), X2=float(p[1]),
                            label=float(p[2])))

# nums = nums.map(lambda nums: [int(num) for num in nums])
# xy.pprint()
xy.pprint()
xy.foreachRDD(train_model)

ssc.start()  # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate
