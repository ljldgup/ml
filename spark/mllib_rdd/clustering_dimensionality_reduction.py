from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.clustering import StreamingKMeans
from pyspark.streaming import StreamingContext

sc = SparkContext("local[2]", "test")
ssc = StreamingContext(sc, 1)


############################################################################
# stream k-means对流式数据进行聚类，并测试
# spark内部采用指数加权的形式对质心进行重新分配，decayFactor是旧质心位置衰减系数
def parse(lp):
    label = float(lp[lp.find('(') + 1: lp.find(')')])
    vec = Vectors.dense(lp[lp.find('[') + 1: lp.find(']')].split(','))

    return LabeledPoint(label, vec)


trainingData = sc.textFile("kmeans_data.txt") \
    .map(lambda line: Vectors.dense([float(x) for x in line.strip().split(' ')]))

testingData = sc.textFile("streaming_kmeans_data_test.txt").map(parse)

trainingQueue = [trainingData]
testingQueue = [testingData]

trainingStream = ssc.queueStream(trainingQueue)
testingStream = ssc.queueStream(testingQueue)

# spark内部采用指数加权的形式对质心进行重新分配，decayFactor是旧质心位置衰减系数
model = StreamingKMeans(k=2, decayFactor=1.0).setRandomCenters(3, 1.0, 0)

model.trainOn(trainingStream)
# 打印出来的是test集的结果
result = model.predictOnValues(testingStream.map(lambda lp: (lp.label, lp.features)))
result.pprint()

ssc.start()
ssc.stop(stopSparkContext=True, stopGraceFully=True)

# ####################################################################
# # svd分解
# from pyspark.mllib.linalg import Vectors
# from pyspark.mllib.linalg.distributed import RowMatrix
#
# rows = sc.parallelize([
#     Vectors.sparse(5, {1: 1.0, 3: 7.0}),
#     Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
#     Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
# ])
#
# mat = RowMatrix(rows)
#
# # Compute the top 5 singular values and corresponding singular vectors.
# svd = mat.computeSVD(5, computeU=True)
# U = svd.U  # The U factor is a RowMatrix.
# s = svd.s  # The singular values are stored in a local dense vector.
# V = svd.V  # The V factor is a local dense matrix.


