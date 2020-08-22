from pyspark.sql import SparkSession
import os

os.environ["hadoop.home.dir"] = r"D:\tools\spark-3.0.0-bin-hadoop2.7\winutils"

spark = SparkSession \
    .builder \
    .appName("Python mllib example") \
    .getOrCreate()
sc = spark.sparkContext

from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint


# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])


# 模型训练和预测
# rdd使用train和predict
# dataframe使用fit 和 transform
data = sc.textFile("../data/mllib/sample_svm_data.txt")
# 注意这里要用LabeledPoint来生成feature列
parsedData = data.map(parsePoint)

# 使用LBFGS进行优化
model = LogisticRegressionWithLBFGS.train(parsedData)

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))

# Save and load model
model.save(sc, "pythonLogisticRegressionWithLBFGSModel")
sameModel = LogisticRegressionModel.load(sc, "pythonLogisticRegressionWithLBFGSModel")

###################################################################
# 随机森林
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

data = MLUtils.loadLibSVMFile(sc, '../data/mllib/sample_libsvm_data.txt')
(trainingData, testData) = data.randomSplit([0.7, 0.3])

#  Setting featureSubsetStrategy="auto" lets the algorithm choose.
# Maximum number of bins used for splitting features. 特征值的最大箱数，分割区间数
model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(testData.count())
print('Test Error = ' + str(testErr))
print('Learned classification forest model:')
print(model.toDebugString())

# Save and load model
# model.save(sc, "target/tmp/myRandomForestClassificationModel")
# sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestClassificationModel")
