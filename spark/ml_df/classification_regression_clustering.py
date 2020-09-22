from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, SparkSession
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
'''

    spark.mllib 包含基于RDD的原始算法API。
    spark.ml 则提供了基于DataFrames 高层次的API，可以用来构建机器学习管道。

'''
spark = SparkSession \
    .builder \
    .appName("Python mllib example") \
    .getOrCreate()
sc = spark.sparkContext

# Load and parse the data file, converting it to a DataFrame.
data = spark.read.format("libsvm").load("../data/mllib/sample_libsvm_data.txt")

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# 将分类少于4的特征变换成标签，分类多于4的看做连续
featureIndexer = \
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# 随机分割成7:3 训练测试集
(trainingData, testData) = data.randomSplit([0.7, 0.3])

rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# 将label转index 最后输出转回label
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

model = pipeline.fit(trainingData)

# 貌似输出预测数据都用transform，官方文档未见到使用predict
predictions = model.transform(testData)
# 加False显示完整，不截断
predictions.select("predictedLabel", "label", "features").show(5, False)

# 损失评估在spark中通常使用xxxEvaluator,标明predictionCol
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)

'''
　　　　1）LDA是有监督的降维方法，而PCA是无监督的降维方法
　　　　2）LDA降维最多降到类别数k-1的维数，而PCA没有这个限制。
　　　　3）LDA除了可以用于降维，还可以用于分类。
　　　　4）LDA选择分类性能最好的投影方向，而PCA选择样本点投影具有最大方差的方向。
'''
from pyspark.ml.clustering import LDA

# Loads data.
dataset = spark.read.format("libsvm").load("../data/mllib/sample_lda_libsvm_data.txt")

# Trains a LDA model.
lda = LDA(k=10, maxIter=10)
model = lda.fit(dataset)

ll = model.logLikelihood(dataset)
# perplexity是一种信息理论的测量方法，b的perplexity值定义为基于b的熵的能量（b可以是一个概率分布，或者概率模型），通常用于概率模型的比较
lp = model.logPerplexity(dataset)
print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
print("The upper bound on perplexity: " + str(lp))

# Describe topics.
topics = model.describeTopics(3)
print("The topics described by their top-weighted terms:")
topics.show(truncate=False)

# Shows the result
transformed = model.transform(dataset)
transformed.show(truncate=False)

# 生存分析
from pyspark.ml.regression import AFTSurvivalRegression
from pyspark.ml.linalg import Vectors

training = spark.createDataFrame([
    (1.218, 1.0, Vectors.dense(1.560, -0.605)),
    (2.949, 0.0, Vectors.dense(0.346, 2.158)),
    (3.627, 0.0, Vectors.dense(1.380, 0.231)),
    (0.273, 1.0, Vectors.dense(0.520, 1.151)),
    (4.199, 0.0, Vectors.dense(0.795, -0.226))], ["label", "censor", "features"])
quantileProbabilities = [0.3, 0.6]
aft = AFTSurvivalRegression(quantileProbabilities=quantileProbabilities,
                            quantilesCol="quantiles")

model = aft.fit(training)

# Print the coefficients, intercept and scale parameter for AFT survival regression
print("Coefficients: " + str(model.coefficients))
print("Intercept: " + str(model.intercept))
print("Scale: " + str(model.scale))
model.transform(training).show(truncate=False)

# 分解机
from pyspark.ml.classification import FMClassifier

training = spark.read.format("libsvm").load("../data/mllib/sample_libsvm_data.txt")
cls = FMClassifier(maxIter=10, regParam=0.3, factorSize=16)

fmModel = cls.fit(training)
