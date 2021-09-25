from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType

spark = SparkSession \
    .builder \
    .appName("Python mllib example") \
    .getOrCreate()

lines = spark.read.text("sample_movielens_ratings.txt").rdd
parts = lines.map(lambda row: row.value.split("::"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2])

# coldStartStrategy冷启动策略，缺乏评分信息项损失计算的策略，这里是丢弃，避免得到NaN
# Am×n=Um×k × Vk×n Um×k表示用户对隐藏特征的偏好，Vk×n表示产品包含隐藏特征的程度。A是打分矩阵
# ALS随机初始化U,V 然后轮流使用优化U,V
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

# 这里是对电影打分的预测，ALS 训练的是打分矩阵 ratings matrix `R`的低阶分解形式， 可以对未打分项进行打分

predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# 基于用户和基于物品的推荐
userRecs = model.recommendForAllUsers(10)
movieRecs = model.recommendForAllItems(10)

# 前三个用户id，推荐前10部电影
users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
# 前三个电影id，推荐前10个用户
movies = ratings.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 10)
