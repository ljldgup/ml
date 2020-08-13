from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split

# 对应流式的dateframe
from pyspark.sql.types import StructType

'''
# nc 127.0.0.1 9999 < readme.txt
nc -L -p 9999  < readme.txt
./bin/spark-submit examples/src/main/python/sql/streaming/structured_network_wordcount.py localhost 9999

'''


def test1():
    spark = SparkSession \
        .builder \
        .appName("StructuredNetworkWordCount") \
        .getOrCreate()

    # Create DataFrame representing the stream of input lines from connection to localhost:9999
    lines = spark \
        .readStream \
        .format("socket") \
        .option("host", "localhost") \
        .option("port", 9999) \
        .load()
    '''
    lines.writeStream \
        .format("console") \
        .start()

    lines.awaitTermination()
    '''
    # explode 由行返回一列，alias设置列名word，直接输出words可以看到
    words = lines.select(
        explode(
            split(lines.value, " ")
        ).alias("word")
    )

    # 聚合然后按照count排序
    wordCounts = words.groupBy("word").count().sort('count', ascending=False)

    # query 可以看做一个查询结果
    query = wordCounts \
        .writeStream \
        .outputMode("complete") \
        .format("console") \
        .start()



    '''
    query = words \
        .writeStream \
        .format("console") \
        .trigger(once=True) \
        .start()
    '''

    query.awaitTermination()


def test2():
    spark = SparkSession \
        .builder \
        .appName("StructuredNetworkWordCount") \
        .getOrCreate()

    userSchema = StructType().add("name", "string").add("age", "integer")
    # Read text from socket
    socketDF = spark \
        .readStream \
        .format("socket") \
        .option("host", "localhost") \
        .option("port", 9999) \
        .load()

    socketDF.isStreaming()  # Returns True for DataFrames that have streaming sources

    socketDF.printSchema()
    # Read all the csv files written atomically in a directory
    # socket不支持预定义schema
    csvDF = spark \
        .readStream \
        .option("sep", ";") \
        .schema(userSchema) \
        .csv("/path/to/directory")  # Equivalent to format("csv").load("/path/to/directory")


'''

words = ...  # streaming DataFrame of schema { timestamp: Timestamp, word: String }

# Group the data by window and word and compute the count of each group
windowedCounts = words \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        window(words.timestamp, "10 minutes", "5 minutes"),
        words.word) \
    .count()


spark \
  .readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", "host1:port1,host2:port2") \
  .option("subscribe", "topic1") \
  .load() \
  .selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)") \
  .writeStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", "host1:port1,host2:port2") \
  .option("topic", "topic1") \
  .trigger(continuous="1 second") \     # only change in query
  .start()


'''
if __name__ == '__main__':
    test1()
