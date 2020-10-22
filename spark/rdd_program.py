from pyspark import SparkContext, SparkConf

# 本地集群
appName = 'rdd_test'
master = 'local'
conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)
# rdd两种创建方式，直接喂数据结构，或者载入磁盘文件
# 接受socket，kafka的时dstream不是rdd，rdd时dstream的下层
data = [1, 2, 3, 4, 5]
distData = sc.parallelize(data)

distFile = sc.textFile("data.txt")
# 返回成集合
distData.collect()
distFile.collect()

rdd = sc.parallelize(range(1, 4)).map(lambda x: (x, "a" * x))
rdd.saveAsSequenceFile("rddSequenceFile")
sorted(sc.sequenceFile("rddSequenceFile").collect())

'''
conf = {"es.resource": "index/type"}  # assume Elasticsearch is running on localhost defaults
rdd = sc.newAPIHadoopRDD("org.elasticsearch.hadoop.mr.EsInputFormat",
                         "org.apache.hadoop.io.NullWritable",
                         "org.elasticsearch.hadoop.mr.LinkedMapWritable",
                         conf=conf)
rdd.first()
'''

lines = sc.textFile("resources/data.txt")
lineLengths = lines.map(lambda s: len(s))
# 由于缓求值上面的Transformations没有立即执行，在执行action型函数时才会执行
totalLength = lineLengths.reduce(lambda a, b: a + b)

# 保存中间结果到内存中,storageLevel=0,MEMORY_ONLY，这是最快的选项
lineLengths.persist(storageLevel=0)
lineLengths.persist().is_cached
lineLengths.cache()

# 最终会返回0,这种使用全局变量的写法无法生效
# spark是在集群的情况下运行的，使用全局变量，或者打印等会出错
# 无法保证数据前后操作顺序
counter = 0
rdd = sc.parallelize(data)

def increment_counter(x):
    global counter
    counter += x

rdd.foreach(increment_counter)

print("Counter value: ", counter)

# 一些操作
distFile.flatMap(lambda s:s.split(" ")).count()

# 只读，所以机器之间不需要传输，改变后其他机器不能读到
broadcastVar = sc.broadcast([1, 2, 3])
broadcastVar.value

# 全局累加器，可用来代替前面的全局变量写法
accum = sc.accumulator(0)
sc.parallelize([1, 2, 3, 4]).foreach(lambda x: accum.add(x))
accum.value