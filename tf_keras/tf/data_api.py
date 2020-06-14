import tensorflow as tf

X = tf.range(10)  # any data tensor
dataset = tf.data.Dataset.from_tensor_slices(X)

for item in dataset:
    print(item)

# batch 大小为7， 重复遍历三次数据
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)

# 函数式操作
dataset = dataset.map(lambda x: x * 2)
dataset = dataset.apply(tf.data.experimental.unbatch())
dataset = dataset.filter(lambda x: x < 10)

# 打乱数据
dataset = tf.data.Dataset.range(10).repeat(3)  # 0 to 9, three times
dataset = dataset.shuffle(buffer_size=5, seed=42).batch(7)
for item in dataset:
    print(item)

train_filepaths = ['1.csv', '2.csv']
filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)
n_readers = 2
dataset = filepath_dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1), cycle_length=n_readers)
for line in dataset.take(5):
    print(line.numpy())

n_inputs = 8
X_mean, X_std = 0, 0


def preprocess(line):
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return (x - X_mean) / X_std, y


def csv_reader_dataset(filepaths, repeat=1, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.shuffle(shuffle_buffer_size).repeat(repeat)
    return dataset.batch(batch_size).prefetch(1)


# onehot 编码
vocab = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
indices = tf.range(len(vocab), dtype=tf.int64)
table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
num_oov_buckets = 2
table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)

categories = tf.constant(["NEAR BAY", "DESERT", "INLAND", "INLAND"])
cat_indices = table.lookup(categories)
cat_one_hot = tf.one_hot(cat_indices, depth=len(vocab) + num_oov_buckets)

embedding_dim = 2
embed_init = tf.random.uniform([len(vocab) + num_oov_buckets, embedding_dim])
embedding_matrix = tf.Variable(embed_init)
categories = tf.constant(["NEAR BAY", "DESERT", "INLAND", "INLAND"])
cat_indices = table.lookup(categories)
tf.nn.embedding_lookup(embedding_matrix, cat_indices)
