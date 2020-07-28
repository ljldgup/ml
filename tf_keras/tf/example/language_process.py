from tensorflow import keras
import numpy as np
import tensorflow as tf

filepath = 'test.txt'
with open(filepath, encoding='utf-8') as f:
    shakespeare_text = f.read()

# 这里char_level对中文也有效
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([shakespeare_text])

tokenizer.texts_to_sequences(["First"])
tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]])
# tokenizer.word_index
max_id = len(tokenizer.word_index)  # number of distinct characters
dataset_size = tokenizer.document_count  # total number of characters

# 这样能使np.array降低用一个维度
[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1

train_size = dataset_size * 90 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

# 直接使用dataset的窗口，相比使用numpy生成，节省一些内存
n_steps = 100
window_length = n_steps + 1  # target = input shifted 1 character ahead
dataset = dataset.window(window_length, shift=1, drop_remainder=True)

# window返回的是dateset，使用flat_map压平后使用
dataset = dataset.flat_map(lambda window: window.batch(window_length))

batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)
# 这里是input  [:-1] 和 target  [1:]
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))

dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))

dataset = dataset.prefetch(1)

model = keras.models.Sequential([
    # 第一个dropout是x和hidden之间的dropout,第二个是hidden-hidden之间的dropout
    keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id], dropout=0.2, recurrent_dropout=0.2),
    keras.layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation="softmax"))
])
# 因为输出的是onehot，所以这里采用稀疏交叉熵损失
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
history = model.fit(dataset, epochs=20)


def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, max_id)

# rnn可输入任意长度的字符串，时间步之间使用的是同一组参数
X_new = preprocess(["How are yo"])
Y_pred = model.predict_classes(X_new)
tokenizer.sequences_to_texts(Y_pred + 1)[0][-1]

def next_char(text, temperature=1):
    X_new = preprocess([text])
    y_proba = model.predict(X_new)[0, -1:, :]
    # 这里使用的是对数概率，原因不明
    rescaled_logits = tf.math.log(y_proba) / temperature

    # 根据对数分布，进行一次采样，注意这里加了1
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]



def complete_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text

print(complete_text("t", temperature=0.2))
print(complete_text("w", temperature=1))
print(complete_text("w", temperature=2))