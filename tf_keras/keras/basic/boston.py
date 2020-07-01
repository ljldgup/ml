import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers

(train_data, train_targets), (test_data, test_targets) = tf.keras.datasets.boston_housing.load_data()

# 数据表征化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def k_fold_crossValidation(build_model_func, k, train_data, train_targets, num_epochs=30):
    all_mae_histories = []
    # //是取整，python中/会得到float
    num_val_samples = len(train_data) // k

    for i in range(k):
        print('processing fold #', i)
        # 第i折数据用于验证
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        # 其余数据拼接用于训练
        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)
        model = build_model_func()
        history = model.fit(partial_train_data, partial_train_targets,
                            validation_data=(val_data, val_targets),
                            epochs=num_epochs, batch_size=1, verbose=0)
        mae_history = history.history['val_mae']
        all_mae_histories.append(mae_history)

if __name__ == '__main__':


    k_fold_crossValidation(build_model, 4, train_data, train_targets)
    model = build_model()
    model.fit(train_data, train_targets,
              epochs=80, batch_size=16, verbose=0)
    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)