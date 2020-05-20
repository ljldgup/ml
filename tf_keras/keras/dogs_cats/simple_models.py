import os
from tensorflow.keras import models, layers, optimizers

# pycharm不会将当前文件目录自动加入自己的sourse_path。右键make_directory as-->Sources Root将当前工作的文件夹加入source_path就可以了。
from tf_keras.keras import tools
from tf_keras.keras.dogs_cats import preprocess


def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    return model


if __name__ == '__main__':
    original_dataset_dir = r'D:\tmp\datesets\python_ml_data\data\kaggle_original_data'
    base_dir = r'D:\tmp\cats_and_dogs_small'

    preprocess.copy_files(base_dir, original_dataset_dir)

    model = build_model()
    model.summary()
    train_generator, validation_generator, test_generator = preprocess.get_image_generator(base_dir)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=50)

    tools.plot_loss(history.history)
    tools.plot_accuracy(history.history)
