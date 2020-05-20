from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, shutil
import numpy as np


# 重新生成记得删除base_dir
def copy_files(base_dir, original_dataset_dir, categorys=['cats', 'dogs']):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
        for item in zip(['train', 'validation', 'test'], [range(0, 1000), range(1000, 1500), range(1500, 2000)]):
            # 创建训练，验证，测试集目录
            first_dir = os.path.join(base_dir, item[0])
            os.mkdir(first_dir)

            for category in categorys:
                # 创建对应类别目录
                second_dir = os.path.join(first_dir, category)
                os.mkdir(second_dir)

                # 拷贝图片到对应的目录'train', 'validation', 'test' 分别 0-999，1000-1499，1500-1999
                fnames = ['{}.{}.jpg'.format(category[:-1], i) for i in item[1]]
                for fname in fnames:
                    src = os.path.join(original_dataset_dir, fname)
                    dst = os.path.join(second_dir, fname)
                    shutil.copyfile(src, dst)


# 能够将硬盘上的图像文件自动转换为预处理好的张量批量

def get_image_generator(base_dir, enchenced=False):
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    if enchenced:
        # 数据增强
        # 只能用于训练集合
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True, )
    else:
        train_datagen = ImageDataGenerator(rescale=1. / 255)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    return [train_generator, validation_generator, test_generator]
