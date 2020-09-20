from PIL import Image
import numpy as np
from sklearn.decomposition import PCA


def loadImage(path):
    img = Image.open(path)
    img.show()
    # 将图像转换成灰度图
    # img = img.convert("L")

    # 图像的大小在size中是（宽，高）
    # 所以width取size的第一个值，height取第二个
    width = img.size[0]
    height = img.size[1]
    data = img.getdata()
    # 直接将3位rpg放在行上，尝试做彩色压缩
    data = np.array(data).reshape(height, width * 3) / 255
    return data


def pca(data, k):
    n_samples, n_features = data.shape
    # 求均值
    mean = np.array([np.mean(data[:, i]) for i in range(n_features)])
    # 去中心化
    normal_data = data - mean
    # 得到协方差矩阵
    matrix_ = np.dot(np.transpose(normal_data), normal_data)
    # 有时会出现复数特征值，导致无法继续计算

    eig_val, eig_vec = np.linalg.eig(matrix_)
    # 第二种求前k个向量，取得
    eigIndex = np.argsort(eig_val)
    eigVecIndex = eigIndex[:-(k + 1):-1]
    feature = eig_vec[:, eigVecIndex]
    new_data = np.dot(normal_data, feature)
    # 将降维后的数据映射回原空间
    '''
    Y = P * X ==>   P(-1) * Y = P(-1) * P * X,  P（-1）是P的逆矩阵, 即 P(-1) * P = 1
    ==>   P(-1) * Y = X
    由于特征向量模为1，彼此点积得0，所以实际上乘以P的转置PT即可，当K小于数据的特征数时，可以起到压缩作用，但无法还原完整的数据
    '''
    rec_data = np.dot(new_data, np.transpose(feature)) + mean
    return rec_data


def error(data, recdata):
    sum1 = 0
    sum2 = 0
    # 计算两幅图像之间的差值矩阵
    D_value = data - recdata
    # 计算两幅图像之间的误差率，即信息丢失率
    for i in range(data.shape[0]):
        sum1 += np.dot(data[i], data[i])
        sum2 += np.dot(D_value[i], D_value[i])
    error = sum2 / sum1
    print(sum2, sum1, error)


def sklearn_pca(data, k):
    pca = PCA(n_components=20).fit(data)
    # 降维
    x_new = pca.transform(data)
    # 还原降维后的数据到原空间
    recdata = pca.inverse_transform(x_new)
    return recdata


if __name__ == '__main__':
    # sklearn中直接用pca的fit和inverse_transform进行逆变换
    data = loadImage("timg.jpg")
    # 降维
    # recdata = pca(data, 40)
    recdata = sklearn_pca(data, 40)
    # 计算误差
    error(data, recdata)
    # 这里返回值有负数，所以直接娶了实部，最终图像图像有缺失, sklearn说明他也是取了实部
    recdata = recdata.reshape(recdata.shape[0], -1, 3)
    recdata = (np.real(recdata) * 255).astype(np.uint8)
    newImg = Image.fromarray(recdata)
    newImg.show()
    # error(data, recdata)
