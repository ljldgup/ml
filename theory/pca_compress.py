from PIL import Image
import numpy as np
from sklearn.decomposition import PCA


def loadImage(path):
    img = Image.open(path)
    img.show()
    # 将图像转换成灰度图
    img = img.convert("L")

    # 图像的大小在size中是（宽，高）
    # 所以width取size的第一个值，height取第二个
    width = img.size[0]
    height = img.size[1]
    data = img.getdata()
    # 直接将3位rpg放在行上，尝试做彩色压缩
    data = np.array(data).reshape(height, -1) / 255
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
    由于特征向量模为1，彼此点积得0，所以特征向量方阵的逆矩阵即为他的转置，
    当K小于数据的特征数时，补0乘以完整逆矩阵，和直接乘以前k各特征向量的转置效果小童
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
    data = loadImage("tun.jpg")

    data = data.reshape(data.shape[0], -1)
    recdata = pca(data, 20)
    recdata = sklearn_pca(data, 20)
    '''
    recdata0 = sklearn_pca(data[:, :, 0], 80)
    recdata1 = sklearn_pca(data[:, :, 1], 80)
    recdata2 = sklearn_pca(data[:, :, 2], 80)
    recdata = np.concatenate([recdata0[:,:,np.newaxis], recdata1[:,:,np.newaxis], recdata2[:,:,np.newaxis]],axis=2)
    '''
    # 计算误差
    # error(data, recdata)
    # 使用彩色，返回值有复数数，取实部还是模，图像都有缺失, sklearn同样如此
    # 即使每个都分开压缩，最后仍然有损失，  进一步发现有些图，即使用灰度图也会有损失
    recdata = recdata.reshape(data.shape[0], -1)
    recdata = (np.real(recdata) * 255).astype(np.uint8)
    newImg = Image.fromarray(recdata)
    newImg.show()
    # error(data, recdata)
