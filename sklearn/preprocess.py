from sklearn import preprocessing
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.impute import SimpleImputer

#填补缺失值
#imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit([[1, 2], [np.nan, 3], [7, 6]])
SimpleImputer()
X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))


#标准化:将特征数据的分布调整成标准正太分布，也叫高斯分布，也就是使得数据的均值维0，方差为1.
iris = datasets.load_iris()
X = preprocessing.scale(iris.data)

#axis=0 每列平均， axis=0 每行平均
#每列均值接近于0，方差为1，标准正太分布。。
print(X.mean(axis=0))
print(X.mean(axis=1))
print(X.std(axis=0))


#分类器encoder
#独热码，在英文文献中称做 one-hot code, 直观来说就是有多少个状态就有多少比特，而且只有一个比特为1，其他全为0的一种码制。
#如红色：1 0 0 ，黄色: 0 1 0，蓝色：0 0 1 。如此一来每两个向量之间的距离都是根号2，在向量空间距离都相等，所以这样不会出现偏序性，基本不会影响基于向量空间度量算法的效果。
enc = OneHotEncoder(handle_unknown='ignore')
X = [['Male', 1], ['Female', 3], ['Female', 2]]
OneHotEncoder(handle_unknown='ignore')
enc.fit(X)
enc.categories_
enc.transform([['Female', 1], ['Male', 4]]).toarray()

#标签分类
le = preprocessing.LabelEncoder()
le.fit(["paris", "seattle", "tokyo", "amsterdam"])
# Transform Categories Into Integers
print(le.transform(["paris", "paris", "tokyo", "amsterdam"]))
# Transform Integers Into Categories
print(le.inverse_transform([1, 2, 0, 2]))
