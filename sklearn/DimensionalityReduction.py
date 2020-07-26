from sklearn.datasets import make_moons, fetch_openml, make_classification
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, LocallyLinearEmbedding

X, y = make_moons(n_samples=1000, noise=0.15)
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]
# W2 就是两个主成分系数
W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)

# 和c1=Vt.T[:, 0]的值一样
pca.components_.T[:, 0]

pca.explained_variance_ratio_

# 选择前95的相关指数
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)

# 通过inverse_transform降维度，注意PCA是无监督算法，不需要标签
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
pca = PCA(n_components=154)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)

# Randomized PCA
rnd_pca = PCA(n_components=154, svd_solver="randomized")
X_reduced = rnd_pca.fit_transform(X_train)

# 逐个batch进行计算
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)
X_reduced = inc_pca.transform(X_train)

filename = 't'
m, n = X.shape
# 将文件映射到内存
X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))
batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)

# 网格搜索最优参数
clf = Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression())
])
param_grid = [{
    "kpca__gamma": np.linspace(0.03, 0.05, 10),
    "kpca__kernel": ["rbf", "sigmoid"]
}]
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)


# 降维可视化
X, y = make_classification(
    n_samples=500, n_features=6, n_classes=4,
    n_redundant=0, n_informative=4,
    random_state=22, n_clusters_per_class=1,
    scale=100)

# n_components将为到2
tsne = TSNE(n_components=2, init='pca', random_state=0)
# 降维后的数据
X2D = tsne.fit_transform(X)

#数据标签
markers = ['v', 's', 'o', 'x']
encoder = LabelEncoder()
encoder.fit(markers)
y_markers = encoder.inverse_transform(y)
# plt.scatter(X2D[:, 0], X2D[:, 1], c=y, s=10,cmap="tab10")

for label in set(y):
    # s形状可输入与x,y同形状的矩阵，alpha透明度，marker形状
    plt.scatter(X2D[:, 0][y == label], X2D[:, 1][y == label], s=80, alpha=0.6, marker=markers[label])

#使用主成分进行降维度
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
X2D = rbf_pca.fit_transform(X)
for label in set(y):
    plt.scatter(X2D[:, 0][y == label], X2D[:, 1][y == label], s=80, alpha=0.6, marker=markers[label])

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)
X2D = lle.fit_transform(X)
plt.scatter(X2D[:, 0], X2D[:, 1], c=y, s=10,cmap="tab10")
'''
for label in set(y):
    plt.scatter(X2D[:, 0][y == label], X2D[:, 1][y == label], s=80, alpha=0.6, marker=markers[label])
'''