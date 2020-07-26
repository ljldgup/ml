from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans
from sklearn.datasets import make_classification, fetch_openml, make_moons, make_blobs
from matplotlib.image import imread  # or `from imageio import imread`
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_digits

import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

X, y = make_classification(n_classes=4,
    n_samples=300, n_features=2,
    n_redundant=0, n_informative=2,
    random_state=22, n_clusters_per_class=1,
    scale=100)
silhouette_scores = []
inertias = []
for i in range(2, 15):
    kmeans = KMeans(n_clusters=i)
    y_pred = kmeans.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    inertias.append(kmeans.inertia_)
# inertial趋于平缓的拐点最好
plt.plot(inertias, c='red')


plt.clf()
#silhouette_scores大效果好
plt.plot(silhouette_scores, c='purple')

# 图片像素分割
image = imread('timg.jpg')
X = image.reshape(-1, 3)

'''
fig, ax = plt.subplots(
    nrows=2,
    ncols=2,
    sharex=True,
    sharey=True, )

for i in range(4):
    kmeans = KMeans(n_clusters=i * 3 + 2).fit(X)
    # 这里几个质心代替了所有像素
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_img = segmented_img.reshape(image.shape)
    ax[i // 2][i % 2].imshow(segmented_img / 255)
    ax[i // 2][i % 2].set_title('{} color'.format(i * 3 + 2))
plt.tight_layout()
plt.show()
'''

# MiniBatchKMeans 明显要比KMeans快的多
fig, ax = plt.subplots(
    nrows=2,
    ncols=2,
    sharex=True,
    sharey=True, )

for i in range(4):
    minibatch_kmeans = MiniBatchKMeans(n_clusters=i * 3 + 2)
    minibatch_kmeans.fit(X)
    # 这里几个质心代替了所有像素
    segmented_img = minibatch_kmeans.cluster_centers_[minibatch_kmeans.labels_]
    segmented_img = segmented_img.reshape(image.shape)
    ax[i // 2][i % 2].imshow(segmented_img / 255)
    ax[i // 2][i % 2].set_title('{} color'.format(i * 3 + 2))
plt.tight_layout()
plt.show()

'''
X_digits, y_digits = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)
# 使用kmeans完成集成模型
pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=50)),
    ("log_reg", LogisticRegression()),
])
pipeline.fit(X_train, y_train)

param_grid = dict(kmeans__n_clusters=range(2, 100))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)
'''

'''
# 半监督学习
mnist = fetch_openml('mnist_784', version=1)
print(mnist.keys())
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
k = 50
kmeans = KMeans(n_clusters=k)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]
log_reg = LogisticRegression()
# y_representative_digits 是手工编辑的
# 先进性kmeans聚类，再将聚类结果预测为所需分类
log_reg.fit(X_representative_digits, y_representative_digits)
log_reg.score(X_test, y_test)
'''

X, y = make_moons(n_samples=1000, noise=0.05)
dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)
dbscan.labels_
len(dbscan.core_sample_indices_)
dbscan.core_sample_indices_
dbscan.components_

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])
X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
knn.predict(X_new)
knn.predict_proba(X_new)

# 读取数据，绘制图像
x, y = make_blobs(n_samples=100, n_features=2, centers=6)
print(x.shape)

# 基于Agglomerativeclustering完成聚类
model = AgglomerativeClustering(n_clusters=4)
pred_y = model.fit_predict(x)
print(pred_y)

# 画图显示样本数据
plt.figure('Agglomerativeclustering', facecolor='lightgray')
plt.title('Agglomerativeclustering', fontsize=16)
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.tick_params(labelsize=10)
plt.scatter(x[:, 0], x[:, 1], s=80, c=pred_y, cmap='brg', label='Samples')
plt.legend()
plt.show()

kneighbors_graph()