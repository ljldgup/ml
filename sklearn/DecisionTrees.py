from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.tree import DecisionTreeRegressor

iris = load_iris()
X = iris.data[:, 2:]  # petal length and width
y = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

with open('./wine.dot', 'w', encoding='utf-8') as f:
    export_graphviz(
        tree_clf,
        out_file=f,
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)
