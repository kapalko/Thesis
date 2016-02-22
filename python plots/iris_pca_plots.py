print(__doc__)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA


iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))


plt.subplot(1, 2, 1)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], c=c, label=target_name)
# plt.legend()
plt.title('IRIS dataset')
plt.xlabel('Sepal length', fontsize=14)
plt.ylabel('Sepal width', fontsize=14)
plt.xticks(())
plt.yticks(())

plt.subplot(1, 2, 2)
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
# plt.legend()
plt.title('PCA of IRIS dataset')
plt.xticks(())
plt.yticks(())
plt.xlabel('PC 1', fontsize=14)
plt.ylabel('PC 2', fontsize=14)

plt.show()