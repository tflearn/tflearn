""" K-Means Example """

from __future__ import division, print_function, absolute_import

from tflearn.estimators import KMeans

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=False)

# K-Means training
m = KMeans(n_clusters=10, distance='squared_euclidean')
m.fit(X, display_step=10)

# Testing
print("Clusters center coordinates:")
print(m.cluster_centers_vars)

print("X[0] nearest cluster:")
print(m.labels_[0])

print("Predicting testX[0] nearest cluster:")
print(m.predict(testX[0]))

print("Transforming testX[0] to a cluster-distance space:")
print(m.transform(testX[0]))
