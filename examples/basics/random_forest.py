""" Random Forest example. """

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.estimators import RandomForestClassifier

# Data loading and pre-processing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=False)

m = RandomForestClassifier(n_estimators=100, max_nodes=1000)
m.fit(X, Y, batch_size=10000, display_step=10)

print("Compute the accuracy on train set:")
print(m.evaluate(X, Y, tflearn.accuracy_op))

print("Compute the accuracy on test set:")
print(m.evaluate(testX, testY, tflearn.accuracy_op))

print("Digits for test images id 0 to 5:")
print(m.predict(testX[:5]))
print("True digits:")
print(testY[:5])
