""" Multiple Regression/Multi target Regression Example

The input features have 10 dimensions, and target features are 2 dimension.

"""

from __future__ import absolute_import, division, print_function

import tflearn
import numpy as np

# Regression data- 10 training instances
#10 input features per instance.
X=np.random.rand(10,10).tolist()
#2 output features per instance
Y=np.random.rand(10,2).tolist()

# Multiple Regression graph, 10-d input layer
input_ = tflearn.input_data(shape=[None,10])
#10-d fully connected layer
r1 = tflearn.fully_connected(input_,10)
#2-d fully connected layer for output
r1 = tflearn.fully_connected(r1,2)
r1 = tflearn.regression(r1, optimizer='sgd', loss='mean_square',
                                        metric='R2', learning_rate=0.01)

m = tflearn.DNN(r1)
m.fit(X,Y, n_epoch=100, show_metric=True, snapshot_epoch=False)

#Predict for 1 instance
testinstance=np.random.rand(1,10).tolist()
print("\nInput features:  ",testinstance)
print("\n Predicted output: ")
print(m.predict(testinstance))
