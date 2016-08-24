# -*- coding: utf-8 -*-
"""
Example on how to use Dask with TFLearn. Dask is a simple task scheduling
system that uses directed acyclic graphs (DAGs) of tasks to break up large
computations into many small ones. It can handle large dataset that could
not fit totally in ram memory. Note that this example just give a quick
compatibility demonstration. In practice, there is no so much need to use
Dask for small dataset such as CIFAR-10.
"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.data_utils import *
from tflearn.layers.estimator import *

# Load CIFAR-10 Dataset
from tflearn.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data()
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)

# Create DASK array using numpy arrays
# (Note that it can work with HDF5 Dataset too)
import dask.array as da
X = da.from_array(np.asarray(X), chunks=(1000, 1000, 1000, 1000))
Y = da.from_array(np.asarray(Y), chunks=(1000, 1000, 1000, 1000))
X_test = da.from_array(np.asarray(X_test), chunks=(1000, 1000, 1000, 1000))
Y_test = da.from_array(np.asarray(Y_test), chunks=(1000, 1000, 1000, 1000))

# Build network
network = input_data(shape=[None, 32, 32, 3])
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = dropout(network, 0.75)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = dropout(network, 0.5)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96, run_id='cifar10_cnn')
