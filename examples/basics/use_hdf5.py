# -*- coding: utf-8 -*-
"""
Example on how to use HDF5 dataset with TFLearn. HDF5 is a data model,
library, and file format for storing and managing data. It can handle large
dataset that could not fit totally in ram memory. Note that this example
just give a quick compatibility demonstration. In practice, there is no so
real need to use HDF5 for small dataset such as CIFAR-10.
"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.data_utils import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

# CIFAR-10 Dataset
from tflearn.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data()
Y = to_categorical(Y)
Y_test = to_categorical(Y_test)

# Create a hdf5 dataset from CIFAR-10 numpy array
import h5py
h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('cifar10_X', data=X)
h5f.create_dataset('cifar10_Y', data=Y)
h5f.create_dataset('cifar10_X_test', data=X_test)
h5f.create_dataset('cifar10_Y_test', data=Y_test)
h5f.close()

# Load hdf5 dataset
h5f = h5py.File('data.h5', 'r')
X = h5f['cifar10_X']
Y = h5f['cifar10_Y']
X_test = h5f['cifar10_X_test']
Y_test = h5f['cifar10_Y_test']

# Build network
network = input_data(shape=[None, 32, 32, 3], dtype=tf.float32)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
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

h5f.close()
