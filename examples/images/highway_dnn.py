# -*- coding: utf-8 -*-

""" Deep Neural Network for MNIST dataset classification task using 
a highway network

References:

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
    [https://arxiv.org/abs/1505.00387](https://arxiv.org/abs/1505.00387)

"""
from __future__ import division, print_function, absolute_import

import tflearn

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)

# Building deep neural network
input_layer = tflearn.input_data(shape=[None, 784])
dense1 = tflearn.fully_connected(input_layer, 64, activation='elu',
                                 regularizer='L2', weight_decay=0.001)
                 
                 
#install a deep network of highway layers
highway = dense1                              
for i in range(10):
    highway = tflearn.highway(highway, 64, activation='elu',
                              regularizer='L2', weight_decay=0.001, transform_dropout=0.8)
                              
                              
softmax = tflearn.fully_connected(highway, 10, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=20, validation_set=(testX, testY),
          show_metric=True, run_id="highway_dense_model")
