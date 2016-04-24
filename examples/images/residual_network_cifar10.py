# -*- coding: utf-8 -*-

""" Deep Residual Network.

Applying a Deep Residual Network to CIFAR-10 Dataset classification task.
This implementation is different from the original paper, as a convolution
in TensorFlow can't yet have a kernel size < strides. Average pooling is
used instead for downsampling.

References:
    - K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image
      Recognition, 2015.
    - Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.

Links:
    - [Deep Residual Network](http://arxiv.org/pdf/1512.03385.pdf)
    - [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

"""

from __future__ import division, print_function, absolute_import

import tflearn
import tflearn.data_utils as du

# Data loading
from tflearn.datasets import cifar10
(X, Y), (testX, testY) = cifar10.load_data()
# Data pre-processing
X, mean = du.featurewise_zero_center(X)
X, std = du.featurewise_std_normalization(X)
testX = du.featurewise_zero_center(testX, mean)
testX = du.featurewise_std_normalization(testX, std)
Y = du.to_categorical(Y, 10)
testY = du.to_categorical(testY, 10)

# Building Residual Network
net = tflearn.input_data(shape=[None, 32, 32, 3])
net = tflearn.conv_2d(net, 32, 3)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.shallow_residual_block(net, 4, 32, regularizer='L2')
net = tflearn.shallow_residual_block(net, 1, 32, downsample=True,
                                     regularizer='L2')
net = tflearn.shallow_residual_block(net, 4, 64, regularizer='L2')
net = tflearn.shallow_residual_block(net, 1, 64, downsample=True,
                                     regularizer='L2')
net = tflearn.shallow_residual_block(net, 5, 128, regularizer='L2')
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, 10, activation='softmax')
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=16000, staircase=True)
net = tflearn.regression(net, optimizer=mom,
                         loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet_cifar10',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=1.0)
model.fit(X, Y, n_epoch=200, validation_set=(testX, testY),
          show_metric=True, batch_size=128, shuffle=True,
          run_id='resnet_cifar10')
