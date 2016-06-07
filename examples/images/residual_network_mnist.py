# -*- coding: utf-8 -*-

""" Deep Residual Network.

Applying a Deep Residual Network to MNIST Dataset classification task.

References:
    - K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image
      Recognition, 2015.
    - Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
      learning applied to document recognition." Proceedings of the IEEE,
      86(11):2278-2324, November 1998.

Links:
    - [Deep Residual Network](http://arxiv.org/pdf/1512.03385.pdf)
    - [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

"""

from __future__ import division, print_function, absolute_import

import tflearn
import tflearn.data_utils as du

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])
X, mean = du.featurewise_zero_center(X)
testX = du.featurewise_zero_center(testX, mean)

# Building Residual Network
net = tflearn.input_data(shape=[None, 28, 28, 1])
net = tflearn.conv_2d(net, 64, 3, activation='relu', bias=False)
# Residual blocks
net = tflearn.residual_bottleneck(net, 3, 16, 64)
net = tflearn.residual_bottleneck(net, 1, 32, 128, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 32, 128)
net = tflearn.residual_bottleneck(net, 1, 64, 256, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 64, 256)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.regression(net, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.1)
# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet_mnist',
                    max_checkpoints=10, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=100, validation_set=(testX, testY),
          show_metric=True, batch_size=256, run_id='resnet_mnist')
