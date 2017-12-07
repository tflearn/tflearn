# -*- coding: utf-8 -*-
""" Densely Connected Convolutional Networks.

Applying a 'DenseNet' to CIFAR-10 Dataset classification task.

References:
    - G. Huang, Z. Liu, K. Q. Weinberger, L. van der Maaten. Densely Connected 
        Convolutional Networks, 2016.

Links:
    - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
    - [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

"""

from __future__ import division, print_function, absolute_import

import tflearn

# Growth Rate (12, 16, 32, ...)
k = 12

# Depth (40, 100, ...)
L = 40
nb_layers = int((L - 4) / 3)

# Data loading
from tflearn.datasets import cifar10
(X, Y), (testX, testY) = cifar10.load_data()
Y = tflearn.data_utils.to_categorical(Y)
testY = tflearn.data_utils.to_categorical(testY)

# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop([32, 32], padding=4)

# Building Residual Network
net = tflearn.input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.densenet_block(net, nb_layers, k)
net = tflearn.densenet_block(net, nb_layers, k)
net = tflearn.densenet_block(net, nb_layers, k)
net = tflearn.global_avg_pool(net)

# Regression
net = tflearn.fully_connected(net, 10, activation='softmax')
opt = tflearn.Nesterov(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=opt,
                         loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, checkpoint_path='model_densenet_cifar10',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)

model.fit(X, Y, n_epoch=200, validation_set=(testX, testY),
          snapshot_epoch=False, snapshot_step=500,
          show_metric=True, batch_size=128, shuffle=True,
          run_id='densenet_cifar10')
