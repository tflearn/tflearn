# -*- coding: utf-8 -*-

""" DCGAN Example.

Using a variational auto-encoder to generate digits images from noise.
MNIST handwritten digits are used as training examples.

References:
    - Auto-Encoding Variational Bayes The International Conference on Learning
    Representations (ICLR), Banff, 2014. D.P. Kingma, M. Welling
    - Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    - [VAE Paper] https://arxiv.org/abs/1312.6114
    - [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf

import tflearn

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

# Params
original_dim = 784 # MNIST images are 28x28 pixels
batch_size = 32


# Build the Generative Network
def generative_net(x, reuse=False):
    with tf.variable_scope("GenerativeNetwork", reuse=reuse):
        gen_net = tflearn.fully_connected(x, n_units=7*7*64)
        gen_net = tf.reshape(gen_net, shape=[-1, 7, 7, 64])
        gen_net = tflearn.upsample_2d(gen_net, 2)
        gen_net = tflearn.conv_2d(gen_net, 32, 2)
        gen_net = tflearn.upsample_2d(gen_net, 2)
        gen_net = tflearn.conv_2d(gen_net, 1, 2)
        gen_net = tf.nn.sigmoid(gen_net)
    return gen_net


# Build the Discriminative Network
def discriminative_net(x, reuse=False):
    with tf.variable_scope("DiscriminativeNetwork", reuse=reuse):
        disc_net = tflearn.conv_2d(x, 256, 5, strides=2)
        disc_net = tflearn.batch_normalization(disc_net)
        disc_net = tf.nn.relu(disc_net)
        disc_net = tflearn.conv_2d(disc_net, 512, 5, strides=2)
        disc_net = tflearn.batch_normalization(disc_net)
        disc_net = tf.nn.relu(disc_net)
        disc_net = tflearn.fully_connected(disc_net, 2)
        disc_net = tf.nn.softmax(disc_net)
    return disc_net

# Common Session
session = tf.Session()

# Build the discriminative network with input as a 28 x 28 image
disc_net = tflearn.input_data(shape=[None, 28, 28, 1], name='input_discriminator')
disc_net = discriminative_net(disc_net, reuse=False)
disc_net = tflearn.regression(disc_net, loss='categorical_crossentropy',
                              optimizer='adam', learning_rate=0.001,
                              trainable_vars=tflearn.get_layer_variables_by_scope(
                                  "DiscriminativeNetwork"),
                              name='target_discriminator')
disc_net = tflearn.DNN(disc_net, session=session)

# Build the generator network with noise as input
gen_net = tflearn.input_data(shape=[None, 100], name='input_generator')
gen_net = generative_net(gen_net, reuse=False)
gen_net = tflearn.DNN(gen_net)

# Build the stacked Generative - Discriminative Network with noise input
# We need to reuse the networks a second time (so reuse=True to reuse the
# weights instead of re-creating new ones).
gan_net = tflearn.input_data(shape=[None, 100], name='input_stacked')
gan_net = generative_net(gan_net, reuse=True)
# input of discriminator is the output of the generative network this time
gan_net = discriminative_net(gan_net, reuse=True)
gan_net = tflearn.regression(gan_net, loss='categorical_crossentropy',
                             optimizer='adam', learning_rate=0.0001,
                             trainable_vars=tflearn.get_layer_variables_by_scope(
                                 "GenerativeNetwork"),
                             name='target_stacked')
gan_net = tflearn.DNN(gan_net, session=session)

init = tf.group(tf.global_variables_initializer())
session.run(tf.variables_initializer(tf.get_collection_ref('is_training')))
session.run(init)

for i in range(10):

    # Step1: Training Discriminator alone
    # Create random uniform distribution noise
    noise = np.random.uniform(0., 1., size=[batch_size, 100])
    generate_images = gen_net.predict({'input_generator': noise})

    # Train 1 step of the discriminator
    # Get a random batch from X (random index of batch size)
    rand_index = np.random.randint(0, X.shape[0], size=batch_size)
    X_real = X[rand_index, :, :, :]

    # Labels for real images
    Y_real = np.ones(shape=[batch_size])
    # Labels for fake images
    Y_fake = np.zeros(shape=[batch_size])

    # Concatenate data to feed the discriminator
    X_batch = np.concatenate((X_real, generate_images))
    Y_batch = np.concatenate((Y_real, Y_fake))
    Y_batch = tflearn.data_utils.to_categorical(Y_batch, 2)

    loss_disc = disc_net.fit_batch({'input_discriminator': X_batch},
                                   {'target_discriminator': Y_batch})
    print("Loss Disc", loss_disc)

    # Step2: Training Generator-Discriminator stack om input noise
    noise = np.random.uniform(0., 1., size=[batch_size, 100])
    Y_batch = np.ones(shape=[batch_size])
    Y_batch = tflearn.data_utils.to_categorical(Y_batch, 2)

    loss_gan = gan_net.fit_batch({'input_stacked': noise},
                                 {'target_stacked': Y_batch})
    print("Loss GAN", loss_gan)

f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    pred_dict = {'input_generator': [np.random.uniform(0., 1., size=[100])]}
    g = gen_net.predict(pred_dict)[0]
    temp = [[ii, ii, ii] for ii in list(g)]
    a[0][i].imshow(np.reshape(temp, (28, 28, 3)))
    temp = [[ii, ii, ii] for ii in list(g)]
    a[1][i].imshow(np.reshape(temp, (28, 28, 3)))
f.show()
plt.draw()
plt.waitforbuttonpress()
















