# -*- coding: utf-8 -*-

""" Variational Auto-Encoder Example.

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

# Params
original_dim = 784 # MNIST images are 28x28 pixels
hidden_dim = 256
latent_dim = 2

# Building the encoder
encoder = tflearn.input_data(shape=[None, 784], name='input_images')
encoder = tflearn.fully_connected(encoder, hidden_dim, activation='relu')
z_mean = tflearn.fully_connected(encoder, latent_dim)
z_std = tflearn.fully_connected(encoder, latent_dim)

# Sampler: Normal (gaussian) random distribution
eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0,
                       name='epsilon')
z = z_mean + tf.exp(z_std / 2) * eps

# Building the decoder (with scope to re-use these layers later)
decoder = tflearn.fully_connected(z, hidden_dim, activation='relu',
                                  scope='decoder_h')
decoder = tflearn.fully_connected(decoder, original_dim, activation='sigmoid',
                                  scope='decoder_out')

# Define VAE Loss
def vae_loss(x_reconstructed, x_true):
    # Reconstruction loss
    encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \
                         + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
    # KL Divergence loss
    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)

net = tflearn.regression(decoder, optimizer='rmsprop', learning_rate=0.001,
                         loss=vae_loss, metric=None, name='target_images')

# We will need 2 models, one for training that will learn the latent
# representation, and one that can take random normal noise as input and
# use the decoder part of the network to generate an image

# Train the VAE
training_model = tflearn.DNN(net, tensorboard_verbose=0)
training_model.fit({'input_images': X}, {'target_images': X}, n_epoch=100,
                   validation_set=(testX, testX), batch_size=256, run_id="vae")

# Build an image generator (re-using the decoding layers)
# Input data is a normal (gaussian) random distribution (with dim = latent_dim)
input_noise = tflearn.input_data(shape=[None, latent_dim], name='input_noise')
decoder = tflearn.fully_connected(input_noise, hidden_dim, activation='relu',
                                  scope='decoder_h', reuse=True)
decoder = tflearn.fully_connected(decoder, original_dim, activation='sigmoid',
                                  scope='decoder_out', reuse=True)
generator_model = tflearn.DNN(decoder, session=training_model.session)

# Building a manifold of generated digits
n = 25 # Figure row size
figure = np.zeros((28 * n, 28 * n))
# Random normal distributions to feed network with
x_axis = norm.ppf(np.linspace(0., 1., n))
y_axis = norm.ppf(np.linspace(0., 1., n))

for i, x in enumerate(x_axis):
    for j, y in enumerate(y_axis):
        samples = np.array([[x, y]])
        x_reconstructed = generator_model.predict({'input_noise': samples})
        digit = np.array(x_reconstructed[0]).reshape(28, 28)
        figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
