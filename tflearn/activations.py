# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf

import tflearn
from . import initializations
from . import variables as va
from .utils import get_from_module


def get(identifier):
    if hasattr(identifier, '__call__'):
        return identifier
    else:
        return get_from_module(identifier, globals(), 'activation')


""" Activation Functions """


def linear(x):
    """ Linear.

    f(x) = x

    Arguments:
        x : A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
            `int16`, or `int8`.

    Returns:
        The incoming Tensor (without changes).
    """
    return x


def tanh(x):
    """ Tanh.

    Computes hyperbolic tangent of `x` element-wise.

    Arguments:
        x: A Tensor with type `float`, `double`, `int32`, `complex64`, `int64`,
            or `qint32`.

    Returns:
        A Tensor with the same type as `x` if `x.dtype != qint32` otherwise
          the return type is `quint8`.
    """
    return tf.tanh(x)


def sigmoid(x):
    """ Sigmoid.

    Computes sigmoid of `x` element-wise.
    Specifically, `y = 1 / (1 + exp(-x))`.

    Arguments:
        x: A Tensor with type `float`, `double`, `int32`, `complex64`, `int64`,
            or `qint32`.

    Returns:
        A Tensor with the same type as `x` if `x.dtype != qint32` otherwise
        the return type is `quint8`.
    """
    return tf.nn.sigmoid(x)


def softmax(x):
    """ Softmax.

    Computes softmax activations.

    For each batch `i` and class `j` we have

      softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))

    Arguments:
        x: A `Tensor`. Must be one of the following types: `float32`,
            `float64`. 2-D with shape `[batch_size, num_classes]`.

    Returns:
        A `Tensor`. Has the same type as `x`. Same shape as `x`.
    """
    return tf.nn.softmax(x)


def softplus(x):
    """ Softplus.

    Computes softplus: `log(exp(features) + 1)`.

    Arguments:
        x: A `Tensor`. Must be one of the following types: `float32`,
            `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`.

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    return tf.nn.softplus(x)


def softsign(x):
    """ Softsign.

    Computes softsign: `features / (abs(features) + 1)`.

    Arguments:
        x: A `Tensor`. Must be one of the following types: `float32`,
            `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`.

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    return tf.nn.softsign(x)


def relu(x):
    """ ReLU.

    Computes rectified linear: `max(features, 0)`.

    Arguments:
        x: A `Tensor`. Must be one of the following types: `float32`,
            `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`.

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    return tf.nn.relu(x)


def relu6(x):
    """ ReLU6.

    Computes Rectified Linear 6: `min(max(features, 0), 6)`.

    Arguments:
        x: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
            `int16`, or `int8`.

    Returns:
        A `Tensor` with the same type as `x`.
    """
    return tf.nn.relu6(x)


def leaky_relu(x, alpha=0.1, name="LeakyReLU"):
    """ LeakyReLU.

    Modified version of ReLU, introducing a nonzero gradient for negative
    input.

    Arguments:
        x: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
            `int16`, or `int8`.
        alpha: `float`. slope.
        name: A name for this activation op (optional).

    Returns:
        A `Tensor` with the same type as `x`.

    References:
        Rectifier Nonlinearities Improve Neural Network Acoustic Models,
        Maas et al. (2013).

    Links:
        [http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf]
        (http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)

    """

    with tf.name_scope(name) as scope:
        m_x = tf.nn.relu(-x)
        x = tf.nn.relu(x)
        x -= alpha * m_x

    x.scope = scope

    return x

# Shortcut
leakyrelu = leaky_relu


def prelu(x, channel_shared=False, weights_init='zeros', trainable=True,
          restore=True, reuse=False, scope=None, name="PReLU"):
    """ PReLU.

    Parametric Rectified Linear Unit.

    Arguments:
        x: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
            `int16`, or `int8`.
        channel_shared: `bool`. Single weight is shared by all channels
        weights_init: `str`. Weights initialization. Default: zeros.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. Restore or not alphas.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        name: A name for this activation op (optional).

    Attributes:
        scope: `str`. This op scope.
        alphas: `Variable`. PReLU alphas.

    Returns:
        A `Tensor` with the same type as `x`.

    References:
        Delving Deep into Rectifiers: Surpassing Human-Level Performance
        on ImageNet Classification. He et al., 2014.

    Links:
        [http://arxiv.org/pdf/1502.01852v1.pdf]
        (http://arxiv.org/pdf/1502.01852v1.pdf)

    """
    if channel_shared:
        w_shape = (1,)
    else:
        w_shape = tflearn.utils.get_incoming_shape(x)[-1:]

    with tf.variable_scope(scope, default_name=name, values=[x],
                           reuse=reuse) as scope:
        W_init = initializations.get(weights_init)()
        alphas = va.variable("alphas", shape=w_shape, initializer=W_init,
                             restore=restore, trainable=trainable)

        x = tf.nn.relu(x) + tf.multiply(alphas, (x - tf.abs(x))) * 0.5

    x.scope = scope
    x.alphas = alphas

    return x


def elu(x):
    """ ELU.

    Exponential Linear Unit.

    Arguments:
        x : A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
            `int16`, or `int8`

    Returns:
        A `tuple` of `tf.Tensor`. This layer inference, i.e. output Tensors
        at training and testing time.

    References:
        Fast and Accurate Deep Network Learning by Exponential Linear Units,
        Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter. 2015.

    Links:
        [http://arxiv.org/abs/1511.07289](http://arxiv.org/abs/1511.07289)

    """

    return tf.nn.elu(x)


def crelu(x):
    """ CReLU

    Computes Concatenated ReLU.

    Concatenates a ReLU which selects only the positive part of the activation
    with a ReLU which selects only the negative part of the activation. Note
    that as a result this non-linearity doubles the depth of the activations.

    Arguments:
        x : A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
            `int16`, or `int8`.

    Returns:
        A `Tensor` with the same type as `x`.

    Links:
        [https://arxiv.org/abs/1603.05201](https://arxiv.org/abs/1603.05201)

    """

    return tf.nn.crelu(x)


def selu(x):
    """ SELU.

    Scaled Exponential Linear Unit.

    Arguments
        x : A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
            `int16`, or `int8`

    References:
        Self-Normalizing Neural Networks, Klambauer et al., 2017.

    Links:
        [https://arxiv.org/abs/1706.02515](https://arxiv.org/abs/1706.02515)

    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))
