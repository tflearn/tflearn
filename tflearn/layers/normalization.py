# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow.compat.v1 as tf
from tensorflow.python.training import moving_averages

import tflearn
from .. import utils
from .. import variables as vs
from ..utils import get_from_module


def get(identifier):
    if hasattr(identifier, '__call__'):
        return identifier
    else:
        return get_from_module(identifier, globals(), 'normalization')


def batch_normalization(incoming, beta=0.0, gamma=1.0, epsilon=1e-5,
                        decay=0.9, stddev=0.002, trainable=True,
                        restore=True, reuse=False, scope=None,
                        name="BatchNormalization"):
    """ Batch Normalization.

    Normalize activations of the previous layer at each batch.

    Arguments:
        incoming: `Tensor`. Incoming Tensor.
        beta: `float`. Default: 0.0.
        gamma: `float`. Default: 1.0.
        epsilon: `float`. Defalut: 1e-5.
        decay: `float`. Default: 0.9.
        stddev: `float`. Standard deviation for weights initialization.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: `str`. A name for this layer (optional).

    References:
        Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shif. Sergey Ioffe, Christian Szegedy. 2015.

    Links:
        [http://arxiv.org/pdf/1502.03167v3.pdf](http://arxiv.org/pdf/1502.03167v3.pdf)

    """

    input_shape = utils.get_incoming_shape(incoming)
    input_ndim = len(input_shape)

    gamma_init = tf.random_normal_initializer(mean=gamma, stddev=stddev)

    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:
        name = scope.name
        beta = vs.variable('beta', shape=[input_shape[-1]],
                           initializer=tf.constant_initializer(beta),
                           trainable=trainable, restore=restore)
        gamma = vs.variable('gamma', shape=[input_shape[-1]],
                            initializer=gamma_init, trainable=trainable,
                            restore=restore)
        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, beta)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, gamma)
        if not restore:
            tf.add_to_collection(tf.GraphKeys.EXCL_RESTORE_VARS, beta)
            tf.add_to_collection(tf.GraphKeys.EXCL_RESTORE_VARS, gamma)

        axis = list(range(input_ndim - 1))

        moving_mean = vs.variable('moving_mean', input_shape[-1:],
                                  initializer=tf.zeros_initializer(),
                                  trainable=False, restore=restore)
        moving_variance = vs.variable('moving_variance',
                                      input_shape[-1:],
                                      initializer=tf.constant_initializer(1.),
                                      trainable=False,
                                      restore=restore)

        # Define a function to update mean and variance
        def update_mean_var():
            mean, variance = tf.nn.moments(incoming, axis)

            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay, zero_debias=False)
            update_moving_variance = moving_averages.assign_moving_average(
                moving_variance, variance, decay, zero_debias=False)

            with tf.control_dependencies(
                    [update_moving_mean, update_moving_variance]):
                return tf.identity(mean), tf.identity(variance)

        # Retrieve variable managing training mode
        is_training = tflearn.get_training_mode()
        mean, var = tf.cond(
            is_training, update_mean_var, lambda: (moving_mean, moving_variance))

        inference = tf.nn.batch_normalization(
            incoming, mean, var, beta, gamma, epsilon)
        inference.set_shape(input_shape)


    # Add attributes for easy access
    inference.scope = scope
    inference.beta = beta
    inference.gamma = gamma

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def local_response_normalization(incoming, depth_radius=5, bias=1.0,
                                 alpha=0.0001, beta=0.75,
                                 name="LocalResponseNormalization"):
    """ Local Response Normalization.

    Input:
        4-D Tensor Layer.

    Output:
        4-D Tensor Layer. (Same dimension as input).

    Arguments:
        incoming: `Tensor`. Incoming Tensor.
        depth_radius: `int`. 0-D.  Half-width of the 1-D normalization window.
            Defaults to 5.
        bias: `float`. An offset (usually positive to avoid dividing by 0).
            Defaults to 1.0.
        alpha: `float`. A scale factor, usually positive. Defaults to 0.0001.
        beta: `float`. An exponent. Defaults to `0.5`.
        name: `str`. A name for this layer (optional).

    """

    with tf.name_scope(name) as scope:
        inference = tf.nn.lrn(incoming, depth_radius=depth_radius,
                              bias=bias, alpha=alpha,
                              beta=beta, name=name)

    inference.scope = scope

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def l2_normalize(incoming, dim, epsilon=1e-12, name="l2_normalize"):
    """ L2 Normalization.

    Normalizes along dimension `dim` using an L2 norm.

    For a 1-D tensor with `dim = 0`, computes
    ```
    output = x / sqrt(max(sum(x**2), epsilon))
    ```

    For `x` with more dimensions, independently normalizes each 1-D slice along
    dimension `dim`.

    Arguments:
        incoming: `Tensor`. Incoming Tensor.
        dim: `int`. Dimension along which to normalize.
        epsilon: `float`. A lower bound value for the norm. Will use
            `sqrt(epsilon)` as the divisor if `norm < sqrt(epsilon)`.
        name: `str`. A name for this layer (optional).

    Returns:
      A `Tensor` with the same shape as `x`.
    """
    with tf.name_scope(name) as name:
        x = tf.convert_to_tensor(incoming, name="x")
        square_sum = tf.reduce_sum(tf.square(x), [dim], keep_dims=True)
        x_inv_norm = tf.rsqrt(tf.maximum(square_sum, epsilon))

    return tf.multiply(x, x_inv_norm, name=name)
