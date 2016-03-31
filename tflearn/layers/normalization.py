# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf

import tflearn
from .. import utils
from .. import variables as vs


def batch_normalization(incoming, beta=0.0, gamma=1.0, epsilon=1e-5,
                        decay=0.999, trainable=True, restore=True,
                        stddev=0.002, name="BatchNormalization"):
    """ Batch Normalization.

    Normalize activations of the previous layer at each batch.

    Arguments:
        incoming: `Tensor`. Incoming Tensor.
        beta: `float`. Default: 0.0.
        gamma: `float`. Default: 1.0.
        epsilon: `float`. Defalut: 1e-5.
        decay: `float`. Default: 0.999.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        stddev: `float`. Standard deviation for weights initialization.
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

    with tf.name_scope(name) as scope:
        beta = vs.variable(scope + 'beta', shape=[input_shape[-1]],
                           initializer=tf.constant_initializer(beta),
                           trainable=trainable, restore=restore)
        gamma = vs.variable(scope + 'gamma', shape=[input_shape[-1]],
                            initializer=gamma_init, trainable=trainable,
                            restore=restore)
        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope, beta)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope, gamma)
        if not restore:
            tf.add_to_collection(tf.GraphKeys.EXCL_RESTORE_VARS, beta)
            tf.add_to_collection(tf.GraphKeys.EXCL_RESTORE_VARS, gamma)

        ema = tf.train.ExponentialMovingAverage(decay=decay)
        axis = [i for i in range(input_ndim - 1)]
        if len(axis) < 1: axis = [0]
        batch_mean, batch_var = tf.nn.moments(incoming, axis,
                                              name='moments')
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(
            batch_var)

        def update_mean_var():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(ema_mean), tf.identity(ema_var)

        is_training = tflearn.get_training_mode()
        mean, var = tf.python.control_flow_ops.cond(
            is_training, update_mean_var, lambda: (ema_mean, ema_var))

        try:
            inference = tf.nn.batch_normalization(
                incoming, mean, var, beta, gamma, epsilon)
        # Fix for old Tensorflow
        except Exception as e:
            inference = tf.nn.batch_norm_with_global_normalization(
                incoming, mean, var, beta, gamma, epsilon,
                scale_after_normalization=True,
            )

    # Add attributes for easy access
    inference.scope = scope
    inference.beta = beta
    inference.gamma = gamma

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

    """

    with tf.name_scope(name) as scope:
        inference = tf.nn.lrn(incoming, depth_radius=depth_radius,
                              bias=bias, alpha=alpha,
                              beta=beta, name=name)

    inference.scope = scope

    return inference
