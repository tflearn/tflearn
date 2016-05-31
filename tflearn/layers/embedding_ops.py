# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf

from .. import variables as vs
from .. import utils
from .. import initializations


def embedding(incoming, input_dim, output_dim, weights_init='truncated_normal',
              trainable=True, restore=True, name="Embedding"):
    """ Embedding.

    Embedding layer for a sequence of ids.

    Input:
        2-D Tensor [samples, ids].

    Output:
        3-D Tensor [samples, embedded_ids, features].

    Arguments:
        incoming: Incoming 2-D Tensor.
        input_dim: list of `int`. Vocabulary size (number of ids).
        output_dim: list of `int`. Embedding size.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'truncated_normal'.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model
        name: A name for this layer (optional). Default: 'Embedding'.

    """

    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 2, "Incoming Tensor shape must be 2-D"
    n_inputs = int(np.prod(input_shape[1:]))

    W_init = weights_init
    if isinstance(weights_init, str):
        W_init = initializations.get(weights_init)()

    with tf.name_scope(name) as scope:
        with tf.device('/cpu:0'):
            W = vs.variable(scope + "W", shape=[input_dim, output_dim],
                            initializer=W_init, trainable=trainable,
                            restore=restore)
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope, W)

        inference = tf.cast(incoming, tf.int32)
        inference = tf.nn.embedding_lookup(W, inference)
        inference = tf.transpose(inference, [1, 0, 2])
        inference = tf.reshape(inference, shape=[-1, output_dim])
        inference = tf.split(0, n_inputs, inference)

    # TODO: easy access those var
    # inference.W = W
    # inference.scope = scope

    return inference
