from __future__ import division, print_function, absolute_import

import tensorflow as tf
from .. import losses


"""
Regularizer contains some useful functions to help add regularization to
weights and activations.
"""


def add_weights_regularizer(variable, loss="L2", weight_decay=0.001,
                            add_to_collection=None):
    """ add_weights_regularizer.

    Add a weights regularizer to the provided Tensor

    Arguments:
        variable: `Variable`. Tensor to add regularization.
        loss: `str`. Regularization mode.
        weight_decay: `float`. Decay to use for regularization.
        add_to_collection: `str`. Add the regularization loss to the
            specified collection. Default: tf.GraphKeys.REGULARIZATION_LOSSES.

    Returns:
        `tf.Tensor`. The weight regularizer.

    """
    if not add_to_collection:
        add_to_collection = tf.GraphKeys.REGULARIZATION_LOSSES
    if isinstance(loss, str):
        regul = losses.get(loss)
        weights_regularizer = regul(variable, weight_decay)
    elif loss and callable(loss):
        weights_regularizer = loss(variable)
    else:
        weights_regularizer = loss
    if add_to_collection:
        tf.add_to_collection(add_to_collection, weights_regularizer)
    return weights_regularizer


def add_activation_regularizer(op, loss="L2", activ_decay=0.001,
                               add_to_collection=None):
    raise NotImplementedError
