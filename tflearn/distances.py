""" Distance Ops """

from __future__ import division, print_function, absolute_import

import tensorflow as tf

from .utils import get_from_module


def get(identifier):
    if hasattr(identifier, '__call__'):
        return identifier
    else:
        return get_from_module(identifier, globals(), 'distances')


def euclidean(a, b):
    return tf.sqrt(tf.reduce_sum(tf.square(a - b),
                                 reduction_indices=0))


def cosine(a, b):
    return 1 - tf.matmul(a, b)
