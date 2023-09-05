""" Distance Ops """

from __future__ import division, print_function, absolute_import

import tensorflow.compat.v1 as tf

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


def manhattan(a, b):
    return tf.reduce_sum(tf.abs(a - b), reduction_indices=0)


def minkowski(a, b, p):
    return tf.pow(tf.reduce_sum(tf.pow(tf.abs(a - b), p), 
                                reduction_indices=0), 1/p)


def mahalanobis(a, b, C):
    diff = a - b
    return tf.sqrt(tf.matmul(tf.matmul(diff, tf.linalg.inv(C)), diff), 
                   reduction_indices=0)


def hamming(a, b):
    return tf.reduce_sum(tf.cast(tf.not_equal(a, b), tf.float32))


def jaccard(a, b):
    intersection = tf.reduce_sum(tf.minimum(a, b))
    union = tf.reduce_sum(tf.maximum(a, b))
    return 1 - (intersection / union)


def canberra(a, b):
    numerator = tf.abs(a - b)
    denominator = tf.abs(a) + tf.abs(b)
    return tf.reduce_sum(numerator / denominator, 
                         reduction_indices=0)


def bray_curtis(a, b):
    numerator = tf.reduce_sum(tf.abs(a - b))
    denominator = tf.reduce_sum(tf.abs(a) + tf.abs(b))
    return numerator / denominator
    
