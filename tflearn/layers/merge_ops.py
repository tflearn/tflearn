# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf


def merge(tensors_list, mode, axis=1, name="Merge"):
    """ Merge.

    Merge a list of `Tensor` into a single one. A merging 'mode' must be
    specified, check below for the different options.

    Input:
        List of Tensors.

    Output:
        Merged Tensors.

    Arguments:
        tensors_list: A list of `Tensor`, A list of tensors to merge.
        mode: `str`. Merging mode, it supports:
            ```
            'concat': concatenate outputs along specified axis
            'elemwise_sum': outputs element-wise sum
            'elemwise_mul': outputs element-wise sum
            'sum': outputs element-wise sum along specified axis
            'mean': outputs element-wise average along specified axis
            'prod': outputs element-wise multiplication along specified axis
            'max': outputs max elements along specified axis
            'min': outputs min elements along specified axis
            'and': `logical and` btw outputs elements along specified axis
            'or': `logical or` btw outputs elements along specified axis
            ```
        axis: `int`. Represents the axis to use for merging mode.
            In most cases: 0 for concat and 1 for other modes.
        name: A name for this layer (optional). Default: 'Merge'.

    """

    assert len(tensors_list) > 1, "Merge required 2 or more tensors."

    with tf.name_scope(name) as scope:
        tensors = [l for l in tensors_list]
        if mode == 'concat':
            inference = tf.concat(tensors, axis)
        elif mode == 'elemwise_sum':
            inference = tensors[0]
            for i in range(1, len(tensors)):
                inference = tf.add(inference, tensors[i])
        elif mode == 'elemwise_mul':
            inference = tensors[0]
            for i in range(1, len(tensors)):
                inference = tf.multiply(inference, tensors[i])
        elif mode == 'sum':
            inference = tf.reduce_sum(tf.concat(tensors, axis),
                                      reduction_indices=axis)
        elif mode == 'mean':
            inference = tf.reduce_mean(tf.concat(tensors, axis),
                                       reduction_indices=axis)
        elif mode == 'prod':
            inference = tf.reduce_prod(tf.concat(tensors, axis),
                                       reduction_indices=axis)
        elif mode == 'max':
            inference = tf.reduce_max(tf.concat(tensors, axis),
                                      reduction_indices=axis)
        elif mode == 'min':
            inference = tf.reduce_min(tf.concat(tensors, axis),
                                      reduction_indices=axis)
        elif mode == 'and':
            inference = tf.reduce_all(tf.concat(tensors, axis),
                                      reduction_indices=axis)
        elif mode == 'or':
            inference = tf.reduce_any(tf.concat(tensors, axis),
                                      reduction_indices=axis)
        else:
            raise Exception("Unknown merge mode", str(mode))

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def merge_outputs(tensor_list, name="MergeOutputs"):
    """ Merge Outputs.

    A layer that concatenate all outputs of a network into a single tensor.

    Input:
        List of Tensors [_shape_].

    Output:
        Concatenated Tensors [nb_tensors, _shape_].

    Arguments:
        tensor_list: list of `Tensor`. The network outputs.
        name: `str`. A name for this layer (optional).

    Returns:
        A `Tensor`.

    """
    with tf.name_scope(name) as scope:
        x = tf.concat(tensor_list, 1)

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, x)

    return x
