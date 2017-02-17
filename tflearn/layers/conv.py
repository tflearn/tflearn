# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
from math import ceil

import tflearn
from .. import variables as vs
from .. import activations
from .. import initializations
from .. import losses
from .. import utils


def conv_2d(incoming, nb_filter, filter_size, strides=1, padding='same',
            activation='linear', bias=True, weights_init='uniform_scaling',
            bias_init='zeros', regularizer=None, weight_decay=0.001,
            trainable=True, restore=True, reuse=False, scope=None,
            name="Conv2D"):
    """ Convolution 2D.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        4-D Tensor [batch, new height, new width, nb_filter].

    Arguments:
        incoming: `Tensor`. Incoming 4-D Tensor.
        nb_filter: `int`. The number of convolutional filters.
        filter_size: `int` or `list of int`. Size of filters.
        strides: 'int` or list of `int`. Strides of conv operation.
            Default: [1 1 1 1].
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        activation: `str` (name) or `function` (returning a `Tensor`) or None.
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'truncated_normal'.
        bias_init: `str` (name) or `Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'Conv2D'.

    Attributes:
        scope: `Scope`. This layer scope.
        W: `Variable`. Variable representing filter weights.
        b: `Variable`. Variable representing biases.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"
    filter_size = utils.autoformat_filter_conv2d(filter_size,
                                                 input_shape[-1],
                                                 nb_filter)
    strides = utils.autoformat_kernel_2d(strides)
    padding = utils.autoformat_padding(padding)

    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:
        name = scope.name

        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        W_regul = None
        if regularizer:
            W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
        W = vs.variable('W', shape=filter_size, regularizer=W_regul,
                        initializer=W_init, trainable=trainable,
                        restore=restore)

        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)

        b = None
        if bias:
            if isinstance(bias_init, str):
                bias_init = initializations.get(bias_init)()
            b = vs.variable('b', shape=nb_filter, initializer=bias_init,
                            trainable=trainable, restore=restore)
            # Track per layer variables
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b)

        inference = tf.nn.conv2d(incoming, W, strides, padding)
        if b: inference = tf.nn.bias_add(inference, b)

        if activation:
            if isinstance(activation, str):
                inference = activations.get(activation)(inference)
            elif hasattr(activation, '__call__'):
                inference = activation(inference)
            else:
                raise ValueError("Invalid Activation.")

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights.
    inference.scope = scope
    inference.W = W
    inference.b = b

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def conv_2d_transpose(incoming, nb_filter, filter_size, output_shape,
                      strides=1, padding='same', activation='linear',
                      bias=True, weights_init='uniform_scaling',
                      bias_init='zeros', regularizer=None, weight_decay=0.001,
                      trainable=True, restore=True, reuse=False, scope=None,
                      name="Conv2DTranspose"):

    """ Convolution 2D Transpose.

    This operation is sometimes called "deconvolution" after (Deconvolutional
    Networks)[http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf], but is
    actually the transpose (gradient) of `conv_2d` rather than an actual
    deconvolution.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        4-D Tensor [batch, new height, new width, nb_filter].

    Arguments:
        incoming: `Tensor`. Incoming 4-D Tensor.
        nb_filter: `int`. The number of convolutional filters.
        filter_size: `int` or `list of int`. Size of filters.
        output_shape: `list of int`. Dimensions of the output tensor.
            Can optionally include the number of conv filters.
            [new height, new width, nb_filter] or [new height, new width].
        strides: `int` or list of `int`. Strides of conv operation.
            Default: [1 1 1 1].
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'truncated_normal'.
        bias_init: `str` (name) or `Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'Conv2DTranspose'.

    Attributes:
        scope: `Scope`. This layer scope.
        W: `Variable`. Variable representing filter weights.
        b: `Variable`. Variable representing biases.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"

    filter_size = utils.autoformat_filter_conv2d(filter_size,
                                                 nb_filter,
                                                 input_shape[-1])
    strides = utils.autoformat_kernel_2d(strides)
    padding = utils.autoformat_padding(padding)

    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:
        name = scope.name

        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        W_regul = None
        if regularizer:
            W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
        W = vs.variable('W', shape=filter_size,
                        regularizer=W_regul, initializer=W_init,
                        trainable=trainable, restore=restore)
        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)

        b = None
        if bias:
            if isinstance(bias_init, str):
                bias_init = initializations.get(bias_init)()
            b = vs.variable('b', shape=nb_filter, initializer=bias_init,
                            trainable=trainable, restore=restore)
            # Track per layer variables
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b)

        # Determine the complete shape of the output tensor.
        batch_size = tf.gather(tf.shape(incoming), tf.constant([0]))
        if len(output_shape) == 2:
            output_shape = output_shape + [nb_filter]
        elif len(output_shape) != 3:
            raise Exception("output_shape length error: "
                            + str(len(output_shape))
                            + ", only a length of 2 or 3 is supported.")
        complete_out_shape = tf.concat([batch_size, tf.constant(output_shape)], 0)

        inference = tf.nn.conv2d_transpose(incoming, W, complete_out_shape,
                                           strides, padding)

        # Reshape tensor so its shape is correct.
        inference.set_shape([None] + output_shape)

        if b: inference = tf.nn.bias_add(inference, b)

        if isinstance(activation, str):
            inference = activations.get(activation)(inference)
        elif hasattr(activation, '__call__'):
            inference = activation(inference)
        else:
            raise ValueError("Invalid Activation.")

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights.
    inference.scope = scope
    inference.W = W
    inference.b = b

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def max_pool_2d(incoming, kernel_size, strides=None, padding='same',
                name="MaxPool2D"):
    """ Max Pooling 2D.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        4-D Tensor [batch, pooled height, pooled width, in_channels].

    Arguments:
        incoming: `Tensor`. Incoming 4-D Layer.
        kernel_size: 'int` or `list of int`. Pooling kernel size.
        strides: 'int` or `list of int`. Strides of conv operation.
            Default: same as kernel_size.
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        name: A name for this layer (optional). Default: 'MaxPool2D'.

    Attributes:
        scope: `Scope`. This layer scope.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"

    kernel = utils.autoformat_kernel_2d(kernel_size)
    strides = utils.autoformat_kernel_2d(strides) if strides else kernel
    padding = utils.autoformat_padding(padding)

    with tf.name_scope(name) as scope:
        inference = tf.nn.max_pool(incoming, kernel, strides, padding)

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights
    inference.scope = scope

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def avg_pool_2d(incoming, kernel_size, strides=None, padding='same',
                name="AvgPool2D"):
    """ Average Pooling 2D.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        4-D Tensor [batch, pooled height, pooled width, in_channels].

    Arguments:
        incoming: `Tensor`. Incoming 4-D Layer.
        kernel_size: 'int` or `list of int`. Pooling kernel size.
        strides: 'int` or `list of int`. Strides of conv operation.
            Default: same as kernel_size.
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        name: A name for this layer (optional). Default: 'AvgPool2D'.

    Attributes:
        scope: `Scope`. This layer scope.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"

    kernel = utils.autoformat_kernel_2d(kernel_size)
    strides = utils.autoformat_kernel_2d(strides) if strides else kernel
    padding = utils.autoformat_padding(padding)

    with tf.name_scope(name) as scope:
        inference = tf.nn.avg_pool(incoming, kernel, strides, padding)

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights
    inference.scope = scope

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def upsample_2d(incoming, kernel_size, name="UpSample2D"):
    """ UpSample 2D.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        4-D Tensor [batch, pooled height, pooled width, in_channels].

    Arguments:
        incoming: `Tensor`. Incoming 4-D Layer to upsample.
        kernel_size: 'int` or `list of int`. Upsampling kernel size.
        name: A name for this layer (optional). Default: 'UpSample2D'.

    Attributes:
        scope: `Scope`. This layer scope.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"
    kernel = utils.autoformat_kernel_2d(kernel_size)

    with tf.name_scope(name) as scope:
        inference = tf.image.resize_nearest_neighbor(
            incoming, size=input_shape[1:3] * tf.constant(kernel[1:3]))
        inference.set_shape((None, input_shape[1] * kernel[1],
                            input_shape[2] * kernel[2], None))

    # Add attributes to Tensor to easy access weights
    inference.scope = scope

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def upscore_layer(incoming, num_classes, shape=None, kernel_size=4,
                  strides=2, trainable=True, restore=True,
                  reuse=False, scope=None, name='Upscore'):
    """ Upscore.

    This implements the upscore layer as used in
    (Fully Convolutional Networks)[http://arxiv.org/abs/1411.4038].
    The upscore layer is initialized as bilinear upsampling filter.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        4-D Tensor [batch, pooled height, pooled width, in_channels].

    Arguments:
        incoming: `Tensor`. Incoming 4-D Layer to upsample.
        num_classes: `int`. Number of output feature maps.
        shape: `list of int`. Dimension of the output map
            [batch_size, new height, new width]. For convinience four values
             are allows [batch_size, new height, new width, X], where X
             is ignored.
        kernel_size: 'int` or `list of int`. Upsampling kernel size.
        strides: 'int` or `list of int`. Strides of conv operation.
            Default: [1 2 2 1].
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
            name: A name for this layer (optional). Default: 'Upscore'.

    Attributes:
        scope: `Scope`. This layer scope.

    Links:
        (Fully Convolutional Networks)[http://arxiv.org/abs/1411.4038]

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"

    strides = utils.autoformat_kernel_2d(strides)
    filter_size = utils.autoformat_filter_conv2d(kernel_size,
                                                 num_classes,
                                                 input_shape[-1])

    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:
        name = scope.name

        if shape is None:
            # Compute shape out of Bottom
            in_shape = tf.shape(incoming)

            h = ((in_shape[1] - 1) * strides[1]) + 1
            w = ((in_shape[2] - 1) * strides[1]) + 1
            new_shape = [in_shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]
        output_shape = tf.stack(new_shape)

        def get_deconv_filter(f_shape):
            """
            Create filter weights initialized as bilinear upsampling.
            """
            width = f_shape[0]
            heigh = f_shape[0]
            f = ceil(width/2.0)
            c = (2 * f - 1 - f % 2) / (2.0 * f)
            bilinear = np.zeros([f_shape[0], f_shape[1]])
            for x in range(width):
                for y in range(heigh):
                    value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                    bilinear[x, y] = value
            weights = np.zeros(f_shape)
            for i in range(f_shape[2]):
                weights[:, :, i, i] = bilinear

            init = tf.constant_initializer(value=weights,
                                           dtype=tf.float32)
            W = vs.variable(name="up_filter", initializer=init,
                            shape=weights.shape, trainable=trainable,
                            restore=restore)
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)
            return W

        weights = get_deconv_filter(filter_size)
        deconv = tf.nn.conv2d_transpose(incoming, weights, output_shape,
                                        strides=strides, padding='SAME')

    deconv.scope = scope

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, deconv)

    return deconv


def conv_1d(incoming, nb_filter, filter_size, strides=1, padding='same',
            activation='linear', bias=True, weights_init='uniform_scaling',
            bias_init='zeros', regularizer=None, weight_decay=0.001,
            trainable=True, restore=True, reuse=False, scope=None,
            name="Conv1D"):
    """ Convolution 1D.

    Input:
        3-D Tensor [batch, steps, in_channels].

    Output:
        3-D Tensor [batch, new steps, nb_filters].

    Arguments:
        incoming: `Tensor`. Incoming 3-D Tensor.
        nb_filter: `int`. The number of convolutional filters.
        filter_size: 'int` or `list of int`. Size of filters.
        strides: 'int` or `list of int`. Strides of conv operation.
            Default: [1 1 1 1].
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'truncated_normal'.
        bias_init: `str` (name) or `Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'Conv1D'.

    Attributes:
        scope: `Scope`. This layer scope.
        W: `Variable`. Variable representing filter weights.
        b: `Variable`. Variable representing biases.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 3, "Incoming Tensor shape must be 3-D"
    filter_size = utils.autoformat_filter_conv2d(filter_size,
                                                 input_shape[-1],
                                                 nb_filter)
    #filter_size = [1, filter_size[1], 1, 1]
    filter_size[1] = 1
    strides = utils.autoformat_kernel_2d(strides)
    strides = [1, strides[1], 1, 1]
    #strides[1] = 1
    padding = utils.autoformat_padding(padding)

    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:
        name = scope.name

        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        W_regul = None
        if regularizer:
            W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
        W = vs.variable('W', shape=filter_size, regularizer=W_regul,
                        initializer=W_init, trainable=trainable,
                        restore=restore)
        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)

        b = None
        if bias:
            if isinstance(bias_init, str):
                bias_init = initializations.get(bias_init)()
            b = vs.variable('b', shape=nb_filter, initializer=bias_init,
                            trainable=trainable, restore=restore)
            # Track per layer variables
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b)

        # Adding dummy dimension to fit with Tensorflow conv2d
        inference = tf.expand_dims(incoming, 2)
        inference = tf.nn.conv2d(inference, W, strides, padding)
        if b: inference = tf.nn.bias_add(inference, b)
        inference = tf.squeeze(inference, [2])

        if isinstance(activation, str):
            inference = activations.get(activation)(inference)
        elif hasattr(activation, '__call__'):
            inference = activation(inference)
        else:
            raise ValueError("Invalid Activation.")

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights.
    inference.scope = scope
    inference.W = W
    inference.b = b

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def max_pool_1d(incoming, kernel_size, strides=None, padding='same',
                name="MaxPool1D"):
    """ Max Pooling 1D.

    Input:
        3-D Tensor [batch, steps, in_channels].

    Output:
        3-D Tensor [batch, pooled steps, in_channels].

    Arguments:
        incoming: `Tensor`. Incoming 3-D Layer.
        kernel_size: `int` or `list of int`. Pooling kernel size.
        strides: `int` or `list of int`. Strides of conv operation.
            Default: same as kernel_size.
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        name: A name for this layer (optional). Default: 'MaxPool1D'.

    Attributes:
        scope: `Scope`. This layer scope.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 3, "Incoming Tensor shape must be 3-D"

    kernel = utils.autoformat_kernel_2d(kernel_size)
    kernel = [1, kernel[1], 1, 1]
    strides = utils.autoformat_kernel_2d(strides) if strides else kernel
    strides = [1, strides[1], 1, 1]
    padding = utils.autoformat_padding(padding)

    with tf.name_scope(name) as scope:
        inference = tf.expand_dims(incoming, 2)
        inference = tf.nn.max_pool(inference, kernel, strides, padding)
        inference = tf.squeeze(inference, [2])

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights
    inference.scope = scope

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def avg_pool_1d(incoming, kernel_size, strides=None, padding='same',
                name="AvgPool1D"):
    """ Average Pooling 1D.

    Input:
        3-D Tensor [batch, steps, in_channels].

    Output:
        3-D Tensor [batch, pooled steps, in_channels].

    Arguments:
        incoming: `Tensor`. Incoming 3-D Layer.
        kernel_size: `int` or `list of int`. Pooling kernel size.
        strides: `int` or `list of int`. Strides of conv operation.
            Default: same as kernel_size.
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        name: A name for this layer (optional). Default: 'AvgPool1D'.

    Attributes:
        scope: `Scope`. This layer scope.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 3, "Incoming Tensor shape must be 3-D"

    kernel = utils.autoformat_kernel_2d(kernel_size)
    kernel = [1, kernel[1], 1, 1]
    strides = utils.autoformat_kernel_2d(strides) if strides else kernel
    padding = utils.autoformat_padding(padding)

    with tf.name_scope(name) as scope:
        inference = tf.expand_dims(incoming, 2)
        inference = tf.nn.avg_pool(inference, kernel, strides, padding)
        inference = tf.squeeze(inference, [2])

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights
    inference.scope = scope

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def conv_3d(incoming, nb_filter, filter_size, strides=1, padding='same',
            activation='linear', bias=True, weights_init='uniform_scaling',
            bias_init='zeros', regularizer=None, weight_decay=0.001,
            trainable=True, restore=True, reuse=False, scope=None,
            name="Conv3D"):
    """ Convolution 3D.

    Input:
        5-D Tensor [batch, in_depth, in_height, in_width, in_channels].

    Output:
        5-D Tensor [filter_depth, filter_height, filter_width, in_channels, out_channels].

    Arguments:
        incoming: `Tensor`. Incoming 5-D Tensor.
        nb_filter: `int`. The number of convolutional filters.
        filter_size: `int` or `list of int`. Size of filters.
        strides: 'int` or list of `int`. Strides of conv operation.
            Default: [1 1 1 1 1]. Must have strides[0] = strides[4] = 1.
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'truncated_normal'.
        bias_init: `str` (name) or `Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'Conv3D'.

    Attributes:
        scope: `Scope`. This layer scope.
        W: `Variable`. Variable representing filter weights.
        b: `Variable`. Variable representing biases.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 5, "Incoming Tensor shape must be 5-D"
    filter_size = utils.autoformat_filter_conv3d(filter_size,
                                                 input_shape[-1],
                                                 nb_filter)
    strides = utils.autoformat_stride_3d(strides)
    padding = utils.autoformat_padding(padding)

    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:
        name = scope.name

        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        W_regul = None
        if regularizer:
            W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
        W = vs.variable('W', shape=filter_size, regularizer=W_regul,
                        initializer=W_init, trainable=trainable,
                        restore=restore)
        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)

        b = None
        if bias:
            if isinstance(bias_init, str):
                bias_init = initializations.get(bias_init)()
            b = vs.variable('b', shape=nb_filter, initializer=bias_init,
                            trainable=trainable, restore=restore)
            # Track per layer variables
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b)

        inference = tf.nn.conv3d(incoming, W, strides, padding)
        if b: inference = tf.nn.bias_add(inference, b)

        if isinstance(activation, str):
            inference = activations.get(activation)(inference)
        elif hasattr(activation, '__call__'):
            inference = activation(inference)
        else:
            raise ValueError("Invalid Activation.")

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights.
    inference.scope = scope
    inference.W = W
    inference.b = b

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def conv_3d_transpose(incoming, nb_filter, filter_size, output_shape,
                      strides=1, padding='same', activation='linear',
                      bias=True, weights_init='uniform_scaling',
                      bias_init='zeros', regularizer=None, weight_decay=0.001,
                      trainable=True, restore=True, reuse=False, scope=None,
                      name="Conv3DTranspose"):

    """ Convolution 3D Transpose.

    This operation is sometimes called "deconvolution" after (Deconvolutional
    Networks)[http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf], but is
    actually the transpose (gradient) of `conv_3d` rather than an actual
    deconvolution.

    Input:
        5-D Tensor [batch, depth, height, width, in_channels].

    Output:
        5-D Tensor [batch, new depth, new height, new width, nb_filter].

    Arguments:
        incoming: `Tensor`. Incoming 5-D Tensor.
        nb_filter: `int`. The number of convolutional filters.
        filter_size: `int` or `list of int`. Size of filters.
        output_shape: `list of int`. Dimensions of the output tensor.
            Can optionally include the number of conv filters.
            [new depth, new height, new width, nb_filter] or [new depth, new height, new width].
        strides: `int` or list of `int`. Strides of conv operation.
            Default: [1 1 1 1 1].
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'truncated_normal'.
        bias_init: `str` (name) or `Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'Conv2DTranspose'.

    Attributes:
        scope: `Scope`. This layer scope.
        W: `Variable`. Variable representing filter weights.
        b: `Variable`. Variable representing biases.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 5, "Incoming Tensor shape must be 5-D"

    filter_size = utils.autoformat_filter_conv3d(filter_size,
                                                 nb_filter,
                                                 input_shape[-1])
    strides = utils.autoformat_stride_3d(strides)
    padding = utils.autoformat_padding(padding)

    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:
        name = scope.name

        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        W_regul = None
        if regularizer:
            W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
        W = vs.variable('W', shape=filter_size,
                        regularizer=W_regul, initializer=W_init,
                        trainable=trainable, restore=restore)
        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)

        b = None
        if bias:
            if isinstance(bias_init, str):
                bias_init = initializations.get(bias_init)()
            b = vs.variable('b', shape=nb_filter, initializer=bias_init,
                            trainable=trainable, restore=restore)
            # Track per layer variables
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b)

        # Determine the complete shape of the output tensor.
        batch_size = tf.gather(tf.shape(incoming), tf.constant([0]))
        if len(output_shape) == 3:
            output_shape = output_shape + [nb_filter]
        elif len(output_shape) != 4:
            raise Exception("output_shape length error: "
                            + str(len(output_shape))
                            + ", only a length of 3 or 4 is supported.")
        complete_out_shape = tf.concat([batch_size, tf.constant(output_shape)], 0)

        inference = tf.nn.conv3d_transpose(incoming, W, complete_out_shape,
                                           strides, padding)

        # Reshape tensor so its shape is correct.
        inference.set_shape([None] + output_shape)

        if b: inference = tf.nn.bias_add(inference, b)

        if isinstance(activation, str):
            inference = activations.get(activation)(inference)
        elif hasattr(activation, '__call__'):
            inference = activation(inference)
        else:
            raise ValueError("Invalid Activation.")

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights.
    inference.scope = scope
    inference.W = W
    inference.b = b

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def max_pool_3d(incoming, kernel_size, strides=1, padding='same',
                name="MaxPool3D"):
    """ Max Pooling 3D.

    Input:
        5-D Tensor [batch, depth, rows, cols, channels].

    Output:
        5-D Tensor [batch, pooled depth, pooled rows, pooled cols, in_channels].

    Arguments:
        incoming: `Tensor`. Incoming 5-D Layer.
        kernel_size: 'int` or `list of int`. Pooling kernel size.Must have kernel_size[0] = kernel_size[1] = 1
        strides: 'int` or `list of int`. Strides of conv operation.Must have strides[0] = strides[4] = 1.
            Default: [1 1 1 1 1]
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        name: A name for this layer (optional). Default: 'MaxPool3D'.

    Attributes:
        scope: `Scope`. This layer scope.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 5, "Incoming Tensor shape must be 5-D"

    kernel = utils.autoformat_kernel_3d(kernel_size)
    strides = utils.autoformat_stride_3d(strides)
    padding = utils.autoformat_padding(padding)

    with tf.name_scope(name) as scope:
        inference = tf.nn.max_pool3d(incoming, kernel, strides, padding)

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights
    inference.scope = scope

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def avg_pool_3d(incoming, kernel_size, strides=None, padding='same',
                name="AvgPool3D"):
    """ Average Pooling 3D.

    Input:
        5-D Tensor [batch, depth, rows, cols, channels].

    Output:
        5-D Tensor [batch, pooled depth, pooled rows, pooled cols, in_channels].

    Arguments:
        incoming: `Tensor`. Incoming 5-D Layer.
        kernel_size: 'int` or `list of int`. Pooling kernel size.Must have kernel_size[0] = kernel_size[1] = 1
        strides: 'int` or `list of int`. Strides of conv operation.Must have strides[0] = strides[4] = 1.
            Default: [1 1 1 1 1]
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        name: A name for this layer (optional). Default: 'AvgPool3D'.

    Attributes:
        scope: `Scope`. This layer scope.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 5, "Incoming Tensor shape must be 5-D"

    kernel = utils.autoformat_kernel_3d(kernel_size)
    strides = utils.autoformat_stride_3d(strides)
    padding = utils.autoformat_padding(padding)

    with tf.name_scope(name) as scope:
        inference = tf.nn.avg_pool3d(incoming, kernel, strides, padding)

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights
    inference.scope = scope

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def global_max_pool(incoming, name="GlobalMaxPool"):
    """ Global Max Pooling.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        2-D Tensor [batch, pooled dim]

    Arguments:
        incoming: `Tensor`. Incoming 4-D Tensor.
        name: A name for this layer (optional). Default: 'GlobalMaxPool'.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"

    with tf.name_scope(name):
        inference = tf.reduce_max(incoming, [1, 2])

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def global_avg_pool(incoming, name="GlobalAvgPool"):
    """ Global Average Pooling.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        2-D Tensor [batch, pooled dim]

    Arguments:
        incoming: `Tensor`. Incoming 4-D Tensor.
        name: A name for this layer (optional). Default: 'GlobalAvgPool'.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"

    with tf.name_scope(name):
        inference = tf.reduce_mean(incoming, [1, 2])

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def residual_block(incoming, nb_blocks, out_channels, downsample=False,
                   downsample_strides=2, activation='relu', batch_norm=True,
                   bias=True, weights_init='variance_scaling',
                   bias_init='zeros', regularizer='L2', weight_decay=0.0001,
                   trainable=True, restore=True, reuse=False, scope=None,
                   name="ResidualBlock"):
    """ Residual Block.

    A residual block as described in MSRA's Deep Residual Network paper.
    Full pre-activation architecture is used here.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        4-D Tensor [batch, new height, new width, nb_filter].

    Arguments:
        incoming: `Tensor`. Incoming 4-D Layer.
        nb_blocks: `int`. Number of layer blocks.
        out_channels: `int`. The number of convolutional filters of the
            convolution layers.
        downsample: `bool`. If True, apply downsampling using
            'downsample_strides' for strides.
        downsample_strides: `int`. The strides to use when downsampling.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        batch_norm: `bool`. If True, apply batch normalization.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'uniform_scaling'.
        bias_init: `str` (name) or `tf.Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'ShallowBottleneck'.

    References:
        - Deep Residual Learning for Image Recognition. Kaiming He, Xiangyu
            Zhang, Shaoqing Ren, Jian Sun. 2015.
        - Identity Mappings in Deep Residual Networks. Kaiming He, Xiangyu
            Zhang, Shaoqing Ren, Jian Sun. 2015.

    Links:
        - [http://arxiv.org/pdf/1512.03385v1.pdf]
            (http://arxiv.org/pdf/1512.03385v1.pdf)
        - [Identity Mappings in Deep Residual Networks]
            (https://arxiv.org/pdf/1603.05027v2.pdf)

    """
    resnet = incoming
    in_channels = incoming.get_shape().as_list()[-1]

    # Variable Scope fix for older TF
    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:

        name = scope.name #TODO

        for i in range(nb_blocks):

            identity = resnet

            if not downsample:
                downsample_strides = 1

            if batch_norm:
                resnet = tflearn.batch_normalization(resnet)
            resnet = tflearn.activation(resnet, activation)

            resnet = conv_2d(resnet, out_channels, 3,
                             downsample_strides, 'same', 'linear',
                             bias, weights_init, bias_init,
                             regularizer, weight_decay, trainable,
                             restore)

            if batch_norm:
                resnet = tflearn.batch_normalization(resnet)
            resnet = tflearn.activation(resnet, activation)

            resnet = conv_2d(resnet, out_channels, 3, 1, 'same',
                             'linear', bias, weights_init,
                             bias_init, regularizer, weight_decay,
                             trainable, restore)

            # Downsampling
            if downsample_strides > 1:
                identity = tflearn.avg_pool_2d(identity, downsample_strides,
                                               downsample_strides)

            # Projection to new dimension
            if in_channels != out_channels:
                ch = (out_channels - in_channels)//2
                identity = tf.pad(identity,
                                  [[0, 0], [0, 0], [0, 0], [ch, ch]])
                in_channels = out_channels

            resnet = resnet + identity

    return resnet


def residual_bottleneck(incoming, nb_blocks, bottleneck_size, out_channels,
                        downsample=False, downsample_strides=2,
                        activation='relu', batch_norm=True, bias=True,
                        weights_init='variance_scaling', bias_init='zeros',
                        regularizer='L2', weight_decay=0.0001,
                        trainable=True, restore=True, reuse=False, scope=None,
                        name="ResidualBottleneck"):
    """ Residual Bottleneck.

    A residual bottleneck block as described in MSRA's Deep Residual Network
    paper. Full pre-activation architecture is used here.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        4-D Tensor [batch, new height, new width, nb_filter].

    Arguments:
        incoming: `Tensor`. Incoming 4-D Layer.
        nb_blocks: `int`. Number of layer blocks.
        bottleneck_size: `int`. The number of convolutional filter of the
            bottleneck convolutional layer.
        out_channels: `int`. The number of convolutional filters of the
            layers surrounding the bottleneck layer.
        downsample: `bool`. If True, apply downsampling using
            'downsample_strides' for strides.
        downsample_strides: `int`. The strides to use when downsampling.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        batch_norm: `bool`. If True, apply batch normalization.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'uniform_scaling'.
        bias_init: `str` (name) or `tf.Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'DeepBottleneck'.

    References:
        - Deep Residual Learning for Image Recognition. Kaiming He, Xiangyu
            Zhang, Shaoqing Ren, Jian Sun. 2015.
        - Identity Mappings in Deep Residual Networks. Kaiming He, Xiangyu
            Zhang, Shaoqing Ren, Jian Sun. 2015.

    Links:
        - [http://arxiv.org/pdf/1512.03385v1.pdf]
            (http://arxiv.org/pdf/1512.03385v1.pdf)
        - [Identity Mappings in Deep Residual Networks]
            (https://arxiv.org/pdf/1603.05027v2.pdf)

    """
    resnet = incoming
    in_channels = incoming.get_shape().as_list()[-1]

    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:

        name = scope.name #TODO

        for i in range(nb_blocks):

            identity = resnet

            if not downsample:
                downsample_strides = 1

            if batch_norm:
                resnet = tflearn.batch_normalization(resnet)
            resnet = tflearn.activation(resnet, activation)

            resnet = conv_2d(resnet, bottleneck_size, 1,
                             downsample_strides, 'valid',
                             'linear', bias, weights_init,
                             bias_init, regularizer, weight_decay,
                             trainable, restore)

            if batch_norm:
                resnet = tflearn.batch_normalization(resnet)
            resnet = tflearn.activation(resnet, activation)

            resnet = conv_2d(resnet, bottleneck_size, 3, 1, 'same',
                             'linear', bias, weights_init,
                             bias_init, regularizer, weight_decay,
                             trainable, restore)

            resnet = conv_2d(resnet, out_channels, 1, 1, 'valid',
                             activation, bias, weights_init,
                             bias_init, regularizer, weight_decay,
                             trainable, restore)

            # Downsampling
            if downsample_strides > 1:
                identity = tflearn.avg_pool_2d(identity, downsample_strides,
                                               downsample_strides)

            # Projection to new dimension
            if in_channels != out_channels:
                ch = (out_channels - in_channels)//2
                identity = tf.pad(identity,
                                  [[0, 0], [0, 0], [0, 0], [ch, ch]])
                in_channels = out_channels

                resnet = resnet + identity
                resnet = tflearn.activation(resnet, activation)

    return resnet


def highway_conv_2d(incoming, nb_filter, filter_size, strides=1, padding='same',
                    activation='linear', weights_init='uniform_scaling',
                    bias_init='zeros', regularizer=None, weight_decay=0.001,
                    trainable=True, restore=True, reuse=False, scope=None,
                    name="HighwayConv2D"):
    """ Highway Convolution 2D.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        4-D Tensor [batch, new height, new width, nb_filter].

    Arguments:
        incoming: `Tensor`. Incoming 4-D Tensor.
        nb_filter: `int`. The number of convolutional filters.
        filter_size: 'int` or `list of int`. Size of filters.
        strides: 'int` or `list of int`. Strides of conv operation.
            Default: [1 1 1 1].
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'truncated_normal'.
        bias_init: `str` (name) or `Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'Conv2D'.

    Attributes:
        scope: `Scope`. This layer scope.
        W: `Variable`. Variable representing filter weights.
        W_T: `Variable`. Variable representing gate weights.
        b: `Variable`. Variable representing biases.
        b_T: `Variable`. Variable representing gate biases.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"
    filter_size = utils.autoformat_filter_conv2d(filter_size,
                                                 input_shape[-1],
                                                 nb_filter)
    strides = utils.autoformat_kernel_2d(strides)
    padding = utils.autoformat_padding(padding)

    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:

        name = scope.name

        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        W_regul = None
        if regularizer:
            W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
        W = vs.variable('W', shape=filter_size, regularizer=W_regul,
                        initializer=W_init, trainable=trainable,
                        restore=restore)
        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)

        bias_init = initializations.get(bias_init)()
        b = vs.variable('b', shape=nb_filter, initializer=bias_init,
                        trainable=trainable, restore=restore)
        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b)

        # Weight and bias for the transform gate
        W_T = vs.variable('W_T', shape=nb_filter,
                          regularizer=None, initializer=W_init,
                          trainable=trainable, restore=restore)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' +
                             name, W_T)

        b_T = vs.variable('b_T', shape=nb_filter,
                          initializer=tf.constant_initializer(-3),
                          trainable=trainable, restore=restore)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' +
                             name, b_T)

        if isinstance(activation, str):
            activation = activations.get(activation)
        elif hasattr(activation, '__call__'):
            activation = activation
        else:
            raise ValueError("Invalid Activation.")

        # Shared convolution for gating
        convolved = tf.nn.conv2d(incoming, W, strides, padding)
        H = activation(convolved + b)
        T = tf.sigmoid(tf.multiply(convolved, W_T) + b_T)
        C = tf.subtract(1.0, T)
        inference = tf.add(tf.multiply(H, T), tf.multiply(convolved, C))

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights.
    inference.scope = scope
    inference.W = W
    inference.W_T = W_T
    inference.b = b
    inference.b_T = b_T

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def highway_conv_1d(incoming, nb_filter, filter_size, strides=1, padding='same',
                    activation='linear', weights_init='uniform_scaling',
                    bias_init='zeros', regularizer=None, weight_decay=0.001,
                    trainable=True, restore=True, reuse=False, scope=None,
                    name="HighwayConv1D"):
    """ Highway Convolution 1D.

    Input:
        3-D Tensor [batch, steps, in_channels].

    Output:
        3-D Tensor [batch, new steps, nb_filters].

    Arguments:
        incoming: `Tensor`. Incoming 3-D Tensor.
        nb_filter: `int`. The number of convolutional filters.
        filter_size: 'int` or `list of int`. Size of filters.
        strides: 'int` or `list of int`. Strides of conv operation.
            Default: [1 1 1 1].
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'truncated_normal'.
        bias_init: `str` (name) or `Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'HighwayConv1D'.

    Attributes:
        scope: `Scope`. This layer scope.
        W: `Variable`. Variable representing filter weights.
        W_T: `Variable`. Variable representing gate weights.
        b: `Variable`. Variable representing biases.
        b_T: `Variable`. Variable representing gate biases.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 3, "Incoming Tensor shape must be 3-D"
    filter_size = utils.autoformat_filter_conv2d(filter_size,
                                                 input_shape[-1],
                                                 nb_filter)
    # filter_size = [1, filter_size[1], 1, 1]
    filter_size[1] = 1
    strides = utils.autoformat_kernel_2d(strides)
    # strides = [1, strides[1], 1, 1]
    strides[1] = 1
    padding = utils.autoformat_padding(padding)

    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:

        name = scope.name

        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        W_regul = None
        if regularizer:
            W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
        W = vs.variable('W', shape=filter_size,
                        regularizer=W_regul, initializer=W_init,
                        trainable=trainable, restore=restore)
        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)

        bias_init = initializations.get(bias_init)()
        b = vs.variable('b', shape=nb_filter, initializer=bias_init,
                        trainable=trainable, restore=restore)
        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b)

        # Weight and bias for the transform gate
        W_T = vs.variable('W_T', shape=nb_filter,
                        regularizer=None, initializer=W_init,
                        trainable=trainable, restore=restore)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W_T)

        b_T = vs.variable('b_T', shape=nb_filter,
                          initializer=tf.constant_initializer(-3),
                          trainable=trainable, restore=restore)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b_T)

        if isinstance(activation, str):
            activation = activations.get(activation)
        elif hasattr(activation, '__call__'):
            activation = activation
        else:
            raise ValueError("Invalid Activation.")

        # Adding dummy dimension to fit with Tensorflow conv2d
        inference = tf.expand_dims(incoming, 2)
        # Shared convolution for gating
        convolved = tf.nn.conv2d(inference, W, strides, padding)
        H = activation(tf.squeeze(convolved + b, [2]))
        T = tf.sigmoid(tf.squeeze(tf.multiply(convolved, W_T) + b_T, [2]))
        C = tf.subtract(1.0, T)
        Q = tf.multiply(H, T)
        R = tf.multiply(tf.squeeze(convolved, [2]), C)
        inference = tf.add(Q, R)

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights.
    inference.scope = scope
    inference.W = W
    inference.W_T = W_T
    inference.b = b
    inference.b_T = b_T

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference
