# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf

import  tflearn
from tflearn import utils
from tflearn import objectives
from tflearn import metrics
from tflearn import optimizers
from tflearn.helpers.trainer import TrainOp


def regression(incoming, placeholder=None, optimizer='adam',
               loss='categorical_crossentropy', metric='default',
               learning_rate=0.001, dtype=tf.float32, batch_size=64,
               shuffle_batches=True, trainable_vars=None, restore=True,
               op_name=None, name=None):
    """ Regression.

    Input:
        2-D Tensor Layer.

    Output:
        2-D Tensor Layer (Same as input).

    Arguments:
        incoming: `Tensor`. Incoming 2-D Tensor.
        placeholder: `Tensor`. This regression target (label) placeholder.
            If 'None' provided, a placeholder will be added automatically.
            You can retrieve that placeholder through graph key: 'TARGETS',
            or the 'placeholder' attribute of this function's returned tensor.
        optimizer: `str` (name), `Optimizer` or `function`. Optimizer to use.
            Default: 'sgd' (Stochastic Descent Gradient).
        loss: `str` (name) or `function`. Loss function used by this layer
            optimizer. Default: 'categorical_crossentropy'.
        metric: `str`, `Metric` or `function`. The metric to be used.
            Default: 'default' metric is 'accuracy'. To disable metric
            calculation, set it to 'None'.
        learning_rate: `float`. This layer optimizer's learning rate.
        dtype: `tf.types`. This layer placeholder type. Default: tf.float32.
        batch_size: `int`. Batch size of data to use for training. tflearn
            supports different batch size for every optimizers. Default: 64.
        shuffle_batches: `bool`. Shuffle or not this optimizer batches at
            every epoch. Default: True.
        trainable_vars: list of `Variable`. If specified, this regression will
            only update given variable weights. Else, all trainale variable
            are going to be updated.
        restore: `bool`. If False, variables related to optimizers such
            as moving averages will not be restored when loading a
            pre-trained model.
        op_name: A name for this layer optimizer (optional).
            Default: optimizer op name.
        name: A name for this layer's placeholder scope.

    Attributes:
        placeholder: `Tensor`. Placeholder for feeding labels.

    """

    input_shape = utils.get_incoming_shape(incoming)

    if not placeholder:
        pscope = "TargetsData" if not name else name
        with tf.name_scope(pscope):
            placeholder = tf.placeholder(shape=input_shape, dtype=dtype, name="Y")

    tf.add_to_collection(tf.GraphKeys.TARGETS, placeholder)

    step_tensor = None
    # Building Optimizer
    if isinstance(optimizer, str):
        _opt = optimizers.get(optimizer)(learning_rate)
        op_name = op_name if op_name else type(_opt).__name__
        _opt.build()
        optimizer = _opt.get_tensor()
    elif isinstance(optimizer, optimizers.Optimizer):
        op_name = op_name if op_name else type(optimizer).__name__
        if optimizer.has_decay:
            step_tensor = tf.Variable(0., name="Training_step",
                                      trainable=False)
        optimizer.build(step_tensor)
        optimizer = optimizer.get_tensor()
    elif hasattr(optimizer, '__call__'):
        try:
            optimizer, step_tensor = optimizer(learning_rate)
        except Exception as e:
            print(e.message)
            print("Reminder: Custom Optimizer function must return (optimizer, "
                  "step_tensor) and take one argument: 'learning_rate'. "
                  "Note that returned step_tensor can be 'None' if no decay.")
            exit()
    elif not isinstance(optimizer, tf.train.Optimizer):
        raise ValueError("Invalid Optimizer type.")

    inputs = tf.get_collection(tf.GraphKeys.INPUTS)
    #inputs = tf.concat(0, utils.get_tensor_parents_placeholders(incoming))

    # Building metric
    # No auto accuracy for linear regression
    if len(input_shape) == 1 and metric == 'default':
        metric = None
    if metric is not None:
        # Default metric is accuracy
        if metric == 'default': metric = 'accuracy'
        if isinstance(metric, str):
            metric = metrics.get(metric)()
            metric.build(incoming, placeholder, inputs)
            metric = metric.get_tensor()
        elif isinstance(metric, metrics.Metric):
            metric.build(incoming, placeholder, inputs)
            metric = metric.get_tensor()
        elif hasattr(metric, '__call__'):
            try:
                metric = metric(incoming, placeholder, inputs)
            except Exception as e:
                print(e.message)
                print('Reminder: Custom metric function arguments must be '
                      'define as follow: custom_metric(y_pred, y_true, x).')
                exit()
        elif not isinstance(metric, tf.Tensor):
            ValueError("Invalid Metric type.")

    # Building other ops (loss, training ops...)
    if isinstance(loss, str):
        loss = objectives.get(loss)(incoming, placeholder)
    # Check if function
    elif hasattr(loss, '__call__'):
        try:
            loss = loss(incoming, placeholder)
        except Exception as e:
            print(e.message)
            print('Reminder: Custom loss function arguments must be define as '
                  'follow: custom_loss(y_pred, y_true).')
            exit()
    elif not isinstance(loss, tf.Tensor):
        raise ValueError("Invalid Loss type.")

    tr_vars = trainable_vars
    if not tr_vars:
        tr_vars = tf.trainable_variables()

    if not restore:
        tf.add_to_collection(tf.GraphKeys.EXCL_RESTORE_VARS, 'moving_avg')
        tf.add_to_collection(tf.GraphKeys.EXCL_RESTORE_VARS,
                             optimizer._name + '/')

    tr_op = TrainOp(loss=loss,
                    optimizer=optimizer,
                    metric=metric,
                    trainable_vars=tr_vars,
                    batch_size=batch_size,
                    shuffle=shuffle_batches,
                    step_tensor=step_tensor,
                    name=op_name)

    tf.add_to_collection(tf.GraphKeys.TRAIN_OPS, tr_op)

    if not hasattr(incoming, '__len__'):
        incoming.placeholder = placeholder

    return incoming
