from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf

import tflearn

from tflearn import utils
from tflearn import variables as va
from tflearn import activations
from tflearn import initializations
from tflearn import losses


def input_data(shape=None, placeholder=None, dtype=tf.float32,
               data_preprocessing=None, data_augmentation=None,
               name="InputData"):
    """ Input Data.

    `input_data` is used as a data entry (placeholder) of a network.
    This placeholder will be feeded with data when training

    This layer is used to keep track of the network inputs, by adding the
    placeholder to INPUTS graphkey. TFLearn training functions may retrieve
    those variables to setup the network training process.

    Input:
        List of `int` (Shape), to create a new placeholder.
            Or
        `Tensor` (Placeholder), to use an existing placeholder.

    Output:
        Placeholder Tensor with given shape.

    Arguments:
        shape: list of `int`. An array or tuple representing input data shape.
            It is required if no placeholder provided. First element should
            be 'None' (representing batch size), if not provided, it will be
            added automatically.
        placeholder: A Placeholder to use for feeding this layer (optional).
            If not specified, a placeholder will be automatically created.
            You can retrieve that placeholder through graph key: 'INPUTS',
            or the 'placeholder' attribute of this function's returned tensor.
        dtype: `tf.type`, Placeholder data type (optional). Default: float32.
        data_preprocessing: A `DataPreprocessing` subclass object to manage
            real-time data pre-processing when training and predicting (such
            as zero center data, std normalization...).
        data_augmentation: `DataAugmentation`. A `DataAugmentation` subclass
            object to manage real-time data augmentation while training (
            such as random image crop, random image flip, random sequence
            reverse...).
        name: `str`. A name for this layer (optional).

    """
    if not shape and not placeholder:
        raise Exception("`shape` or `placeholder` argument is required.")

    if not placeholder:
        # Add 'None' if missing
        assert shape is not None, "A shape or a placeholder must be provided."
        if len(shape) > 1:
            if shape[0] is not None:
                shape = list(shape)
                shape = [None] + shape
        with tf.name_scope(name) as scope:
            placeholder = tf.placeholder(shape=shape, dtype=dtype, name="X")

    # Keep track of inputs
    tf.add_to_collection(tf.GraphKeys.INPUTS, placeholder)
    # Keep track of data preprocessing and augmentation
    tf.add_to_collection(tf.GraphKeys.DATA_PREP, data_preprocessing)
    tf.add_to_collection(tf.GraphKeys.DATA_AUG, data_augmentation)

    return placeholder


def fully_connected(incoming, n_units, activation='linear', bias=True,
                    weights_init='truncated_normal', bias_init='zeros',
                    regularizer=None, weight_decay=0.001, trainable=True,
                    restore=True, name="FullyConnected"):
    """ Fully Connected.

    A fully connected layer.

    Input:
        (2+)-D Tensor [samples, input dim]. If not 2D, input will be flatten.

    Output:
        2D Tensor [samples, n_units].

    Arguments:
        incoming: `Tensor`. Incoming (2+)D Tensor.
        n_units: `int`, number of units for this layer.
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
       name: A name for this layer (optional). Default: 'FullyConnected'.

    Attributes:
        scope: `Scope`. This layer scope.
        W: `Tensor`. Variable representing units weights.
        b: `Tensor`. Variable representing biases.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) > 1, "Incoming Tensor shape must be at least 2-D"
    n_inputs = int(np.prod(input_shape[1:]))

    # Build variables and inference.
    with tf.name_scope(name) as scope:

        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        W_regul = None
        if regularizer:
            W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
        W = va.variable(scope + 'W', shape=[n_inputs, n_units],
                        regularizer=W_regul, initializer=W_init,
                        trainable=trainable, restore=restore)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope, W)

        b = None
        if bias:
            b_init = initializations.get(bias_init)()
            b = va.variable(scope + 'b', shape=[n_units],
                            initializer=b_init, trainable=trainable,
                            restore=restore)
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope, b)

        inference = incoming
        # If input is not 2d, flatten it.
        if len(input_shape) > 2:
            inference = tf.reshape(inference, [-1, n_inputs])

        inference = tf.matmul(inference, W)
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

    return inference


def dropout(incoming, keep_prob, name="Dropout"):
    """ Dropout.

    Outputs the input element scaled up by `1 / keep_prob`. The scaling is so
    that the expected sum is unchanged.

    Arguments:
        incoming : A `Tensor`. The incoming tensor.
        keep_prob : A float representing the probability that each element
            is kept.
        name : A name for this layer (optional).

    References:
        Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
        N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever & R. Salakhutdinov,
        (2014), Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.

    Links:
      [https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf]
        (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)

    """

    with tf.name_scope(name) as scope:

        inference = incoming

        def apply_dropout():
            if type(inference) in [list, np.array]:
                for x in inference:
                    x = tf.nn.dropout(x, keep_prob)
                return inference
            else:
                return tf.nn.dropout(inference, keep_prob)

        is_training = tflearn.get_training_mode()
        inference = tf.cond(is_training, apply_dropout, lambda: inference)

    return inference


def custom_layer(incoming, custom_fn, **kwargs):
    """ Custom Layer.

    A custom layer that can apply any operations to the incoming Tensor or
    list of `Tensor`. The custom function can be pass as a parameter along
    with its parameters.

    Arguments:
        incoming : A `Tensor` or list of `Tensor`. Incoming tensor.
        custom_fn : A custom `function`, to apply some ops on incoming tensor.
        **kwargs: Some custom parameters that custom function might need.

    """
    name = "CustomLayer"
    if 'name' in kwargs:
        name = kwargs['name']
    with tf.name_scope(name):
        inference = custom_fn(incoming, **kwargs)

    return inference


def reshape(incoming, new_shape, name="Reshape"):
    """ Reshape.

    A layer that reshape the incoming layer tensor output to the desired shape.

    Arguments:
        incoming: A `Tensor`. The incoming tensor.
        new_shape: A list of `int`. The desired shape.
        name: A name for this layer (optional).

    """

    with tf.name_scope(name) as scope:
        inference = incoming
        if isinstance(inference, list):
            inference = tf.concat(inference, 0)
            inference = tf.cast(inference, tf.float32)
        inference = tf.reshape(inference, shape=new_shape)

    inference.scope = scope

    return inference


def flatten(incoming, name="Flatten"):
    """ Flatten.

    Flatten the incoming Tensor.

    Input:
        (2+)-D `Tensor`.

    Output:
        2-D `Tensor` [batch, flatten_dims].

    Arguments:
        incoming: `Tensor`. The incoming tensor.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) > 1, "Incoming Tensor shape must be at least 2-D"
    dims = int(np.prod(input_shape[1:]))
    return reshape(incoming, [-1, dims], name)


def activation(incoming, activation='linear'):

    """ Activation.

    Apply given activation to incoming tensor.

    Arguments:
        incoming: A `Tensor`. The incoming tensor.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.

    """

    if isinstance(activation, str):
        return activations.get(activation)(incoming)
    elif hasattr(incoming, '__call__'):
        return activation(incoming)
    else:
        raise ValueError('Unknown activation type.')


def single_unit(incoming, activation='linear', bias=True, trainable=True,
                restore=True, name="Linear"):
    """ Single Unit.

    A single unit (Linear) Layer.

    Input:
        1-D Tensor [samples]. If not 2D, input will be flatten.

    Output:
        1-D Tensor [samples].

    Arguments:
        incoming: `Tensor`. Incoming Tensor.
        activation: `str` (name) or `function`. Activation applied to this
            layer (see tflearn.activations). Default: 'linear'.
        bias: `bool`. If True, a bias is used.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        name: A name for this layer (optional). Default: 'Dense'.

    Attributes:
        W: `Tensor`. Variable representing weight.
        b: `Tensor`. Variable representing bias.

    """
    input_shape = utils.get_incoming_shape(incoming)
    n_inputs = int(np.prod(input_shape[1:]))

    # Build variables and inference.
    with tf.name_scope(name) as scope:

        W = va.variable(scope + 'W', shape=[n_inputs],
                        initializer=tf.constant_initializer(np.random.randn()),
                        trainable=trainable, restore=restore)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope, W)

        b = None
        if bias:
            b = va.variable(scope + 'b', shape=[n_inputs],
                            initializer=tf.constant_initializer(np.random.randn()),
                            trainable=trainable, restore=restore)
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope, b)

        inference = incoming
        # If input is not 2d, flatten it.
        if len(input_shape) > 1:
            inference = tf.reshape(inference, [-1])

        inference = tf.mul(inference, W)
        if b: inference = tf.add(inference, b)

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

    return inference
    
    
def highway(incoming, n_units, activation='linear', transform_dropout=None,
            weights_init='truncated_normal', bias_init='zeros',
            regularizer=None, weight_decay=0.001, trainable=True,
            restore=True, name="FullyConnectedHighway"):
    """ Fully Connected Highway.

    A fully connected highway network layer, with some inspiration from
    [https://github.com/fomorians/highway-fcn](https://github.com/fomorians/highway-fcn).

    Input:
        (2+)-D Tensor [samples, input dim]. If not 2D, input will be flatten.

    Output:
        2D Tensor [samples, n_units].

    Arguments:
        incoming: `Tensor`. Incoming (2+)D Tensor.
        n_units: `int`, number of units for this layer.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        transform_dropout: `float`: Keep probability on the highway transform gate.
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
         name: A name for this layer (optional). Default: 'FullyConnectedHighway'.

    Attributes:
        scope: `Scope`. This layer scope.
        W: `Tensor`. Variable representing units weights.
        W_t: `Tensor`. Variable representing units weights for transform gate.
        b: `Tensor`. Variable representing biases.
        b_t: `Tensor`. Variable representing biases for transform gate.
        
    Links:
        [https://arxiv.org/abs/1505.00387](https://arxiv.org/abs/1505.00387)

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) > 1, "Incoming Tensor shape must be at least 2-D"
    n_inputs = int(np.prod(input_shape[1:]))

    # Build variables and inference.
    with tf.name_scope(name) as scope:

        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        W_regul = None
        if regularizer:
            W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
        W = va.variable(scope + 'W', shape=[n_inputs, n_units],
                        regularizer=W_regul, initializer=W_init,
                        trainable=trainable, restore=restore)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope, W)

        b = None
        b_init = initializations.get(bias_init)()
        b = va.variable(scope + 'b', shape=[n_units],
                        initializer=b_init, trainable=trainable,
                        restore=restore)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope, b)
            
        #weight and bias for the transform gate
        with tf.name_scope('transform_gate') as transform_gate:
            W_T = va.variable(transform_gate + 'W', shape=[n_inputs, n_units],
                            regularizer=None, initializer=W_init,
                            trainable=trainable, restore=restore)
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + transform_gate, W_T) 
            
            b_T = va.variable(transform_gate + 'b', shape=[n_units],
                            initializer=tf.constant_initializer(-1), trainable=trainable,
                            restore=restore)
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + transform_gate, b_T)

        # If input is not 2d, flatten it.
        if len(input_shape) > 2:
            incoming = tf.reshape(incoming, [-1, n_inputs])

        if isinstance(activation, str):
            activation = activations.get(activation)
        elif hasattr(activation, '__call__'):
            activation = activation
        else:
            raise ValueError("Invalid Activation.")
            
        H = activation(tf.matmul(incoming, W) + b)
        T = tf.sigmoid(tf.matmul(incoming, W_T) + b_T)
        if transform_dropout:
            T = dropout(T, transform_dropout)
        C = tf.sub(1.0, T)

        inference = tf.add(tf.mul(H, T), tf.mul(incoming, C))

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights.
    inference.scope = scope
    inference.W = W
    inference.W_t = W_T
    inference.b = b
    inference.b_t = b_T

    return inference
