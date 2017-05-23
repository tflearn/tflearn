from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import standard_ops

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

    This layer is used for inputting (aka. feeding) data to a network.
    A TensorFlow placeholder will be used if it is supplied,
    otherwise a new placeholder will be created with the given shape.

    Either a shape or placeholder must be provided, otherwise an
    exception will be raised.

    Furthermore, the placeholder is added to TensorFlow collections
    so it can be retrieved using tf.get_collection(tf.GraphKeys.INPUTS)
    as well as tf.GraphKeys.LAYER_TENSOR + '/' + name. Similarly for
    the data preprocessing and augmentation objects which are stored in
    the collections with tf.GraphKeys.DATA_PREP and tf.GraphKeys.DATA_AUG.
    This allows other parts of TFLearn to easily retrieve and use these
    objects by referencing these graph-keys.

    Input:
        List of `int` (Shape), to create a new placeholder.
            Or
        `Tensor` (Placeholder), to use an existing placeholder.

    Output:
        Placeholder Tensor with given shape.

    Arguments:
        shape: list of `int`. An array or tuple representing input data shape.
            It is required if no placeholder is provided. First element should
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

    # We need either a placeholder or a shape, otherwise raise an exception.
    if placeholder is None:
        if shape is None:
            raise Exception("Either a `shape` or `placeholder` argument is required to consruct an input layer.")

        # We have a shape but no placeholder, so we must now create a placeholder.

        # Ensure the first element of shape is None by prepending None if necessary.
        # TODO: Why is there a len(shape)>1 condition? Please explain here.
        if len(shape) > 1 and shape[0] is not None:
            shape = list(shape)
            shape = [None] + shape

        # Create a new tf.placeholder with the given shape.
        with tf.name_scope(name):
            placeholder = tf.placeholder(shape=shape, dtype=dtype, name="X")

    # Store the placeholder object in TensorFlow collections so it can be
    # retrieved and used elsewhere.
    tf.add_to_collection(tf.GraphKeys.INPUTS, placeholder)
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, placeholder)

    # Store the objects for data-preprocessing and -augmentation
    # in TensorFlow collections so they can be retrieved and used elsewhere.
    tf.add_to_collection(tf.GraphKeys.DATA_PREP, data_preprocessing)
    tf.add_to_collection(tf.GraphKeys.DATA_AUG, data_augmentation)

    return placeholder


def fully_connected(incoming, n_units, activation='linear', bias=True,
                    weights_init='truncated_normal', bias_init='zeros',
                    regularizer=None, weight_decay=0.001, trainable=True,
                    restore=True, reuse=False, scope=None,
                    name="FullyConnected"):
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
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'FullyConnected'.

    Attributes:
        scope: `Scope`. This layer scope.
        W: `Tensor`. Variable representing units weights.
        b: `Tensor`. Variable representing biases.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) > 1, "Incoming Tensor shape must be at least 2-D"
    n_inputs = int(np.prod(input_shape[1:]))

    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:
        name = scope.name

        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        W_regul = None
        if regularizer:
            W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
        W = va.variable('W', shape=[n_inputs, n_units], regularizer=W_regul,
                        initializer=W_init, trainable=trainable,
                        restore=restore)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)

        b = None
        if bias:
            if isinstance(bias_init, str):
                bias_init = initializations.get(bias_init)()
            b = va.variable('b', shape=[n_units], initializer=bias_init,
                            trainable=trainable, restore=restore)
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b)

        inference = incoming
        # If input is not 2d, flatten it.
        if len(input_shape) > 2:
            inference = tf.reshape(inference, [-1, n_inputs])

        inference = tf.matmul(inference, W)
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


def dropout(incoming, keep_prob, noise_shape=None, name="Dropout"):
    """ Dropout.

    Outputs the input element scaled up by `1 / keep_prob`. The scaling is so
    that the expected sum is unchanged.

    By default, each element is kept or dropped independently. If noise_shape
    is specified, it must be broadcastable to the shape of x, and only dimensions
    with noise_shape[i] == shape(x)[i] will make independent decisions. For
    example, if shape(x) = [k, l, m, n] and noise_shape = [k, 1, 1, n], each
    batch and channel component will be kept independently and each row and column
    will be kept or not kept together.

    Arguments:
        incoming : A `Tensor`. The incoming tensor.
        keep_prob : A float representing the probability that each element
            is kept.
        noise_shape : A 1-D Tensor of type int32, representing the shape for
            randomly generated keep/drop flags.
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
                    x = tf.nn.dropout(x, keep_prob, noise_shape)
                return inference
            else:
                return tf.nn.dropout(inference, keep_prob, noise_shape)

        is_training = tflearn.get_training_mode()
        inference = tf.cond(is_training, apply_dropout, lambda: inference)

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

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
            inference = tf.concat(0, inference)
            inference = tf.cast(inference, tf.float32)
        inference = tf.reshape(inference, shape=new_shape)

    inference.scope = scope

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

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
    x = reshape(incoming, [-1, dims], name)

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, x)

    return x


def activation(incoming, activation='linear', name='activation'):

    """ Activation.

    Apply given activation to incoming tensor.

    Arguments:
        incoming: A `Tensor`. The incoming tensor.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.

    """

    if isinstance(activation, str):
        x = activations.get(activation)(incoming)
    elif hasattr(incoming, '__call__'):
        x = activation(incoming)
    else:
        raise ValueError('Unknown activation type.')

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, x)

    return x


def single_unit(incoming, activation='linear', bias=True, trainable=True,
                restore=True, reuse=False, scope=None, name="Linear"):
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
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'Linear'.

    Attributes:
        W: `Tensor`. Variable representing weight.
        b: `Tensor`. Variable representing bias.

    """
    input_shape = utils.get_incoming_shape(incoming)
    n_inputs = int(np.prod(input_shape[1:]))

    # Build variables and inference.
    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:
        name = scope.name

        W = va.variable('W', shape=[n_inputs],
                        initializer=tf.constant_initializer(np.random.randn()),
                        trainable=trainable, restore=restore)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)

        b = None
        if bias:
            b = va.variable('b', shape=[n_inputs],
                            initializer=tf.constant_initializer(np.random.randn()),
                            trainable=trainable, restore=restore)
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b)

        inference = incoming
        # If input is not 2d, flatten it.
        if len(input_shape) > 1:
            inference = tf.reshape(inference, [-1])

        inference = tf.multiply(inference, W)
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

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def highway(incoming, n_units, activation='linear', transform_dropout=None,
            weights_init='truncated_normal', bias_init='zeros',
            regularizer=None, weight_decay=0.001, trainable=True,
            restore=True, reuse=False, scope=None,
            name="FullyConnectedHighway"):
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
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
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
    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:
        name = scope.name

        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        W_regul = None
        if regularizer:
            W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
        W = va.variable('W', shape=[n_inputs, n_units], regularizer=W_regul,
                        initializer=W_init, trainable=trainable,
                        restore=restore)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)

        if isinstance(bias_init, str):
            bias_init = initializations.get(bias_init)()
        b = va.variable('b', shape=[n_units], initializer=bias_init,
                        trainable=trainable, restore=restore)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b)

        # Weight and bias for the transform gate
        W_T = va.variable('W_T', shape=[n_inputs, n_units],
                          regularizer=None, initializer=W_init,
                          trainable=trainable, restore=restore)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W_T)

        b_T = va.variable('b_T', shape=[n_units],
                          initializer=tf.constant_initializer(-1),
                          trainable=trainable, restore=restore)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b_T)

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
        C = tf.subtract(1.0, T)

        inference = tf.add(tf.multiply(H, T), tf.multiply(incoming, C))

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights.
    inference.scope = scope
    inference.W = W
    inference.W_t = W_T
    inference.b = b
    inference.b_t = b_T

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def one_hot_encoding(target, n_classes, on_value=1.0, off_value=0.0,
                     name="OneHotEncoding"):
    """ One Hot Encoding.

    Transform numeric labels into a binary vector.

    Input:
        The Labels Placeholder.

    Output:
        2-D Tensor, The encoded labels.

    Arguments:
        target: `Placeholder`. The labels placeholder.
        n_classes: `int`. Total number of classes.
        on_value: `scalar`. A scalar defining the on-value.
        off_value: `scalar`. A scalar defining the off-value.
        name: A name for this layer (optional). Default: 'OneHotEncoding'.

    """

    with tf.name_scope(name):
        if target.dtype != dtypes.int64:
            target = standard_ops.to_int64(target)

        target = standard_ops.one_hot(target, n_classes,
                                      on_value=on_value,
                                      off_value=off_value)

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, target)

    return target


def time_distributed(incoming, fn, args=None, scope=None):
    """ Time Distributed.

    This layer applies a function to every timestep of the input tensor. The
    custom function first argument must be the input tensor at every timestep.
    Additional parameters for the custom function may be specified in 'args'
    argument (as a list).

    Examples:
        ```python
        # Applying a fully_connected layer at every timestep
        x = time_distributed(input_tensor, fully_connected, [64])

        # Using a conv layer at every timestep with a scope
        x = time_distributed(input_tensor, conv_2d, [64, 3], scope='tconv')
        ```

    Input:
        (3+)-D Tensor [samples, timestep, input_dim].

    Output:
        (3+)-D Tensor [samples, timestep, output_dim].

    Arguments:
        incoming: `Tensor`. The incoming tensor.
        fn: `function`. A function to apply at every timestep. This function
            first parameter must be the input tensor per timestep. Additional
            parameters may be specified in 'args' argument.
        args: `list`. A list of parameters to use with the provided function.
        scope: `str`. A scope to give to each timestep tensor. Useful when
            sharing weights. Each timestep tensor scope will be generated
            as 'scope'-'i' where i represents the timestep id. Note that your
            custom function will be required to have a 'scope' parameter.

    Returns:
        A Tensor.

    """
    if not args: args = list()
    assert isinstance(args, list), "'args' must be a list."

    if not isinstance(incoming, tf.Tensor):
        incoming = tf.transpose(tf.stack(incoming), [1, 0, 2])

    input_shape = utils.get_incoming_shape(incoming)
    timestep = input_shape[1]
    x = tf.unstack(incoming, axis=1)
    if scope:
        x = [fn(x[i], scope=scope+'-'+str(i), *args)
             for i in range(timestep)]
    else:
        x = [fn(x[i], *args) for i in range(timestep)]

    x = list(map(lambda t: tf.reshape(t, [-1, 1]+utils.get_incoming_shape(t)[1:]), x))
    return tf.concat(x, 1)


def multi_target_data(name_list, shape, dtype=tf.float32):
    """ Multi Target Data.

    Create and concatenate multiple placeholders. To be used when a regression
    layer uses targets from different sources.

    Arguments:
        name_list: list of `str`. The names of the target placeholders.
        shape: list of `int`. The shape of the placeholders.
        dtype: `tf.type`, Placeholder data type (optional). Default: float32.

    Return:
        A `Tensor` of the concatenated placeholders.

    """
    placeholders = []
    for i in range(len(name_list)):
        with tf.name_scope(name_list[i]):
            p = tf.placeholder(shape=shape, dtype=dtype, name='Y')
        if p not in tf.get_collection(tf.GraphKeys.TARGETS):
            tf.add_to_collection(tf.GraphKeys.TARGETS, p)
        placeholders.append(p)

    return tf.concat(placeholders, axis=0)
