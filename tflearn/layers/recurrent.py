# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import nn_ops

from .. import utils
from .. import activations
from .. import initializations


# --------------------------
#  RNN Layers
# --------------------------


def simple_rnn(incoming, n_units, activation='sigmoid', dropout=None,
               bias=True, weights_init='truncated_normal', return_seq=False,
               return_states=False, initial_state=None, sequence_length=None,
               trainable=True, restore=True, name="SimpleRNN"):
    """ Simple RNN.

    Simple Recurrent Layer.

    Input:
        3-D Tensor [samples, timesteps, input dim].

    Output:
        if `return_seq`: 3-D Tensor [samples, timesteps, output dim].
        else: 2-D Tensor [samples, output dim].

    Arguments:
        incoming: `Tensor`. Incoming 3-D Tensor.
        n_units: `int`, number of units for this layer.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        dropout: `tuple` of `float`: (input_keep_prob, output_keep_prob). The
            input and output keep probability.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (See tflearn.initializations) Default: 'truncated_normal'.
        return_seq: `bool`. If True, returns the full sequence instead of
            last sequence output only.
        return_states: `bool`. If True, returns a tuple with output and
            states: (output, states).
        initial_state: `Tensor`. An initial state for the RNN.  This must be
            a tensor of appropriate type and shape [batch_size x cell.state_size].
        sequence_length: Specifies the length of each sequence in inputs.
            An int32 or int64 vector (tensor) size `[batch_size]`.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        name: `str`. A name for this layer (optional).

    """
    input_shape = utils.get_incoming_shape(incoming)
    W_init = weights_init
    if isinstance(weights_init, str):
        W_init = initializations.get(weights_init)()

    with tf.name_scope(name) as scope:
        cell = BasicRNNCell(n_units, activation, bias, W_init,
                            trainable, restore)
        out_cell = cell
        # Apply dropout
        if dropout:
            if type(dropout) in [tuple, list]:
                in_keep_prob = dropout[0]
                out_keep_prob = dropout[1]
            elif isinstance(dropout, float):
                in_keep_prob, out_keep_prob = dropout, dropout
            else:
                raise Exception("Invalid dropout type (must be a 2-D tuple of "
                                "float)")
            out_cell = DropoutWrapper(cell, in_keep_prob, out_keep_prob)

        inference = incoming
        # If a tensor given, convert it to a per timestep list
        if type(inference) not in [list, np.array]:
            ndim = len(input_shape)
            assert ndim >= 3, "Input dim should be at least 3."
            axes = [1, 0] + list(range(2, ndim))
            inference = tf.transpose(inference, (axes))
            inference = tf.unpack(inference)

        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope,
                             cell.W)
        if bias:
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope,
                                 cell.b)

        outputs, states = _rnn(out_cell, inference, dtype=tf.float32,
                               initial_state=initial_state, scope=scope[:-1],
                               sequence_length=sequence_length)

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, outputs[-1])

    o = outputs if return_seq else outputs[-1]
    s = states if return_seq else states[-1]

    return (o, s) if return_states else o


def lstm(incoming, n_units, activation='sigmoid', inner_activation='tanh',
         dropout=None, bias=True, weights_init='truncated_normal',
         forget_bias=1.0, return_seq=False, return_states=False,
         initial_state=None, sequence_length=None, trainable=True,
         restore=True, name="LSTM"):
    """ LSTM.

    Long Short Term Memory Recurrent Layer.

    Input:
        3-D Tensor [samples, timesteps, input dim].

    Output:
        if `return_seq`: 3-D Tensor [samples, timesteps, output dim].
        else: 2-D Tensor [samples, output dim].

    Arguments:
        incoming: `Tensor`. Incoming 3-D Tensor.
        n_units: `int`, number of units for this layer.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'sigmoid'.
        inner_activation: `str` (name) or `function` (returning a `Tensor`).
            LSTM inner activation. Default: 'tanh'.
        dropout: `tuple` of `float`: (input_keep_prob, output_keep_prob). The
            input and output keep probability.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (See tflearn.initializations) Default: 'truncated_normal'.
        forget_bias: `float`. Bias of the forget gate. Default: 1.0.
        return_seq: `bool`. If True, returns the full sequence instead of
            last sequence output only.
        return_states: `bool`. If True, returns a tuple with output and
            states: (output, states).
        initial_state: `Tensor`. An initial state for the RNN.  This must be
            a tensor of appropriate type and shape [batch_size x cell.state_size].
        sequence_length: Specifies the length of each sequence in inputs.
            An int32 or int64 vector (tensor) size `[batch_size]`.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        name: `str`. A name for this layer (optional).

    References:
        Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber,
        Neural Computation 9(8): 1735-1780, 1997.

    Links:
        [http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf]
        (http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)

    """
    input_shape = utils.get_incoming_shape(incoming)
    W_init = weights_init
    if isinstance(weights_init, str):
        W_init = initializations.get(weights_init)()

    with tf.name_scope(name) as scope:
        cell = BasicLSTMCell(n_units, activation, inner_activation, bias,
                             W_init, forget_bias, trainable, restore)
        out_cell = cell
        # Apply dropout
        if dropout:
            if type(dropout) in [tuple, list]:
                in_keep_prob = dropout[0]
                out_keep_prob = dropout[1]
            elif isinstance(dropout, float):
                in_keep_prob, out_keep_prob = dropout, dropout
            else:
                raise Exception("Invalid dropout type (must be a 2-D tuple of "
                                "float)")
            out_cell = DropoutWrapper(cell, in_keep_prob, out_keep_prob)

        inference = incoming
        # If a tensor given, convert it to a per timestep list
        if type(inference) not in [list, np.array]:
            ndim = len(input_shape)
            assert ndim >= 3, "Input dim should be at least 3."
            axes = [1, 0] + list(range(2, ndim))
            inference = tf.transpose(inference, (axes))
            inference = tf.unpack(inference)

        outputs, states = _rnn(out_cell, inference, dtype=tf.float32,
                               initial_state=initial_state, scope=scope[:-1],
                               sequence_length=sequence_length)
        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope, cell.W)
        if bias:
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope,
                                 cell.b)
        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, outputs[-1])

    o = outputs if return_seq else outputs[-1]
    s = states if return_seq else states[-1]

    return (o, s) if return_states else o


def gru(incoming, n_units, activation='sigmoid', inner_activation='tanh',
        dropout=None, bias=True, weights_init='truncated_normal',
        return_seq=False, return_states=False, initial_state=None,
        sequence_length=None, trainable=True, restore=True, name="GRU"):
    """ GRU.

    Gated Recurrent Unit Layer.

    Input:
        3-D Tensor Layer [samples, timesteps, input dim].

    Output:
        if `return_seq`: 3-D Tensor [samples, timesteps, output dim].
        else: 2-D Tensor [samples, output dim].

    Arguments:
        incoming: `Tensor`. Incoming 3-D Tensor.
        n_units: `int`, number of units for this layer.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'sigmoid'.
        inner_activation: `str` (name) or `function` (returning a `Tensor`).
            GRU inner activation. Default: 'tanh'.
        dropout: `tuple` of `float`: (input_keep_prob, output_keep_prob). The
            input and output keep probability.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (See tflearn.initializations) Default: 'truncated_normal'.
        return_seq: `bool`. If True, returns the full sequence instead of
            last sequence output only.
        return_states: `bool`. If True, returns a tuple with output and
            states: (output, states).
        initial_state: `Tensor`. An initial state for the RNN.  This must be
            a tensor of appropriate type and shape [batch_size x cell.state_size].
        sequence_length: Specifies the length of each sequence in inputs.
            An int32 or int64 vector (tensor) size `[batch_size]`.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        name: `str`. A name for this layer (optional).

    References:
        Learning Phrase Representations using RNN Encoder–Decoder for
        Statistical Machine Translation, K. Cho et al., 2014.

    Links:
        [http://arxiv.org/abs/1406.1078](http://arxiv.org/abs/1406.1078)

    """
    input_shape = utils.get_incoming_shape(incoming)
    W_init = weights_init
    if isinstance(weights_init, str):
        W_init = initializations.get(weights_init)()

    with tf.name_scope(name) as scope:
        cell = GRUCell(n_units, activation, inner_activation, bias, W_init,
                       trainable, restore)
        out_cell = cell
        # Apply dropout
        if dropout:
            if type(dropout) in [tuple, list]:
                in_keep_prob = dropout[0]
                out_keep_prob = dropout[1]
            elif isinstance(dropout, float):
                in_keep_prob, out_keep_prob = dropout, dropout
            else:
                raise Exception("Invalid dropout type (must be a 2-D tuple of "
                                "float)")
            out_cell = DropoutWrapper(cell, in_keep_prob, out_keep_prob)

        inference = incoming
        # If a tensor given, convert it to a per timestep list
        if type(inference) not in [list, np.array]:
            ndim = len(input_shape)
            assert ndim >= 3, "Input dim should be at least 3."
            axes = [1, 0] + list(range(2, ndim))
            inference = tf.transpose(inference, (axes))
            inference = tf.unpack(inference)

        outputs, states = _rnn(out_cell, inference, dtype=tf.float32,
                               initial_state=initial_state, scope=scope[:-1],
                               sequence_length=sequence_length)

        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope,
                             cell.W[0])
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope,
                             cell.W[1])
        if bias:
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope,
                                 cell.b[0])
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope,
                                 cell.b[1])
        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, outputs[-1])

    o = outputs if return_seq else outputs[-1]
    s = states if return_seq else states[-1]

    return (o, s) if return_states else o


def bidirectional_rnn(incoming, rnncell_fw, rnncell_bw, return_seq=False,
                      return_states=False, initial_state_fw=None,
                      initial_state_bw=None, sequence_length=None,
                      name="BidirectionalRNN"):
    """ Bidirectional RNN.

    Build a bidirectional recurrent neural network, it requires 2 RNN Cells
    to process sequence in forward and backward order. Any RNN Cell can be
    used i.e. SimpleRNN, LSTM, GRU... with its own parameters. But the two
    cells number of units must match.

    Input:
        3-D Tensor Layer [samples, timesteps, input dim].

    Output:
        if `return_seq`: 3-D Tensor [samples, timesteps, output dim].
        else: 2-D Tensor Layer [samples, output dim].

    Arguments:
        incoming: `Tensor`. The incoming Tensor.
        rnncell_fw: `RNNCell`. The RNN Cell to use for foward computation.
        rnncell_bw: `RNNCell`. The RNN Cell to use for backward computation.
        return_seq: `bool`. If True, returns the full sequence instead of
            last sequence output only.
        return_states: `bool`. If True, returns a tuple with output and
            states: (output, states).
        initial_state_fw: `Tensor`. An initial state for the forward RNN.
            This must be a tensor of appropriate type and shape [batch_size
            x cell.state_size].
        initial_state_bw: `Tensor`. An initial state for the backward RNN.
            This must be a tensor of appropriate type and shape [batch_size
            x cell.state_size].
        sequence_length: Specifies the length of each sequence in inputs.
            An int32 or int64 vector (tensor) size `[batch_size]`.
        name: `str`. A name for this layer (optional).

    """
    assert (rnncell_fw._num_units == rnncell_bw._num_units), \
        "RNN Cells number of units must match!"

    with tf.name_scope(name) as scope:

        inference = incoming
        # If a tensor given, convert it to a per timestep list
        if type(inference) not in [list, np.array]:
            input_shape = utils.get_incoming_shape(inference)
            ndim = len(input_shape)
            assert ndim >= 3, "Input dim should be at least 3."
            axes = [1, 0] + list(range(2, ndim))
            inference = tf.transpose(inference, (axes))
            inference = tf.unpack(inference)

        outputs, states_fw, states_bw = _bidirectional_rnn(
            rnncell_fw, rnncell_bw, inference,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            sequence_length=sequence_length,
            scope="BiRNN")

        c = tf.GraphKeys.LAYER_VARIABLES
        for v in [rnncell_fw.W, rnncell_fw.b, rnncell_bw.W, rnncell_bw.b]:
            if hasattr(v, "__len__"):
                for var in v: tf.add_to_collection(c, var)
            else:
                tf.add_to_collection(c, v)

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, outputs[-1])

    o = outputs if return_seq else outputs[-1]
    sfw = states_fw if return_seq else states_fw[-1]
    sbw = states_fw if return_seq else states_bw[-1]

    return (o, sfw, sbw) if return_states else o


def dynamic_rnn(incoming, rnncell, sequence_length=None, time_major=False,
                return_seq=False, return_states=False, initial_state=None,
                name="DynamicRNN"):
    """ Dynamic RNN.

    RNN with dynamic sequence length.

    Unlike `rnn`, the input `incoming` is not a Python list of `Tensors`.
    Instead, it is a single `Tensor` where the maximum time is either the
    first or second dimension (see the parameter `time_major`).  The
    corresponding output is a single `Tensor` having the same number of time
    steps and batch size.

    The parameter `sequence_length` is required and dynamic calculation is
    automatically performed.

    Input:
        3-D Tensor Layer [samples, timesteps, input dim].

    Output:
        if `return_seq`: 3-D Tensor [samples, timesteps, output dim].
        else: 2-D Tensor Layer [samples, output dim].

    Arguments:
        incoming: `Tensor`. The incoming 3-D Tensor.
        rnncell: `RNNCell`. The RNN Cell to use for computation.
        sequence_length: `int32` `Tensor`. A Tensor of shape [batch_size].
            (Optional).
        time_major: The shape format of the `inputs` and `outputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using time_major = False is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        return_seq: `bool`. If True, returns the full sequence instead of
            last sequence output only.
        return_states: `bool`. If True, returns a tuple with output and
            states: (output, states).
        initial_state: `Tensor`. An initial state for the RNN.  This must be
            a tensor of appropriate type and shape [batch_size x cell.state_size].
        name: `str`. A name for this layer (optional).

    """

    # Variables initialization
    with tf.name_scope(name) as scope:

        inference = incoming
        # If a tensor given, convert it to a per timestep list
        if type(inference) not in [list, np.array]:
            input_shape = utils.get_incoming_shape(inference)
            ndim = len(input_shape)
            assert ndim >= 3, "Input dim should be at least 3."
            axes = [1, 0] + list(range(2, ndim))
            inference = tf.transpose(inference, (axes))
            inference = tf.unpack(inference)

        outputs, final_state = _dynamic_rnn(rnncell, inference,
                                            initial_state=initial_state,
                                            sequence_length=sequence_length,
                                            time_major=time_major,
                                            scope="DynamicRNN")

        c = tf.GraphKeys.LAYER_VARIABLES
        for v in [rnncell.W, rnncell.b]:
            if hasattr(v, "__len__"):
                for var in v: tf.add_to_collection(c, var)
            else:
                tf.add_to_collection(c, v)

    o = outputs if return_seq else outputs[-1]
    s = final_state # Only final_state available

    return (o, s) if return_states else o

# --------------------------
#  RNN Cells
# --------------------------


class RNNCell(object):
    """ RNNCell.

    Abstract object representing an RNN cell.

    An RNN cell, in the most abstract setting, is anything that has
    a state -- a vector of floats of size self.state_size -- and performs some
    operation that takes inputs of size self.input_size. This operation
    results in an output of size self.output_size and a new state.

    """

    def __call__(self, inputs, state, scope):
        """ Run this RNN cell on inputs, starting from the given state.

        Arguments:
            inputs: 2D Tensor with shape [batch_size x self.input_size].
            state: 2D Tensor with shape [batch_size x self.state_size].
            scope: VariableScope for the created subgraph; defaults to
                class name.

        Returns:
            A pair containing:
            - Output: A 2D Tensor with shape [batch_size x self.output_size]
            - New state: A 2D Tensor with shape [batch_size x self.state_size].
        """
        raise NotImplementedError("Abstract method")

    @property
    def input_size(self):
        """Integer: size of inputs accepted by this cell."""
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        """Integer: size of state used by this cell."""
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, dtype):
        """Return state tensor (shape [batch_size x state_size]) filled with 0.

        Arguments:
            batch_size: int, float, or unit Tensor representing the batch size.
            dtype: the data type to use for the state.

        Returns:
            A 2D Tensor of shape [batch_size x state_size] filled with zeros.
        """
        zeros = array_ops.zeros(
            array_ops.pack([batch_size, self.state_size]), dtype=dtype)
        zeros.set_shape([None, self.state_size])
        return zeros


class BasicRNNCell(RNNCell):
    """The most basic RNN cell."""

    def __init__(self, num_units, activation='tanh', bias=True, W_init=None,
                 trainable=True, restore=True):
        self._num_units = num_units
        if isinstance(activation, str):
            self.activation = activations.get(activation)
        elif hasattr(activation, '__call__'):
            self.activation = activation
        else:
            raise ValueError("Invalid Activation.")
        self.W = None
        self.b = None
        if isinstance(W_init, str):
            W_init = initializations.get(W_init)()
        self.W_init = W_init
        self.bias = bias
        self.trainable = trainable
        self.restore = restore

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope):
        """Most basic RNN: output = new_state = tanh(W * input + U * state + B)."""
        self.W, self.b, concat = _linear([inputs, state], self._num_units,
                                         self.bias, self.W, self.b,
                                         self.W_init, trainable=self.trainable,
                                         restore=self.restore, scope=scope)
        output = self.activation(concat)
        return output, output


class BasicLSTMCell(RNNCell):
    """ Basic LSTM recurrent network cell.

    The implementation is based on: http://arxiv.org/pdf/1409.2329v5.pdf.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    Biases of the forget gate are initialized by default to 1 in order to reduce
    the scale of forgetting in the beginning of the training.
    """

    def __init__(self, num_units, activation='sigmoid',
                 inner_activation='tanh', bias=True, W_init=None,
                 forget_bias=1.0, trainable=True, restore=True):
        self._num_units = num_units
        self._forget_bias = forget_bias
        if isinstance(activation, str):
            self.activation = activations.get(activation)
        elif hasattr(activation, '__call__'):
            self.activation = activation
        else:
            raise ValueError("Invalid Activation.")
        if isinstance(inner_activation, str):
            self.inner_activation = activations.get(inner_activation)
        elif hasattr(inner_activation, '__call__'):
            self.inner_activation = inner_activation
        else:
            raise ValueError("Invalid Activation.")
        self.W = None
        self.b = None
        if isinstance(W_init, str):
            W_init = initializations.get(W_init)()
        self.W_init = W_init
        self.bias = bias
        self.trainable = trainable
        self.restore = restore

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return 2 * self._num_units

    def __call__(self, inputs, state, scope):
        # Parameters of gates are concatenated into one multiply for efficiency.
        c, h = array_ops.split(1, 2, state)
        self.W, self.b, concat = _linear([inputs, h], 4 * self._num_units,
                                         self.bias, self.W, self.b,
                                         self.W_init,
                                         trainable=self.trainable,
                                         restore=self.restore,
                                         scope=scope)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(1, 4, concat)

        new_c = c * self.activation(f + self._forget_bias) + self.activation(
            i) * self.inner_activation(j)
        new_h = self.inner_activation(new_c) * self.activation(o)
        return new_h, array_ops.concat(1, [new_c, new_h])


class GRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, activation='sigmoid',
                 inner_activation='tanh', bias=True, W_init=None,
                 input_size=None, trainable=True, restore=True):
        self._num_units = num_units
        self._input_size = num_units if input_size is None else input_size
        if isinstance(activation, str):
            self.activation = activations.get(activation)
        elif hasattr(activation, '__call__'):
            self.activation = activation
        else:
            raise ValueError("Invalid Activation.")
        if isinstance(inner_activation, str):
            self.inner_activation = activations.get(inner_activation)
        elif hasattr(inner_activation, '__call__'):
            self.inner_activation = inner_activation
        else:
            raise ValueError("Invalid Activation.")
        self.W = [None, None]
        self.b = [None, None]
        if isinstance(W_init, str):
            W_init = initializations.get(W_init)()
        self.W_init = W_init
        self.bias = bias
        self.trainable = trainable
        self.restore = restore

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope):
        """Gated recurrent unit (GRU) with nunits cells."""
        # We start with bias of 1.0 to not reset and not update.
        self.W[0], self.b[0], r_u = _linear([inputs, state],
                                            2 * self._num_units,
                                            self.bias, self.W[0],
                                            self.b[0], self.W_init, 1.0,
                                            trainable=self.trainable,
                                            restore=self.restore,
                                            scope=scope + "/Gates")
        r, u = array_ops.split(1, 2, r_u)
        r, u = self.activation(r), self.activation(u)

        self.W[1], self.b[1], c = _linear([inputs, r * state], self._num_units,
                                          self.bias, self.W[1],
                                          self.b[1], self.W_init,
                                          trainable=self.trainable,
                                          restore=self.restore,
                                          scope=scope + "/Candidate")
        c = self.inner_activation(c)
        new_h = u * state + (1 - u) * c
        return new_h, new_h


class DropoutWrapper(RNNCell):
    """Operator adding dropout to inputs and outputs of the given cell."""

    def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
                 seed=None):
        """Create a cell with added input and/or output dropout.
        Dropout is never used on the state.
        Args:
          cell: an RNNCell, a projection to output_size is added to it.
          input_keep_prob: unit Tensor or float between 0 and 1, input keep
            probability; if it is float and 1, no input dropout will be added.
          output_keep_prob: unit Tensor or float between 0 and 1, output keep
            probability; if it is float and 1, no output dropout will be added.
          seed: (optional) integer, the randomness seed.
        Raises:
          TypeError: if cell is not an RNNCell.
          ValueError: if keep_prob is not between 0 and 1.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")
        if (isinstance(input_keep_prob, float) and
                not (input_keep_prob >= 0.0 and input_keep_prob <= 1.0)):
            raise ValueError(
                "Parameter input_keep_prob must be between 0 and 1: %d"
                % input_keep_prob)
        if (isinstance(output_keep_prob, float) and
                not (output_keep_prob >= 0.0 and output_keep_prob <= 1.0)):
            raise ValueError(
                "Parameter input_keep_prob must be between 0 and 1: %d"
                % output_keep_prob)
        self._cell = cell
        self._input_keep_prob = input_keep_prob
        self._output_keep_prob = output_keep_prob
        self._seed = seed

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell with the declared dropouts."""
        if (not isinstance(self._input_keep_prob, float) or
                    self._input_keep_prob < 1):
            inputs = nn_ops.dropout(inputs, self._input_keep_prob,
                                    seed=self._seed)
        output, new_state = self._cell(inputs, state, scope)
        if (not isinstance(self._output_keep_prob, float) or
                    self._output_keep_prob < 1):
            output = nn_ops.dropout(output, self._output_keep_prob,
                                    seed=self._seed)
        return output, new_state

# --------------------------
#  RNN calculations
# --------------------------


def _rnn(cell, inputs, initial_state=None, dtype=None, sequence_length=None,
         scope=None):
    """ Creates a recurrent neural network specified by RNNCell "cell".

    The simplest form of RNN network generated is:
      state = cell.zero_state(...)
      outputs = []
      states = []
      for input_ in inputs:
        output, state = cell(input_, state)
        outputs.append(output)
        states.append(state)
      return (outputs, states)

    However, a few other options are available:

    An initial state can be provided.
    If sequence_length is provided, dynamic calculation is performed.

    Dynamic calculation returns, at time t:
      (t >= max(sequence_length)
          ? (zeros(output_shape), zeros(state_shape))
          : cell(input, state)

    Thus saving computational time when unrolling past the max sequence length.

    Arguments:
      cell: An instance of RNNCell.
      inputs: A length T list of inputs, each a tensor of shape
        [batch_size, cell.input_size].
      initial_state: (optional) An initial state for the RNN.  This must be
        a tensor of appropriate type and shape [batch_size x cell.state_size].
      dtype: (optional) The data type for the initial state.  Required if
        initial_state is not provided.
      sequence_length: An int64 vector (tensor) size [batch_size].
      scope: VariableScope for the created subgraph; defaults to "RNN".

    Returns:
      A pair (outputs, states) where:
        outputs is a length T list of outputs (one for each input)
        states is a length T list of states (one state following each input)

    Raises:
      TypeError: If "cell" is not an instance of RNNCell.
      ValueError: If inputs is None or an empty list.
    """

    if not isinstance(cell, RNNCell):
        raise TypeError("cell must be an instance of RNNCell")
    if not isinstance(inputs, list):
        raise TypeError("inputs must be a list")
    if not inputs:
        raise ValueError("inputs must not be empty")

    outputs = []
    states = []
    batch_size = array_ops.shape(inputs[0])[0]
    if initial_state is not None:
        state = initial_state
    else:
        if not dtype:
            raise ValueError("If no initial_state is provided, dtype must be.")
        state = cell.zero_state(batch_size, dtype)

    if sequence_length:  # Prepare variables
        zero_output_state = (
            array_ops.zeros(array_ops.pack([batch_size, cell.output_size]),
                            inputs[0].dtype),
            array_ops.zeros(array_ops.pack([batch_size, cell.state_size]),
                            state.dtype))
        max_sequence_length = tf.reduce_max(sequence_length)

    for time, input_ in enumerate(inputs):
        def output_state():
            return cell(input_, state, scope)

        if sequence_length:
            (output, state) = control_flow_ops.cond(
                time >= max_sequence_length,
                lambda: zero_output_state, output_state)
        else:
            (output, state) = output_state()

        outputs.append(output)
        states.append(state)

    return (outputs, states)


def _bidirectional_rnn(cell_fw, cell_bw, inputs,
                       initial_state_fw=None, initial_state_bw=None,
                       dtype=tf.float32, sequence_length=None, scope=None):
    """ Creates a bidirectional recurrent neural network.
    Similar to the unidirectional case above (rnn) but takes input and builds
    independent forward and backward RNNs with the final forward and backward
    outputs depth-concatenated, such that the output will have the format
    [time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
    forward and backward cell must match. The initial state for both directions
    is zero by default (but can be set optionally) and no intermediate states are
    ever returned -- the network is fully unrolled for the given (passed in)
    length(s) of the sequence(s) or completely unrolled if length(s) is not given.

    Arguments:
        cell_fw: An instance of RNNCell, to be used for forward direction.
        cell_bw: An instance of RNNCell, to be used for backward direction.
        inputs: A length T list of inputs, each a tensor of shape
            [batch_size, cell.input_size].
        initial_state_fw: (optional) An initial state for the forward RNN.
            This must be a tensor of appropriate type and shape
            [batch_size x cell.state_size].
        initial_state_bw: (optional) Same as for initial_state_fw.
        dtype: (optional) The data type for the initial state.  Required if either
            of the initial states are not provided.
        sequence_length: (optional) An int32/int64 vector, size [batch_size],
            containing the actual lengths for each of the sequences.
        scope: VariableScope for the created subgraph; defaults to "BiRNN"

    Returns:
      A tuple (outputs, output_state_fw, output_state_bw) where:
        outputs is a length T list of outputs (one for each input), which
        are depth-concatenated forward and backward outputs
        output_state_fw is the final state of the forward rnn
        output_state_bw is the final state of the backward rnn

    Raises:
      TypeError: If "cell_fw" or "cell_bw" is not an instance of RNNCell.
      ValueError: If inputs is None or an empty list.

    """

    if not isinstance(cell_fw, RNNCell):
        raise TypeError("cell_fw must be an instance of RNNCell")
    if not isinstance(cell_bw, RNNCell):
        raise TypeError("cell_bw must be an instance of RNNCell")
    if not isinstance(inputs, list):
        raise TypeError("inputs must be a list")
    if not inputs:
        raise ValueError("inputs must not be empty")

    name = scope or "BiRNN"
    # Forward direction
    with tf.name_scope(name + "_FW") as fw_scope:
        output_fw, output_state_fw = _rnn(cell_fw, inputs, initial_state_fw,
                                         dtype,
                                         sequence_length, scope=fw_scope)

    # Backward direction
    with tf.name_scope(name + "_BW") as bw_scope:
        tmp, output_state_bw = _rnn(cell_bw,
                                   _reverse_seq(inputs, sequence_length),
                                   initial_state_bw, dtype, sequence_length,
                                   scope=bw_scope)
    output_bw = _reverse_seq(tmp, sequence_length)
    # Concat each of the forward/backward outputs
    outputs = [array_ops.concat(1, [fw, bw])
               for fw, bw in zip(output_fw, output_bw)]

    return outputs, output_state_fw, output_state_bw


def _reverse_seq(input_seq, lengths):
    """ Reverse a list of Tensors up to specified lengths.

    Arguments:
        input_seq: Sequence of seq_len tensors of dimension (batch_size, depth)
        lengths: A tensor of dimension batch_size, containing lengths for each
            sequence in the batch. If "None" is specified, simply reverses
            the list.

    Returns:
      time-reversed sequence

    """
    if lengths is None:
        return list(reversed(input_seq))

    input_shape = tensor_shape.matrix(None, None)
    for input_ in input_seq:
        input_shape.merge_with(input_.get_shape())
        input_.set_shape(input_shape)

    # Join into (time, batch_size, depth)
    s_joined = array_ops.pack(input_seq)

    # TODO(schuster, ebrevdo): Remove cast when reverse_sequence takes int32
    if lengths is not None:
        lengths = math_ops.to_int64(lengths)

    # Reverse along dimension 0
    s_reversed = array_ops.reverse_sequence(s_joined, lengths, 0, 1)
    # Split again into list
    result = array_ops.unpack(s_reversed)
    for r in result:
        r.set_shape(input_shape)
    return result


def _dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                 dtype=None, parallel_iterations=None, swap_memory=False,
                 time_major=False, scope=None):
    """ Creates a recurrent neural network specified by RNNCell "cell".
    This function is functionally identical to the function `rnn` above, but
    performs fully dynamic unrolling of `inputs`.
    Unlike `rnn`, the input `inputs` is not a Python list of `Tensors`.  Instead,
    it is a single `Tensor` where the maximum time is either the first or second
    dimension (see the parameter `time_major`).  The corresponding output is
    a single `Tensor` having the same number of time steps and batch size.
    The parameter `sequence_length` is required and dynamic calculation is
    automatically performed.

    Arguments:
      cell: An instance of RNNCell.
      inputs: The RNN inputs.
        If time_major == False (default), this must be a tensor of shape:
          `[batch_size, max_time, input_size]`.
        If time_major == True, this must be a tensor of shape:
          `[max_time, batch_size, input_size]`.
      sequence_length: (optional) An int32/int64 vector sized `[batch_size]`.
      initial_state: (optional) An initial state for the RNN.  This must be
        a tensor of appropriate type and shape `[batch_size x cell.state_size]`.
      dtype: (optional) The data type for the initial state.  Required if
        initial_state is not provided.
      parallel_iterations: (Default: 32).  The number of iterations to run in
        parallel.  Those operations which do not have any temporal dependency
        and can be run in parallel, will be.  This parameter trades off
        time for space.  Values >> 1 use more memory but take less time,
        while smaller values use less memory but computations take longer.
      swap_memory: Swap the tensors produced in forward inference but needed
        for back prop from GPU to CPU.
      time_major: The shape format of the `inputs` and `outputs` Tensors.
        If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
        If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
        Using time_major = False is a bit more efficient because it avoids
        transposes at the beginning and end of the RNN calculation.  However,
        most TensorFlow data is batch-major, so by default this function
        accepts input and emits output in batch-major form.
      scope: VariableScope for the created subgraph; defaults to "RNN".

    Returns:
      A pair (outputs, state) where:
        outputs: The RNN output `Tensor`.
          If time_major == False (default), this will be a `Tensor` shaped:
            `[batch_size, max_time, cell.output_size]`.
          If time_major == True, this will be a `Tensor` shaped:
            `[max_time, batch_size, cell.output_size]`.
        state: The final state, shaped:
          `[batch_size, cell.state_size]`.

    Raises:
      TypeError: If "cell" is not an instance of RNNCell.
      ValueError: If inputs is None or an empty list.
    """

    if not isinstance(cell, RNNCell):
        raise TypeError("cell must be an instance of RNNCell")

        # By default, time_major==False and inputs are batch-major: shaped
        #   [batch, time, depth]
        # For internal calculations, we transpose to [time, batch, depth]
    if not time_major:
        inputs = array_ops.transpose(inputs, [1, 0, 2])  # (B,T,D) => (T,B,D)

    parallel_iterations = parallel_iterations or 32
    if sequence_length is not None:
        sequence_length = math_ops.to_int32(sequence_length)
        sequence_length = array_ops.identity(  # Just to find it in the graph.
            sequence_length, name="sequence_length")

    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with tf.variable_scope(scope or "RNN") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)
        input_shape = array_ops.shape(inputs)
        batch_size = input_shape[1]

        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("If no initial_state is provided, dtype must be.")
            state = cell.zero_state(batch_size, dtype)

        def _assert_has_shape(x, shape):
            x_shape = array_ops.shape(x)
            packed_shape = array_ops.pack(shape)
            return logging_ops.Assert(
                math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)),
                ["Expected shape for Tensor %s is " % x.name,
                 packed_shape, " but saw shape: ", x_shape])

        if sequence_length is not None:
            # Perform some shape validation
            with ops.control_dependencies(
                    [_assert_has_shape(sequence_length, [batch_size])]):
                sequence_length = array_ops.identity(
                    sequence_length, name="CheckSeqLen")

        (outputs, final_state) = _dynamic_rnn_loop(
            cell, inputs, state, parallel_iterations=parallel_iterations,
            swap_memory=swap_memory, sequence_length=sequence_length)

        # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
        # If we are performing batch-major calculations, transpose output back
        # to shape [batch, time, depth]
        if not time_major:
            outputs = array_ops.transpose(outputs, [1, 0, 2])  # (T,B,D) => (B,T,D)

        return outputs, final_state


def _dynamic_rnn_loop(cell, inputs, initial_state, parallel_iterations,
                      swap_memory, sequence_length=None):
    """Internal implementation of Dynamic RNN.

    Arguments:
      cell: An instance of RNNCell.
      inputs: A `Tensor` of shape [time, batch_size, depth].
      initial_state: A `Tensor` of shape [batch_size, depth].
      parallel_iterations: Positive Python int.
      swap_memory: A Python boolean
      sequence_length: (optional) An `int32` `Tensor` of shape [batch_size].

    Returns:
      Tuple (final_outputs, final_state).
      final_outputs:
        A `Tensor` of shape [time, batch_size, depth]`.
      final_state:
        A `Tensor` of shape [batch_size, depth].

    Raises:
      ValueError: If the input depth cannot be inferred via shape inference
        from the inputs.
    """
    state = initial_state
    assert isinstance(parallel_iterations,
                      int), "parallel_iterations must be int"

    # Construct an initial output
    input_shape = array_ops.shape(inputs)
    (time_steps, batch_size, _) = array_ops.unpack(input_shape, 3)

    inputs_got_shape = inputs.get_shape().with_rank(3)
    (const_time_steps, const_batch_size,
     const_depth) = inputs_got_shape.as_list()

    if const_depth is None:
        raise ValueError(
            "Input size (depth of inputs) must be accessible via shape inference, "
            "but saw value None.")

    # Prepare dynamic conditional copying of state & output
    zero_output = array_ops.zeros(
        array_ops.pack([batch_size, cell.output_size]), inputs.dtype)
    if sequence_length is not None:
        min_sequence_length = math_ops.reduce_min(sequence_length)
        max_sequence_length = math_ops.reduce_max(sequence_length)

    time = array_ops.constant(0, dtype=tf.int32, name="time")

    with ops.op_scope([], "dynamic_rnn") as scope:
        base_name = scope

    output_ta = tensor_array_ops.TensorArray(
        dtype=inputs.dtype, size=time_steps,
        tensor_array_name=base_name + "output")

    input_ta = tensor_array_ops.TensorArray(
        dtype=inputs.dtype, size=time_steps,
        tensor_array_name=base_name + "input")

    input_ta = input_ta.unpack(inputs)

    def _time_step(time, state, output_ta_t):
        """Take a time step of the dynamic RNN.
        Args:
          time: int32 scalar Tensor.
          state: Vector.
          output_ta_t: `TensorArray`, the output with existing flow.
        Returns:
          The tuple (time + 1, new_state, output_ta_t with updated flow).
        """

        input_t = input_ta.read(time)
        # Restore some shape information
        input_t.set_shape([const_batch_size, const_depth])

        call_cell = lambda: cell(input_t, state)

        if sequence_length is not None:
            (output, new_state) = _rnn_step(
                time=time,
                sequence_length=sequence_length,
                min_sequence_length=min_sequence_length,
                max_sequence_length=max_sequence_length,
                zero_output=zero_output,
                state=state,
                call_cell=call_cell,
                skip_conditionals=True)
        else:
            (output, new_state) = call_cell()

        output_ta_t = output_ta_t.write(time, output)

        return (time + 1, new_state, output_ta_t)

    (_, final_state, output_final_ta) = control_flow_ops.while_loop(
        cond=lambda time, _1, _2: time < time_steps,
        body=_time_step,
        loop_vars=(time, state, output_ta),
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory)

    final_outputs = output_final_ta.pack()
    # Restore some shape information
    final_outputs.set_shape([
        const_time_steps, const_batch_size, cell.output_size])

    return final_outputs, final_state


def _rnn_step(time, sequence_length, min_sequence_length, max_sequence_length,
              zero_output, state, call_cell, skip_conditionals=False):
    """Calculate one step of a dynamic RNN minibatch.
    Returns an (output, state) pair conditioned on the sequence_lengths.
    When skip_conditionals=False, the pseudocode is something like:
    if t >= max_sequence_length:
      return (zero_output, state)
    if t < min_sequence_length:
      return call_cell()
    # Selectively output zeros or output, old state or new state depending
    # on if we've finished calculating each row.
    new_output, new_state = call_cell()
    final_output = np.vstack([
      zero_output if time >= sequence_lengths[r] else new_output_r
      for r, new_output_r in enumerate(new_output)
    ])
    final_state = np.vstack([
      state[r] if time >= sequence_lengths[r] else new_state_r
      for r, new_state_r in enumerate(new_state)
    ])
    return (final_output, final_state)

    Arguments:
      time: Python int, the current time step
      sequence_length: int32 `Tensor` vector of size [batch_size]
      min_sequence_length: int32 `Tensor` scalar, min of sequence_length
      max_sequence_length: int32 `Tensor` scalar, max of sequence_length
      zero_output: `Tensor` vector of shape [output_size]
      state: `Tensor` matrix of shape [batch_size, state_size]
      call_cell: lambda returning tuple of (new_output, new_state) where
        new_output is a `Tensor` matrix of shape [batch_size, output_size]
        new_state is a `Tensor` matrix of shape [batch_size, state_size]
      skip_conditionals: Python bool, whether to skip using the conditional
        calculations.  This is useful for dynamic_rnn, where the input tensor
        matches max_sequence_length, and using conditionals just slows
        everything down.

    Returns:
      A tuple of (final_output, final_state) as given by the pseudocode above:
        final_output is a `Tensor` matrix of shape [batch_size, output_size]
        final_state is a `Tensor` matrix of shape [batch_size, state_size]
    """
    state_shape = state.get_shape()

    def _copy_some_through(new_output, new_state):
        # Use broadcasting select to determine which values should get
        # the previous state & zero output, and which values should get
        # a calculated state & output.
        copy_cond = (time >= sequence_length)
        return (math_ops.select(copy_cond, zero_output, new_output),
                math_ops.select(copy_cond, state, new_state))

    def _maybe_copy_some_through():
        """Run RNN step.  Pass through either no or some past state."""
        new_output, new_state = call_cell()

        return control_flow_ops.cond(
            # if t < min_seq_len: calculate and return everything
            time < min_sequence_length, lambda: (new_output, new_state),
            # else copy some of it through
            lambda: _copy_some_through(new_output, new_state))

    # but benefits from removing cond() and its gradient.  We should
    # profile with and without this switch here.
    if skip_conditionals:
        # Instead of using conditionals, perform the selective copy at all time
        # steps.  This is faster when max_seq_len is equal to the number of unrolls
        # (which is typical for dynamic_rnn).
        new_output, new_state = call_cell()
        (final_output, final_state) = _copy_some_through(new_output, new_state)
    else:
        empty_update = lambda: (zero_output, state)

        (final_output, final_state) = control_flow_ops.cond(
            # if t >= max_seq_len: copy all state through, output zeros
            time >= max_sequence_length, empty_update,
            # otherwise calculation is required: copy some or all of it through
            _maybe_copy_some_through)

    final_output.set_shape(zero_output.get_shape())
    final_state.set_shape(state_shape)
    return (final_output, final_state)


def _linear(args, output_size, bias, W=None, b=None, W_init=None,
           bias_start=0.0, trainable=True, restore=True, scope=None):
    """ Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Arguments:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: `int`. Second dimension of W[i].
        bias: `bool`. Whether to add a bias term or not.
        W: `Tensor`. The weights. If None, it will be automatically created.
        b: `Tensor`. The bias. If None, it will be automatically created.
        W_init: `str`. Weights initialization mode. See
            tflearn.initializations.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
        A `tuple` containing:
        - W: `Tensor` variable holding the weights.
        - b: `Tensor` variable holding the bias.
        - res: `2D tf.Tensor` with shape [batch x output_size] equal to
            sum_i(args[i] * W[i]).

    """
    # Creates W if it hasn't be created yet.
    if not W:
        assert args
        if not isinstance(args, (list, tuple)):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape().as_list() for a in args]
        for shape in shapes:
            if len(shape) != 2:
                raise ValueError(
                    "Linear is expecting 2D arguments: %s" % str(shapes))
            if not shape[1]:
                raise ValueError(
                    "Linear expects shape[1] of arguments: %s" % str(shapes))
            else:
                total_arg_size += shape[1]
        with tf.variable_scope(scope, reuse=False):
            W = tf.get_variable(name="W", shape=[total_arg_size, output_size],
                                initializer=W_init, trainable=trainable)
            if not restore:
                tf.add_to_collection(tf.GraphKeys.EXCL_RESTORE_VARS, W)

    # Now the computation.
    if len(args) == 1:
        res = tf.matmul(args[0], W)
    else:
        res = tf.matmul(array_ops.concat(1, args), W)
    if not bias:
        return W, None, res

    # Creates b if it hasn't be created yet.
    if not b:
        with tf.variable_scope(scope, reuse=False) as vs:
            b = tf.get_variable(
                "b", [output_size],
                initializer=init_ops.constant_initializer(bias_start),
                trainable=trainable)
            if not restore:
                tf.add_to_collection(tf.GraphKeys.EXCL_RESTORE_VARS, b)
    return W, b, res + b
