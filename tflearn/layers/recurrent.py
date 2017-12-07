# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
try:
    from tensorflow.python.ops.rnn import rnn_cell_impl as _rnn_cell, dynamic_rnn as _drnn, static_rnn as _rnn, static_bidirectional_rnn as _brnn
    core_rnn_cell = _rnn_cell
except:
    # Fix for TF 1.1.0 and under
    from tensorflow.contrib.rnn.python.ops.core_rnn import static_rnn as _rnn, static_bidirectional_rnn as _brnn
    from tensorflow.python.ops.rnn import rnn_cell_impl as _rnn_cell, dynamic_rnn as _drnn
    from tensorflow.contrib.rnn.python.ops import core_rnn_cell

from tensorflow.python.util.nest import is_sequence

from .. import config
from .. import utils
from .. import activations
from .. import initializations
from .. import variables as va
from .normalization import batch_normalization

# --------------------------
#  RNN Layers
# --------------------------

def _rnn_template(incoming, cell, dropout=None, return_seq=False,
                  return_state=False, initial_state=None, dynamic=False,
                  scope=None, reuse=False, name="LSTM"):
    """ RNN Layer Template. """
    sequence_length = None
    if dynamic:
        sequence_length = retrieve_seq_length_op(
            incoming if isinstance(incoming, tf.Tensor) else tf.stack(incoming))

    input_shape = utils.get_incoming_shape(incoming)

    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:
        name = scope.name

        _cell = cell
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
            cell = DropoutWrapper(cell, in_keep_prob, out_keep_prob)

        inference = incoming
        # If a tensor given, convert it to a per timestep list
        if type(inference) not in [list, np.array]:
            ndim = len(input_shape)
            assert ndim >= 3, "Input dim should be at least 3."
            axes = [1, 0] + list(range(2, ndim))
            inference = tf.transpose(inference, (axes))
            inference = tf.unstack(inference)

        outputs, state = _rnn(cell, inference, dtype=tf.float32,
                              initial_state=initial_state, scope=name,
                              sequence_length=sequence_length)

        # Retrieve RNN Variables
        c = tf.GraphKeys.LAYER_VARIABLES + '/' + scope.name
        for v in [_cell.W, _cell.b]:
            if hasattr(v, "__len__"):
                for var in v: tf.add_to_collection(c, var)
            else:
                tf.add_to_collection(c, v)
        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, outputs[-1])

    if dynamic:
        if return_seq:
            o = tf.stack(outputs, 1)
        else:
            outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])
            o = advanced_indexing_op(outputs, sequence_length)
    else:
        o = tf.stack(outputs, 1) if return_seq else outputs[-1]

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, o)

    return (o, state) if return_state else o


def simple_rnn(incoming, n_units, activation='sigmoid', dropout=None,
               bias=True, weights_init=None, return_seq=False,
               return_state=False, initial_state=None, dynamic=False,
               trainable=True, restore=True, reuse=False, scope=None,
               name="SimpleRNN"):
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
            Default: 'sigmoid'.
        dropout: `tuple` of `float`: (input_keep_prob, output_keep_prob). The
            input and output keep probability.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (See tflearn.initializations)
        return_seq: `bool`. If True, returns the full sequence instead of
            last sequence output only.
        return_state: `bool`. If True, returns a tuple with output and
            states: (output, states).
        initial_state: `Tensor`. An initial state for the RNN.  This must be
            a tensor of appropriate type and shape [batch_size x cell.state_size].
        dynamic: `bool`. If True, dynamic computation is performed. It will not
            compute RNN steps above the sequence length. Note that because TF
            requires to feed sequences of same length, 0 is used as a mask.
            So a sequence padded with 0 at the end must be provided. When
            computation is performed, it will stop when it meets a step with
            a value of 0.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: `str`. A name for this layer (optional).

    """
    cell = BasicRNNCell(n_units, activation=activation, bias=bias,
                        weights_init=weights_init, trainable=trainable,
                        restore=restore, reuse=reuse)
    x = _rnn_template(incoming, cell=cell, dropout=dropout,
                      return_seq=return_seq, return_state=return_state,
                      initial_state=initial_state, dynamic=dynamic,
                      scope=scope, name=name)

    return x


def lstm(incoming, n_units, activation='tanh', inner_activation='sigmoid',
         dropout=None, bias=True, weights_init=None, forget_bias=1.0,
         return_seq=False, return_state=False, initial_state=None,
         dynamic=False, trainable=True, restore=True, reuse=False,
         scope=None, name="LSTM"):
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
            Default: 'tanh'.
        inner_activation: `str` (name) or `function` (returning a `Tensor`).
            LSTM inner activation. Default: 'sigmoid'.
        dropout: `tuple` of `float`: (input_keep_prob, output_keep_prob). The
            input and output keep probability.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (See tflearn.initializations).
        forget_bias: `float`. Bias of the forget gate. Default: 1.0.
        return_seq: `bool`. If True, returns the full sequence instead of
            last sequence output only.
        return_state: `bool`. If True, returns a tuple with output and
            states: (output, states).
        initial_state: `Tensor`. An initial state for the RNN.  This must be
            a tensor of appropriate type and shape [batch_size x cell.state_size].
        dynamic: `bool`. If True, dynamic computation is performed. It will not
            compute RNN steps above the sequence length. Note that because TF
            requires to feed sequences of same length, 0 is used as a mask.
            So a sequence padded with 0 at the end must be provided. When
            computation is performed, it will stop when it meets a step with
            a value of 0.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: `str`. A name for this layer (optional).

    References:
        Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber,
        Neural Computation 9(8): 1735-1780, 1997.

    Links:
        [http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf]
        (http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)

    """
    cell = BasicLSTMCell(n_units, activation=activation,
                         inner_activation=inner_activation,
                         forget_bias=forget_bias, bias=bias,
                         weights_init=weights_init, trainable=trainable,
                         restore=restore, reuse=reuse)
    x = _rnn_template(incoming, cell=cell, dropout=dropout,
                      return_seq=return_seq, return_state=return_state,
                      initial_state=initial_state, dynamic=dynamic,
                      scope=scope, name=name)

    return x


def gru(incoming, n_units, activation='tanh', inner_activation='sigmoid',
        dropout=None, bias=True, weights_init=None, return_seq=False,
        return_state=False, initial_state=None, dynamic=False,
        trainable=True, restore=True, reuse=False, scope=None, name="GRU"):
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
            Default: 'tanh'.
        inner_activation: `str` (name) or `function` (returning a `Tensor`).
            GRU inner activation. Default: 'sigmoid'.
        dropout: `tuple` of `float`: (input_keep_prob, output_keep_prob). The
            input and output keep probability.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (See tflearn.initializations).
        return_seq: `bool`. If True, returns the full sequence instead of
            last sequence output only.
        return_state: `bool`. If True, returns a tuple with output and
            states: (output, states).
        initial_state: `Tensor`. An initial state for the RNN.  This must be
            a tensor of appropriate type and shape [batch_size x cell.state_size].
        dynamic: `bool`. If True, dynamic computation is performed. It will not
            compute RNN steps above the sequence length. Note that because TF
            requires to feed sequences of same length, 0 is used as a mask.
            So a sequence padded with 0 at the end must be provided. When
            computation is performed, it will stop when it meets a step with
            a value of 0.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: `str`. A name for this layer (optional).

    References:
        Learning Phrase Representations using RNN Encoder–Decoder for
        Statistical Machine Translation, K. Cho et al., 2014.

    Links:
        [http://arxiv.org/abs/1406.1078](http://arxiv.org/abs/1406.1078)

    """
    cell = GRUCell(n_units, activation=activation,
                   inner_activation=inner_activation, bias=bias,
                   weights_init=weights_init, trainable=trainable,
                   restore=restore, reuse=reuse)
    x = _rnn_template(incoming, cell=cell, dropout=dropout,
                      return_seq=return_seq, return_state=return_state,
                      initial_state=initial_state, dynamic=dynamic,
                      scope=scope, name=name)

    return x


def bidirectional_rnn(incoming, rnncell_fw, rnncell_bw, return_seq=False,
                      return_states=False, initial_state_fw=None,
                      initial_state_bw=None, dynamic=False, scope=None,
                      name="BiRNN"):
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
        dynamic: `bool`. If True, dynamic computation is performed. It will not
            compute RNN steps above the sequence length. Note that because TF
            requires to feed sequences of same length, 0 is used as a mask.
            So a sequence padded with 0 at the end must be provided. When
            computation is performed, it will stop when it meets a step with
            a value of 0.
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: `str`. A name for this layer (optional).

    """
    assert (rnncell_fw._num_units == rnncell_bw._num_units), \
        "RNN Cells number of units must match!"

    sequence_length = None
    if dynamic:
        sequence_length = retrieve_seq_length_op(
            incoming if isinstance(incoming, tf.Tensor) else tf.stack(incoming))

    input_shape = utils.get_incoming_shape(incoming)

    with tf.variable_scope(scope, default_name=name, values=[incoming]) as scope:
        name = scope.name

        # TODO: DropoutWrapper

        inference = incoming
        # If a tensor given, convert it to a per timestep list
        if type(inference) not in [list, np.array]:
            ndim = len(input_shape)
            assert ndim >= 3, "Input dim should be at least 3."
            axes = [1, 0] + list(range(2, ndim))
            inference = tf.transpose(inference, (axes))
            inference = tf.unstack(inference)

        outputs, states_fw, states_bw = _brnn(
            rnncell_fw, rnncell_bw, inference,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            sequence_length=sequence_length,
            dtype=tf.float32)

        c = tf.GraphKeys.LAYER_VARIABLES + '/' + scope.name
        for v in [rnncell_fw.W, rnncell_fw.b, rnncell_bw.W, rnncell_bw.b]:
            if hasattr(v, "__len__"):
                for var in v: tf.add_to_collection(c, var)
            else:
                tf.add_to_collection(c, v)

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, outputs[-1])

    if dynamic:
        if return_seq:
            o = tf.stack(outputs, 1)
        else:
            outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])
            o = advanced_indexing_op(outputs, sequence_length)
    else:
        o = tf.stack(outputs, 1) if return_seq else outputs[-1]

    sfw = states_fw
    sbw = states_bw

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, o)

    return (o, sfw, sbw) if return_states else o


# --------------------------
#  RNN Cells
# --------------------------

class BasicRNNCell(core_rnn_cell.RNNCell):
    """ TF basic RNN cell with extra customization params. """

    def __init__(self, num_units, input_size=None, activation=tf.nn.tanh,
                 bias=True, weights_init=None, trainable=True, restore=True,
                 reuse=False):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated." % self)
        self._num_units = num_units
        if isinstance(activation, str):
            self._activation = activations.get(activation)
        elif hasattr(activation, '__call__'):
            self._activation = activation
        else:
            raise ValueError("Invalid Activation.")
        self.bias = bias
        self.weights_init = weights_init
        if isinstance(weights_init, str):
            self.weights_init = initializations.get(weights_init)()
        self.trainable = trainable
        self.restore = restore
        self.reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Most basic RNN: output = new_state = activation(W * input + U * state + B)."""
        with tf.variable_scope(scope or type(self).__name__):
            # "BasicRNNCell"
            output = self._activation(
                _linear([inputs, state], self._num_units, True, 0.,
                        self.weights_init, self.trainable, self.restore,
                        self.reuse))
            # Retrieve RNN Variables
            with tf.variable_scope('Linear', reuse=True):
                self.W = tf.get_variable('Matrix')
                self.b = tf.get_variable('Bias')

        return output, output


class BasicLSTMCell(core_rnn_cell.RNNCell):
    """ TF Basic LSTM recurrent network cell with extra customization params.

    The implementation is based on: http://arxiv.org/abs/1409.2329.

    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    For advanced models, please use the full LSTMCell that follows.
    """

    def __init__(self, num_units, forget_bias=1.0, input_size=None,
                 state_is_tuple=True, activation=tf.tanh,
                 inner_activation=tf.sigmoid, bias=True, weights_init=None,
                 trainable=True, restore=True, reuse=False, batch_norm = False):
        if not state_is_tuple:
            logging.warn(
                "%s: Using a concatenated state is slower and will soon be "
                "deprecated.  Use state_is_tuple=True." % self)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated." % self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self.batch_norm = batch_norm
        if isinstance(activation, str):
            self._activation = activations.get(activation)
        elif hasattr(activation, '__call__'):
            self._activation = activation
        else:
            raise ValueError("Invalid Activation.")
        if isinstance(inner_activation, str):
            self._inner_activation = activations.get(inner_activation)
        elif hasattr(inner_activation, '__call__'):
            self._inner_activation = inner_activation
        else:
            raise ValueError("Invalid Activation.")
        self.bias = bias
        self.weights_init = weights_init
        if isinstance(weights_init, str):
            self.weights_init = initializations.get(weights_init)()
        self.trainable = trainable
        self.restore = restore
        self.reuse = reuse

    @property
    def state_size(self):
        return (core_rnn_cell.LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = array_ops.split(1, 2, state)
            concat = _linear([inputs, h], 4 * self._num_units, True, 0.,
                             self.weights_init, self.trainable, self.restore,
                             self.reuse)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4,
                                         axis=1)

            # apply batch normalization to inner state and gates
            if self.batch_norm == True:
                i = batch_normalization(i, gamma=0.1, trainable=self.trainable, restore=self.restore, reuse=self.reuse)
                j = batch_normalization(j, gamma=0.1, trainable=self.trainable, restore=self.restore, reuse=self.reuse)
                f = batch_normalization(f, gamma=0.1, trainable=self.trainable, restore=self.restore, reuse=self.reuse)
                o = batch_normalization(o, gamma=0.1, trainable=self.trainable, restore=self.restore, reuse=self.reuse)
            
            new_c = (c * self._inner_activation(f + self._forget_bias) +
                     self._inner_activation(i) *
                     self._activation(j))
            
            # hidden-to-hidden batch normalizaiton
            if self.batch_norm == True:
                batch_norm_new_c = batch_normalization(new_c, gamma=0.1, trainable=self.trainable, restore=self.restore, reuse=self.reuse)
                new_h = self._activation(batch_norm_new_c) * self._inner_activation(o)
            else:
                new_h = self._activation(new_c) * self._inner_activation(o)

            if self._state_is_tuple:
                new_state = core_rnn_cell.LSTMStateTuple(new_c, new_h)
            else:
                new_state = array_ops.concat([new_c, new_h], 1)

            # Retrieve RNN Variables
            with tf.variable_scope('Linear', reuse=True):
                self.W = tf.get_variable('Matrix')
                self.b = tf.get_variable('Bias')

            return new_h, new_state


class GRUCell(core_rnn_cell.RNNCell):
    """ TF GRU Cell with extra customization params. """

    def __init__(self, num_units, input_size=None, activation=tf.tanh,
                 inner_activation=tf.sigmoid, bias=True, weights_init=None,
                 trainable=True, restore=True, reuse=False):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated." % self)
        self._num_units = num_units
        if isinstance(activation, str):
            self._activation = activations.get(activation)
        elif hasattr(activation, '__call__'):
            self._activation = activation
        else:
            raise ValueError("Invalid Activation.")
        if isinstance(inner_activation, str):
            self._inner_activation = activations.get(inner_activation)
        elif hasattr(inner_activation, '__call__'):
            self._inner_activation = inner_activation
        else:
            raise ValueError("Invalid Activation.")
        self.bias = bias
        self.weights_init = weights_init
        if isinstance(weights_init, str):
            self.weights_init = initializations.get(weights_init)()
        self.trainable = trainable
        self.restore = restore
        self.reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                _w = _linear([inputs, state],
                    2 * self._num_units, True, 1.0, self.weights_init,
                    self.trainable, self.restore, self.reuse)
                r, u = array_ops.split(value=_w, num_or_size_splits=2, axis=1)
                r, u = self._inner_activation(r), self._inner_activation(u)
            with tf.variable_scope("Candidate"):
                c = self._activation(
                    _linear([inputs, r * state], self._num_units, True, 0.,
                            self.weights_init, self.trainable, self.restore,
                            self.reuse))
            new_h = u * state + (1 - u) * c

            self.W, self.b = list(), list()
            # Retrieve RNN Variables
            with tf.variable_scope('Gates/Linear', reuse=True):
                self.W.append(tf.get_variable('Matrix'))
                self.b.append(tf.get_variable('Bias'))
            with tf.variable_scope('Candidate/Linear', reuse=True):
                self.W.append(tf.get_variable('Matrix'))
                self.b.append(tf.get_variable('Bias'))

        return new_h, new_h


class DropoutWrapper(core_rnn_cell.RNNCell):
    """Operator adding dropout to inputs and outputs of the given cell."""

    def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
                 seed=None):
        """Create a cell with added input and/or output dropout.

        Dropout is never used on the state.

        Arguments:
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
        if not isinstance(cell, core_rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")
        if (isinstance(input_keep_prob, float) and
                not (input_keep_prob >= 0.0 and input_keep_prob <= 1.0)):
            raise ValueError(
                "Parameter input_keep_prob must be between 0 and 1: %d"
                % input_keep_prob)
        if (isinstance(output_keep_prob, float) and
                not (output_keep_prob >= 0.0 and output_keep_prob <= 1.0)):
            raise ValueError(
                "Parameter output_keep_prob must be between 0 and 1: %d"
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

        is_training = config.get_training_mode()

        if (not isinstance(self._input_keep_prob, float) or
                    self._input_keep_prob < 1):
            inputs = tf.cond(is_training,
                lambda: tf.nn.dropout(inputs,
                                      self._input_keep_prob,
                                      seed=self._seed),
                lambda: inputs)
        output, new_state = self._cell(inputs, state)
        if (not isinstance(self._output_keep_prob, float) or
                    self._output_keep_prob < 1):
            output = tf.cond(is_training,
                lambda: tf.nn.dropout(output,
                                      self._output_keep_prob,
                                      seed=self._seed),
                lambda: output)
        return output, new_state


# --------------------
#   TensorFlow Utils
# --------------------

def _linear(args, output_size, bias, bias_start=0.0, weights_init=None,
            trainable=True, restore=True, reuse=False, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Arguments:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not is_sequence(args):
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

    # Now the computation.
    with tf.variable_scope(scope or "Linear", reuse=reuse):
        matrix = va.variable("Matrix", [total_arg_size, output_size],
                             initializer=weights_init, trainable=trainable,
                             restore=restore)
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(array_ops.concat(args, 1), matrix)
        if not bias:
            return res
        bias_term = va.variable(
            "Bias", [output_size],
            initializer=tf.constant_initializer(bias_start),
            trainable=trainable, restore=restore)
    return res + bias_term


def retrieve_seq_length_op(data):
    """ An op to compute the length of a sequence. 0 are masked. """
    with tf.name_scope('GetLength'):
        used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
    return length


def advanced_indexing_op(input, index):
    """ Advanced Indexing for Sequences. """
    batch_size = tf.shape(input)[0]
    max_length = int(input.get_shape()[1])
    dim_size = int(input.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (index - 1)
    flat = tf.reshape(input, [-1, dim_size])
    relevant = tf.gather(flat, index)
    return relevant
