from __future__ import absolute_import

# Config
from . import config
from .config import is_training, get_training_mode, init_graph

# Import models
from . import models
from .models.dnn import DNN
from .models.generator import SequenceGenerator

# Helpers
from . import helpers
from .helpers.evaluator import Evaluator
from .helpers.trainer import Trainer, TrainOp
from .helpers.regularizer import add_weights_regularizer
from .helpers.summarizer import summarize, summarize_activations, \
    summarize_gradients, summarize_variables, summarize_all

# Predefined ops
from . import metrics
from . import activations
from . import losses
from . import initializations
from . import optimizers
from . import summaries
from . import optimizers
from . import variables
from . import utils
from . import data_utils
from . import collections # Add TFLearn collections to Tensorflow GraphKeys

# Direct ops inclusion
from .optimizers import SGD, AdaGrad, Adam, RMSProp, Momentum, Ftrl
from .activations import linear, tanh, sigmoid, softmax, softplus, softsign,\
    relu, relu6, leaky_relu, prelu, elu
from .variables import variable, get_all_trainable_variable, \
    get_all_variables, get_layer_variables_by_name
from .objectives import categorical_crossentropy, binary_crossentropy, \
    softmax_categorical_crossentropy, hinge_loss, mean_square
from .metrics import Top_k, Accuracy, R2, top_k_op, accuracy_op, r2_op

# Direct layers inclusion
from . import layers
from .layers.conv import conv_2d, max_pool_2d, avg_pool_2d, conv_1d, \
    max_pool_1d, avg_pool_1d, global_avg_pool, shallow_residual_block, deep_residual_block
from .layers.core import input_data, dropout, custom_layer, reshape, \
    flatten, activation, fully_connected, single_unit
from .layers.normalization import batch_normalization, local_response_normalization
from .layers.estimator import regression
from .layers.recurrent import lstm, gru, simple_rnn, bidirectional_rnn, dynamic_rnn, RNNCell, BasicLSTMCell, GRUCell, BasicRNNCell
from .layers.embedding_ops import embedding
from .layers.merge_ops import merge, merge_outputs

# Datasets
from . import datasets

# Utils
from . import data_utils
from . import utils
