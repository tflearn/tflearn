# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf

"""
For handling networks and keep tracks of important parameters, TFLearn is
using Tensorflow collections.
"""
#TODO: Chek if a layer without import tflearn doesn't have problem with those
# Collection for network inputs. Used by `Trainer` class for retrieving all
# data input placeholders.
tf.GraphKeys.INPUTS = 'inputs'

# Collection for network targets. Used by `Trainer` class for retrieving all
# targets (labels) placeholders.
tf.GraphKeys.TARGETS = 'targets'

# Collection for network train ops. Used by `Trainer` class for retrieving all
# optimization processes.
tf.GraphKeys.TRAIN_OPS = 'trainops'

# Collection to retrieve layers variables. Variables are stored according to
# the following pattern: /tf.GraphKeys.LAYER_VARIABLES/layer_name (so there
# will have as many collections as layers with variables).
tf.GraphKeys.LAYER_VARIABLES = 'layer_variables'

# Collection to store all returned tensors for every layer
tf.GraphKeys.LAYER_TENSOR = 'layer_tensor'

# Collection to store all variables that will be restored
tf.GraphKeys.EXCL_RESTORE_VARS = 'restore_variables'

# Collection to store the default graph configuration
tf.GraphKeys.GRAPH_CONFIG = 'graph_config'

# Collection to store all input variable data preprocessing
tf.GraphKeys.DATA_PREP = 'data_preprocessing'

# Collection to store all input variable data preprocessing
tf.GraphKeys.DATA_AUG = 'data_augmentation'

# Collection to store all custom learning rate variable
tf.GraphKeys.LR_VARIABLES = 'lr_variables'
