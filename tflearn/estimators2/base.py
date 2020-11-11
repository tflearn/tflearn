from __future__ import division, print_function, absolute_import

import os
import tensorflow.compat.v1 as tf
from tensorflow.contrib import framework

from ..utils.validate import *


class GraphBranch(object):
    """ A graph branch class used for building part of an Estimator graph.
    """
    def __init__(self, input_tensor=None, output_tensor=None):
        self.input = input_tensor
        self.output = output_tensor
        self._is_ready = False if self.input and self.output else True

    def build(self, input_tensor, output_tensor):
        self.input = input_tensor
        self.output = output_tensor
        self._is_ready = True

    @property
    def is_ready(self):
        return self._is_ready


class BaseEstimator(object):

    def __init__(self, metric=None, log_dir='/tmp/tflearn_logs/',
                 global_step=None, session=None, graph=None, name=None):

        self.name = name

        # Estimator Graph and Session
        self.graph = tf.Graph() if None else graph
        self.session = tf.Session() if None else session
        if global_step is None:
            with self.graph.as_default():
                self.global_step = framework.get_or_create_global_step()

        self.metric = validate_func(metric)

        # Estimator Graph Branches
        self._train = GraphBranch()
        self._pred = GraphBranch()
        self._eval = GraphBranch()

        # Tensor Utils
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        self._is_initialized = False
        self._to_be_restored = False

        # Ops
        self.train_op = None
        self.loss_op = None

    # -----------------
    #  Initializations
    # -----------------
    def _init_graph(self):
        # Initialize all weights
        if not self._is_initialized:
            self.saver = tf.train.Saver()
            self.session.run(tf.global_variables_initializer())
            self._is_initialized = True
        # Restore weights if needed
        if self._to_be_restored:
            self.saver = tf.train.Saver()
            self.saver.restore(self.session, self._to_be_restored)
            self._to_be_restored = False

    # --------------------
    #  Graph Construction
    # --------------------
    def _build_train_graph(self, input_tensor, output_tensor):
        self._train.build(input_tensor, output_tensor)

    def _build_pred_graph(self, input_tensor, output_tensor):
        self._pred.build(input_tensor, output_tensor)

    def _build_eval_graph(self, input_tensor, output_tensor):
        self._eval.build(input_tensor, output_tensor)

    # ---------
    #  Methods
    # ---------
    def fit(self, *args):
        #TODO: Handle multiple fits
        raise NotImplementedError

    def predict(self, *args):
        raise NotImplementedError

    def evaluate(self, *args):
        raise NotImplementedError

    def load(self, *args):
        raise NotImplementedError

    def save(self, *args):
        raise NotImplementedError
