from __future__ import division, print_function, absolute_import

import os
import tensorflow.compat.v1 as tf
from tensorflow.python.ops import resources

from ..utils import validate_func


class GraphBranch(object):
    """ A graph branch class used for building part of an Estimator graph.
    """
    def __init__(self, input_tensor=None, output_tensor=None, params=None):
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.params = params if params is not None else dict()
        self._is_ready = False
        if input_tensor is not None and output_tensor is not None:
            self._is_ready = True

    def build(self, input_tensor, output_tensor, params=None):
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.params = params if params is not None else dict()
        self._is_ready = True

    @property
    def is_ready(self):
        return self._is_ready

    def get_params(self, x):
        if x in self.params.keys():
            return self.params[x]
        else:
            return None


class BaseEstimator(object):

    """ Estimators Graph is only build when fit/predict or evaluate is called.
    """

    def __init__(self, metric=None, log_dir='/tmp/tflearn_logs/',
                 global_step=None, session=None, graph=None, name=None):

        self.name = name

        # Estimator Graph and Session
        self.graph = tf.Graph() if graph is None else graph
        with self.graph.as_default():
            conf = tf.ConfigProto(allow_soft_placement=True)
            self.session = tf.Session(config=conf) if session is None else session
        if global_step is None:
            with self.graph.as_default():
                self.global_step = tf.train.get_or_create_global_step()

        self.metric = validate_func(metric)

        # Estimator Graph Branches
        self._train = GraphBranch()
        self._pred = GraphBranch()
        self._transform = GraphBranch()
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
            init_vars = tf.group(tf.global_variables_initializer(),
                                 resources.initialize_resources(
                                     resources.shared_resources()))
            self.session.run(init_vars)
            self._is_initialized = True
        # Restore weights if needed
        if self._to_be_restored:
            self.saver = tf.train.Saver()
            self.saver.restore(self.session, self._to_be_restored)
            self._to_be_restored = False

    def _init_estimator(self):
        raise NotImplementedError

    # ----------------------
    #  Build Graph Branches
    # ----------------------
    def _build_fit(self, X, Y, batch_size, multi_inputs=False):
        if not self._train._is_ready:
            self._init_graph()
        train_params = {'X': X, 'Y': Y, 'batch_size': batch_size,
                        'multi_inputs': multi_inputs}
        self._train.build(None, None, train_params)

    def _build_pred(self, input_tensor, output_tensor):
        self._pred.build(input_tensor, output_tensor)

    def _build_transform(self, input_tensor, output_tensor):
        self._transform.build(input_tensor, output_tensor)

    def _build_eval(self, X, Y, metric, batch_size, multi_inputs=False):
        eval_params = {'X': X, 'Y': Y, 'batch_size': batch_size,
                       'metric': metric, 'multi_inputs': multi_inputs}
        self._eval.build(None, None, eval_params)

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


class SupervisedEstimator(BaseEstimator):

    def __init__(self, metric=None, log_dir='/tmp/tflearn_logs/',
                 global_step=None, session=None, graph=None, name=None):
        super(SupervisedEstimator, self).__init__(
            metric=metric, log_dir=log_dir, global_step=global_step,
            session=session, graph=graph, name=name)

    def fit(self, X, Y, *args):
        pass
