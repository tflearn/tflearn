from __future__ import division, print_function, absolute_import

from datetime import datetime
import os
import math
import numpy as np
import time

import tensorflow.compat.v1 as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.contrib.tensor_forest.python.ops import data_ops
from tensorflow.python.ops import state_ops, array_ops, math_ops

from ...utils import validate_dim, read_tensor_in_checkpoint
from ...data_utils import get_num_features, get_num_classes, get_num_sample
from ...data_flow import generate_data_tensor
from ..base import BaseEstimator


class ForestEstimator(BaseEstimator):
    """ [WIP] ForesEstimator
    """

    def __init__(self, n_estimators=100, max_nodes=10000,
                 split_after_samples=25, min_samples_split=2,
                 bagging_fraction=1.0, num_splits_to_consider=0,
                 feature_bagging_fraction=1.0, max_fertile_nodes=0,
                 valid_leaf_threshold=1, dominate_method='bootstrap',
                 dominate_fraction=0.99, regression=False, n_classes=None,
                 n_features=None, metric=None, log_dir='/tmp/tflearn_logs/',
                 global_step=None, session=None, graph=None, name=None):

        super(ForestEstimator, self).__init__(metric=metric,
                                              log_dir=log_dir,
                                              global_step=global_step,
                                              session=session,
                                              graph=graph,
                                              name=name)
        self._estimator_built = False

        # Tree Params
        self.n_estimators = n_estimators
        self.max_nodes = max_nodes
        self.split_after_samples = split_after_samples
        self.min_samples_split = min_samples_split
        self.regression = regression
        self.n_classes = n_classes
        self.n_features = n_features
        self.bagging_fraction = bagging_fraction
        self.num_splits_to_consider = num_splits_to_consider
        self.feature_bagging_fraction = feature_bagging_fraction
        self.max_fertile_nodes = max_fertile_nodes
        self.valid_leaf_threshold = valid_leaf_threshold
        self.dominate_method = dominate_method
        self.dominate_fraction = dominate_fraction

    def _build_estimator(self, X=None, Y=None):

        if not self._estimator_built:
            if self.n_features is None:
                self.n_features = get_num_features(X)
            if self.n_classes is None:
                if not self.regression:
                    self.n_classes = get_num_classes(Y)
                else:
                    self.n_classes = get_num_features(Y)

            # Reload params from checkpoint if available
            if self._to_be_restored and self.n_features is None:
                self.n_features = read_tensor_in_checkpoint(
                    'n_features', self._to_be_restored)
            if self._to_be_restored and self.n_classes is None:
                self.n_classes = read_tensor_in_checkpoint(
                    'n_classes', self._to_be_restored)

            # Purity checks
            if self.n_classes is None:
                raise ValueError("'n_classes' cannot be None.")
            if self.n_features is None:
                raise ValueError("'n_features' cannot be None.")

            # Persistent Parameters
            tf.Variable(self.n_classes, dtype=tf.int32, name='n_classes')
            tf.Variable(self.n_features, dtype=tf.int32, name='n_features')

            # Random Forest Parameters
            self.params = tensor_forest.ForestHParams(
                num_classes=self.n_classes, num_features=self.n_features,
                num_trees=self.n_estimators, max_nodes=self.max_nodes,
                split_after_samples=self.split_after_samples,
                min_split_samples=self.min_samples_split,
                regression=self.regression,
                bagging_fraction=self.bagging_fraction,
                num_splits_to_consider=self.num_splits_to_consider,
                feature_bagging_fraction=self.feature_bagging_fraction,
                max_fertile_nodes=self.max_fertile_nodes,
                valid_leaf_threshold=self.valid_leaf_threshold,
                dominate_method=self.dominate_method,
                dominate_fraction=self.dominate_fraction).fill()
            self.forest_graph = tensor_forest.RandomForestGraphs(self.params)
            self._estimator_built = True

    def fit(self, X, Y, batch_size=1024, shuffle=True, display_step=500,
            n_jobs=1, max_steps=None):
        """ fit.

        Build a forest of trees from the data provided.

        Arguments:
            X: `Tensor` or `Tensor list`. The input data. It must be a list of
                `Tensor` in case of multiple inputs.
            Y: `Tensor`. The labels/targets tensor.
            batch_size: `int`. The batch size.
            shuffle: ` bool`. If True, data are shuffled.
            display_step: `int`. The step to display training information.
            n_jobs: `int`. The number of jobs to use for the computation.
            max_steps: `int`. Maximum number of optimization steps to run.

        """

        with self.graph.as_default():

            # Verify data dimension
            validate_dim(X, max_dim=2, min_dim=2, var_name='X')
            if not self.regression:
                validate_dim(Y, max_dim=1, min_dim=1, var_name='Y')
            else:
                validate_dim(Y, min_dim=1, var_name='Y')

            # Get data size
            num_samples = get_num_sample(X)

            # Build Tree Graph
            self._build_estimator(X, Y)

            # Generate Data Tensors. Be aware that every fit with different
            # data will re-create a data tensor.
            if self._train.get_params('X') != hex(id(X)) or \
                self._train.get_params('Y') != hex(id(Y)) or \
                self._train.get_params('batch_size') != batch_size or \
                not self._train.is_ready:

                X, Y, cr = generate_data_tensor(X, Y, batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_threads=8)
                X, _, spec = data_ops.ParseDataTensorOrDict(X)
                Y = data_ops.ParseLabelTensorOrDict(Y)

                self._train_op = tf.group(
                    self.forest_graph.training_graph(X, Y, num_trainers=n_jobs),
                    state_ops.assign_add(self.global_step, 1))
                self._loss_op = self.forest_graph.training_loss(X, Y)
                self._build_fit(X, Y, batch_size)

                # Start QueueRunners
                tf.train.start_queue_runners(sess=self.session)
                if cr: cr.launch_threads(self.session)

                self._init_graph()

            gstep = self.global_step.eval(session=self.session)

            last_loss = []
            loss_val = None
            step = 0

            # Set step to -1 to exit training
            while True:
                # Monitor loss
                last_loss.append(loss_val)
                if len(last_loss) > 10: last_loss.pop(0)

                start_time = time.time()
                if (step) % display_step == 0:
                    _, loss_val = self.session.run(
                        [self._train_op, self._loss_op])  # TODO: Add acc
                else:
                    _, loss_val = self.session.run([self._train_op, self._loss_op])
                duration = time.time() - start_time

                if (step) % display_step == 0:
                    examples_per_sec = batch_size / duration
                    sec_per_batch = duration
                    if self.metric:
                        format_str = '%s: step %d, loss = %.2f, acc = %.2f, ' \
                                     '(%.1f examples/sec; %.3f sec/batch)'
                        print(format_str % (
                            datetime.now(), step + gstep, loss_val,
                            examples_per_sec, sec_per_batch))
                    else:
                        format_str = '%s: step %d, loss = %.2f, ' \
                                     '(%.1f examples/sec; %.3f sec/batch)'
                        print(format_str % (
                            datetime.now(), step + gstep, loss_val,
                            examples_per_sec, sec_per_batch))

                step += 1

                # Automatic stop after ten flat loss
                if len(last_loss) == 10 and len(set(last_loss)) <= 1 and not max_steps:
                    break

                # Max Steps stop
                if max_steps:
                    if step == max_steps:
                        break

            save_path = os.path.join(self.log_dir, 'randomforest.ckpt')
            self.saver.save(sess=self.session,
                            save_path=save_path,
                            global_step=self.global_step)

    def predict(self, X):
        """ predict.

        Predict scores for X.

        Arguments:
            X: `1-D Array` or `2-D Array` of shape (n_samples, n_features).
                The sample(s) to predict.

        Return:
            `Array` or `list` of `Array`. Prediction scores result.

        """
        with self.graph.as_default():
            # Build Tree Graph
            self._build_estimator()
            if not self._pred.is_ready:
                input = tf.placeholder(tf.float32, name='pred_input',
                                       shape=[None, self.n_features])
                output, _, _ = self.forest_graph.inference_graph(input)
                self._build_pred(input, output)
            return self.session.run(self._pred.output_tensor,
                                    feed_dict={self._pred.input_tensor: X})

    def evaluate(self, X, Y, metric, batch_size=None):
        """ evaluate.

        Evaluate the forest model with the given data and metric.

        Arguments:
            X: `2-D Array` of shape (n_samples, n_features).
                The input data to evaluate on.
            Y: `1-D Array` of shape (n_samples). The labels/targets data.
            metric: `func` returning a `Tensor`. The metric function.
            batch_size: `int`. If specified, process the data by batch.

        Return:
            The metric value.

        """

        with self.graph.as_default():
            # Verify data dimension
            validate_dim(X, max_dim=2, min_dim=2, var_name='X')
            if not self.regression:
                validate_dim(Y, max_dim=1, min_dim=1, var_name='Y')
            else:
                validate_dim(Y, min_dim=1, var_name='Y')

            # Get data size
            num_samples = get_num_sample(X)
            capacity = None
            if batch_size is None:
                batch_size = num_samples
                capacity = 1

            # Build Tree Graph
            self._build_estimator(X, Y)

            # Generate Data Tensors. Be aware that every eval with different
            # data will re-create a data tensor.
            if self._eval.get_params('X') != hex(id(X)) or \
                self._eval.get_params('Y') != hex(id(Y)) or \
                self._eval.get_params('batch_size') != batch_size or \
                self._eval.get_params('metric') != metric or \
                not self._eval.is_ready:

                X, Y, cr = generate_data_tensor(X, Y, batch_size=batch_size,
                                                shuffle=False,
                                                num_threads=8,
                                                capacity=capacity)
                X, _, spec = data_ops.ParseDataTensorOrDict(X)
                Y = data_ops.ParseLabelTensorOrDict(Y)

                if not self.params.regression:
                    Y = math_ops.to_float(array_ops.one_hot(math_ops.to_int64(
                        array_ops.squeeze(Y)), self.params.n_classes, 1, 0))
                    Y = tf.reshape(Y, [-1, self.n_classes])

                pred, _, _ = self.forest_graph.inference_graph(X)
                self._eval_op = metric(pred, Y)
                self._build_eval(X, Y, metric, batch_size)

                # Start QueueRunners
                tf.train.start_queue_runners(sess=self.session)
                if cr: cr.launch_threads(self.session)

            n_batches = int(math.ceil(float(num_samples) / batch_size))

            m = 0.
            for i in range(n_batches):
                m += self.session.run(self._eval_op) / n_batches
            return m

    def save(self, save_path):
        """ save.

        Save model to the given path.

        Args:
            path: `str`. The path to save the model.

        """
        if not self._estimator_built:
            with self.graph.as_default():
                self._build_estimator()
        self.saver.save(self.session, os.path.abspath(save_path))

    def load(self, load_path):
        """ load.

        Restore model from the given path.

        Args:
            path: `str`. The model path.

        """
        with self.graph.as_default():
            self.session = tf.Session()
            if self._estimator_built:
                self.saver.restore(self.session, os.path.abspath(load_path))
            else:
                self._to_be_restored = os.path.abspath(load_path)


class RandomForestClassifier(ForestEstimator):
    """ [WIP] Random Forest Classifier.

    """

    def __init__(self, n_estimators=10, max_nodes=100,
                 split_after_samples=25, n_classes=None, n_features=None,
                 metric=None, log_dir='/tmp/tflearn_logs/', global_step=None,
                 session=None, graph=None, name=None):
        super(RandomForestClassifier, self).__init__(
            n_estimators=n_estimators, max_nodes=max_nodes,
            split_after_samples=split_after_samples, regression=False,
            n_classes=n_classes, n_features=n_features, metric=metric,
            log_dir=log_dir, global_step=global_step, session=session,
            graph=graph, name=name)

    def predict(self, X):
        """ predict.

        Predict class for X.

        Arguments:
            X: array-like or sparse matrix of shape = [n_samples, n_features]
                The input samples. Internally, its dtype will be converted to
                ``dtype=np.float32``. If a sparse matrix is provided, it will be
                converted into a sparse ``csr_matrix``.
        Returns:
            Y: array of shape = [n_samples] or [n_samples, n_outputs]
                The predicted classes.
        """
        sc = super(RandomForestClassifier, self)
        return np.argmax(sc.predict(X), axis=1)

    def predict_proba(self, X):
        """ predict_proba.

        Predict class probablities for X.

        Arguments:
            X: array-like or sparse matrix of shape = [n_samples, n_features]
                The input samples. Internally, its dtype will be converted to
                ``dtype=np.float32``. If a sparse matrix is provided, it will be
                converted into a sparse ``csr_matrix``.

        Returns:
            p : array of shape = [n_samples, n_classes], or a list of n_outputs
                such arrays if n_outputs > 1.
                The class probabilities of the input samples. The order of the
                classes corresponds to that in the attribute `classes_`.
        """
        sc = super(RandomForestClassifier, self)
        return sc.predict(X)

    def predict_log_proba(self, X):
        """ predict_log_proba.

        Predict class log-probabilities for X.

        Arguments:
            X: array-like or sparse matrix of shape = [n_samples, n_features]
                The input samples. Internally, its dtype will be converted to
                ``dtype=np.float32``. If a sparse matrix is provided, it will be
                converted into a sparse ``csr_matrix``.
        Returns:
            p: array of shape = [n_samples, n_classes], or a list of n_outputs
                such arrays if n_outputs > 1.
                The class probabilities of the input samples. The order of the
                classes corresponds to that in the attribute `classes_`.
        """
        return np.log(self.predict_proba(X))


class RandomForestRegressor(ForestEstimator):
    """ [WIP] Random Forest Regressor.

    """

    def __init__(self, n_estimators=10, max_nodes=100,
                 split_after_samples=25, n_features=None, num_output=None,
                 metric=None, log_dir='/tmp/tflearn_logs/', global_step=None,
                 session=None, graph=None, name=None):
        super(RandomForestRegressor, self).__init__(
            n_estimators=n_estimators, max_nodes=max_nodes,
            split_after_samples=split_after_samples, regression=True,
            n_classes=num_output, n_features=n_features, metric=metric,
            log_dir=log_dir, global_step=global_step, session=session,
            graph=graph, name=name)
