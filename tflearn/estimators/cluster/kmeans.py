from __future__ import division, print_function, absolute_import

from datetime import datetime
import os
import math
import numpy as np
import time

import tensorflow.compat.v1 as tf
from tensorflow.contrib.factorization.python.ops import clustering_ops as c_ops
from tensorflow.contrib.tensor_forest.python.ops import data_ops
from tensorflow.python.ops import state_ops, array_ops, math_ops

from ...utils import validate_dim, read_tensor_in_checkpoint, prepare_X
from ...data_utils import get_num_features, get_num_sample
from ...data_flow import generate_data_tensor
from ...distances import euclidean, cosine

from ..base import BaseEstimator


class KMeansBase(BaseEstimator):

    def __init__(self, n_clusters, max_iter=300, init=c_ops.RANDOM_INIT,
                 distance=c_ops.SQUARED_EUCLIDEAN_DISTANCE,
                 metric=None, num_features=None, log_dir='/tmp/tflearn_logs/',
                 global_step=None, session=None, graph=None, name=None):
        super(KMeansBase, self).__init__(
            metric=metric, log_dir=log_dir, global_step=global_step,
            session=session, graph=graph, name=name)

        self._estimator_built = False

        # Params
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.distance = distance
        self.num_features = num_features
        self.use_mini_batch = False

    def _build_estimator(self, X=None):

        if not self._estimator_built:

            if self.num_features is None:
                self.num_features = get_num_features(X)

            # Reload params from checkpoint if available
            if self._to_be_restored and self.num_features is None:
                self.num_features = read_tensor_in_checkpoint(
                    'num_features', self._to_be_restored)
            if self._to_be_restored and self.num_classes is None:
                self.num_classes = read_tensor_in_checkpoint(
                    'num_classes', self._to_be_restored)

            # Purity checks
            if self.num_features is None:
                raise ValueError("'num_features' cannot be None.")

            # Persistent Parameters
            tf.Variable(self.num_features, dtype=tf.int32, name='num_features')

            self._kmeans = c_ops.KMeans(X, self.n_clusters,
                                        initial_clusters=self.init,
                                        distance_metric=self.distance,
                                        use_mini_batch=self.use_mini_batch)
            (self._all_scores, self._cluster_idx, self._scores,
             self._cluster_centers_initialized, self._cluster_centers_vars,
             self._init_op, self._train_op) = self._kmeans.training_graph()

            # fix for cluster_idx being a tuple
            self._cluster_idx = self._cluster_idx[0]
            self.avg_distance = tf.reduce_mean(self._scores)

            self._estimator_built = True
            self._init_graph()

    @property
    def cluster_centers_vars(self):
        if self._estimator_built:
            return self.session.run(self._cluster_centers_vars)
        else:
            return None

    @property
    def cluster_idx(self):
        if self._estimator_built:
            return self.session.run(self._cluster_idx)
        else:
            return None

    @property
    def scores(self):
        if self._estimator_built:
            return self.session.run(self._cluster_centers_vars)
        else:
            return None

    @property
    def all_scores(self):
        if self._estimator_built:
            return self.session.run(self._cluster_centers_vars)
        else:
            return None

    # SKLearn bindings
    @property
    def cluster_centers_(self):
        """ Coordinates of cluster centers. """
        return self.cluster_centers_vars

    @property
    def labels_(self):
        """ Labels of each point. """
        return self.cluster_idx

    @property
    def distances_(self):
        """ Distances of each point to its closest cluster center. """
        return self.session.run(self._scores)

    @property
    def all_distances_(self):
        """ Distances of each point to each cluster center. """
        return self.session.run(self._all_scores)

    def _init_graph(self):
        super(KMeansBase, self)._init_graph()
        # Initialize the kmeans op
        self.session.run(self._init_op)

    def fit(self, X, shuffle=True, display_step=500,
            n_jobs=1, max_steps=None, verbose=0, **kwargs):

        with self.graph.as_default():

            # Verify data dimension
            validate_dim(X, max_dim=2, min_dim=2, var_name='X')

            # Get data size
            num_samples = get_num_sample(X)

            # Set batch size
            if 'batch_size' in kwargs.keys():
                batch_size = kwargs['batch_size']
            else:
                batch_size = num_samples

                # Build Tree Graph
            self._build_estimator(X)

            # Generate Data Tensors. Be aware that every fit with different
            # data will re-create a data tensor.
            if self._train.get_params('X') != hex(id(X)) or \
                self._train.get_params('batch_size') != batch_size or \
                not self._train.is_ready:

                #TODO: raise Exception("Fitting different data not supported")

                X, _, cr = generate_data_tensor(X, X, batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_threads=8)
                X, _, spec = data_ops.ParseDataTensorOrDict(X)

                self._train_op = tf.group(
                    self._train_op,
                    state_ops.assign_add(self.global_step, 1))
                self._loss_op = self.avg_distance
                self._build_fit(X, X, batch_size)

                # Start QueueRunners
                tf.train.start_queue_runners(sess=self.session)
                if cr: cr.launch_threads(self.session)

            gstep = self.global_step.eval(session=self.session)

            last_loss = []
            loss_val = None
            step = 0

            # Set step to -1 to exit training
            while True:
                # Monitor loss
                if loss_val: last_loss.append(loss_val)
                if len(last_loss) > 10: last_loss.pop(0)

                start_time = time.time()
                if (step) % display_step == 0:
                    _, loss_val, idx = self.session.run(
                        [self._train_op, self._loss_op, self._cluster_idx])
                else:
                    _, loss_val, idx = self.session.run([self._train_op,
                                                         self._loss_op,
                                                         self._cluster_idx])
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
                # TODO(aymeric): better stopping.
                if len(last_loss) == 10 and np.var(last_loss) <= 0.01 and not max_steps:
                    break

                # Max Steps stop
                if max_steps:
                    if step == max_steps:
                        break

            # save_path = os.path.join(self.log_dir, 'kmeans.ckpt')
            # self.saver.save(sess=self.session,
            #                 save_path=save_path,
            #                 global_step=self.global_step)

    # ------------
    #  Prediction
    # ------------

    def predict(self, X, with_distances=False):
        """ predict.

        Predict the closest cluster.

        Arguments:
            X: `1-D Array` or `2-D Array` of shape (n_samples, n_features).
                The sample(s) to predict.

        Return:
            cluster_indices or (cluster_indices, distances).

        """

        X, orig_ndim = prepare_X(X, 2, max_dim=2, min_dim=1, debug_msg="X")

        with self.graph.as_default():
            # Build Tree Graph
            self._build_estimator()
            if not self._pred.is_ready:
                input = tf.placeholder(tf.float32, name='pred_input',
                                       shape=[None, self.num_features])
                output = c_ops.nearest_neighbors(
                    input, self._cluster_centers_vars, k=1)
                self._build_pred(input, output)
            indices, distances = self.session.run(self._pred.output_tensor,
                feed_dict={self._pred.input_tensor: X})
            indices = indices[0]
            distances = distances[0]
            if orig_ndim == 1:
                indices = indices[0]
                distances = distances[0]
            if with_distances:
                return indices, distances
            return indices

    def transform(self, X):
        """ transform.

        Transform X to a cluster-distance space.

        Arguments:
            X: `Array` or `list` of `Array`. The sample(s) to transform.

        Returns:
            `Array` of shape (n_clusters). The distance of X to each centroid.

        """

        X, orig_ndim = prepare_X(X, 2, max_dim=2, min_dim=1, debug_msg="X")

        with self.graph.as_default():
            # Build Tree Graph
            self._build_estimator()
            if not self._transform.is_ready:
                input = tf.placeholder(tf.float32, name='transform_input',
                                       shape=[None, self.num_features])
                centers = self._cluster_centers_vars
                centers = tf.reshape(centers, shape=[self.n_clusters,
                                                     self.num_features])

                if self.distance == c_ops.SQUARED_EUCLIDEAN_DISTANCE:
                    dist_fn = euclidean
                elif self.distance == c_ops.COSINE_DISTANCE:
                    dist_fn = cosine
                else:
                    raise Exception("Incorrect distance metric.")

                output = tf.map_fn(
                    lambda x: tf.map_fn(
                        lambda y: dist_fn(x, y),
                        centers),
                    input)

                self._build_transform(input, output)
            distances = self.session.run(self._transform.output_tensor,
                feed_dict={self._transform.input_tensor: X})
            if orig_ndim == 1:
                distances = distances[0]
            return distances

    def save(self, save_path):
        """ save.

        Save model to the given path.

        Args:
            save_path: `str`. The path to save the model.

        """
        if not self._estimator_built:
            with self.graph.as_default():
                self._build_estimator()
        self.saver.save(self.session, os.path.abspath(save_path))

    def load(self, load_path):
        """ load.

        Restore model from the given path.

        Args:
            load_path: `str`. The model path.

        """
        with self.graph.as_default():
            self.session = tf.Session()
            if self._estimator_built:
                self.saver.restore(self.session, os.path.abspath(load_path))
            else:
                self._to_be_restored = os.path.abspath(load_path)


class KMeans(KMeansBase):
    """ KMeans.

    K-Means clustering algorithm.

    """

    def __init__(self, n_clusters, max_iter=300, init=c_ops.RANDOM_INIT,
                 distance=c_ops.SQUARED_EUCLIDEAN_DISTANCE,
                 metric=None, num_features=None, log_dir='/tmp/tflearn_logs/',
                 global_step=None, session=None, graph=None, name=None):
        super(KMeans, self).__init__(
            n_clusters, max_iter=max_iter, init=init, distance=distance,
            metric=metric, num_features=num_features, log_dir=log_dir,
            global_step=global_step, session=session, graph=graph,
            name=name)

    def fit(self, X, shuffle=True, display_step=500, n_jobs=1,
            max_steps=None):
        """ fit.

        Compute the K-Means clustering for the input data.

        Arguments:
            X: `Array` or `list` of `Array` of shape (n_samples, n_features).
                The training data.
            shuffle: `bool`. If True, data are shuffled.
            display_step: `int`. The step to display training information.
            n_jobs: `int`. The number of jobs to use for the computation.
            max_steps: `int`. Maximum number of optimization steps to run.

        """

        super(KMeans, self).fit(X, shuffle=shuffle, display_step=display_step,
                                n_jobs=n_jobs, max_steps=max_steps)


class MiniBatchKMeans(KMeans):
    """ MiniBatchKMeans.

    K-Means clustering algorithm with mini batch.

    """

    def __init__(self, n_clusters, max_iter=300, init=c_ops.RANDOM_INIT,
                 distance=c_ops.SQUARED_EUCLIDEAN_DISTANCE,
                 metric=None, num_features=None, log_dir='/tmp/tflearn_logs/',
                 global_step=None, session=None, graph=None, name=None):
        super(MiniBatchKMeans, self).__init__(
            n_clusters, max_iter=max_iter, init=init, distance=distance,
            metric=metric, num_features=num_features, log_dir=log_dir,
            global_step=global_step, session=session, graph=graph,
            name=name)

        self.use_mini_batch = True

    def fit(self, X, batch_size=1024, shuffle=True, display_step=500,
            n_jobs=1, max_steps=None):
        """ fit.

        Compute the K-Means clustering for the input data.

        Arguments:
            X: `Array` or `list` of `Array` of shape (n_samples, n_features).
                The training data.
            shuffle: `bool`. If True, data are shuffled.
            batch_size: `int`. The batch size.
            display_step: `int`. The step to display training information.
            n_jobs: `int`. The number of jobs to use for the computation.
            max_steps: `int`. Maximum number of optimization steps to run.

        """
        super(KMeans, self).fit(X, shuffle=shuffle, display_step=display_step,
                                n_jobs=n_jobs, max_steps=max_steps,
                                batch_size=batch_size)
