from __future__ import division, print_function, absolute_import

from .utils import get_from_module
import tensorflow as tf


def get(identifier):
    return get_from_module(identifier, globals(), 'optimizer')

"""
Metric classes are meant to be used with TFLearn models (such as DNN). For
direct operations to be used with Tensorflow, see below (accuracy_op, ...).
"""

# --------------
# Metric classes
# --------------


class Metric(object):
    """ Base Metric Class.

    Metric class is meant to be used by TFLearn models class. It can be
    first initialized with desired parameters, and a model class will
    build it later using the given network output and targets.

    Attributes:
        tensor: `Tensor`. The metric tensor.

    """
    def __init__(self, name=None):
        self.name = name
        self.tensor = None
        self.built = False

    def build(self, predictions, targets, inputs):
        """ build.

        Build metric method, with common arguments to all Metrics.

        Arguments:
            prediction: `Tensor`. The network to perform prediction.
            targets: `Tensor`. The targets (labels).
            inputs: `Tensor`. The input data.

        """
        raise NotImplementedError

    def get_tensor(self):
        """ get_tensor.

        Get the metric tensor.

        Returns:
            The metric `Tensor`.

        """
        if not self.built:
            raise Exception("Metric class Tensor hasn't be built. 'build' "
                            "method must be invoked before using 'get_tensor'.")
        return self.tensor


class Accuracy(Metric):
    """ Accuracy.

    Computes the model accuracy.  The target predictions are assumed
    to be logits.  

    If the predictions tensor is 1D (ie shape [?], or [?, 1]), then the 
    labels are assumed to be binary (cast as float32), and accuracy is
    computed based on the average number of equal binary outcomes,
    thresholding predictions on logits > 0.  

    Otherwise, accuracy is computed based on categorical outcomes,
    and assumes the inputs (both the model predictions and the labels)
    are one-hot encoded.  tf.argmax is used to obtain categorical
    predictions, for equality comparison.

    Examples:
        ```python
        # To be used with TFLearn estimators
        acc = Accuracy()
        regression = regression(net, metric=acc)
        ```

    Arguments:
        name: The name to display.

    """

    def __init__(self, name=None):
        super(Accuracy, self).__init__(name)

    def build(self, predictions, targets, inputs=None):
        """ Build accuracy, comparing predictions and targets. """
        self.built = True
        pshape = predictions.get_shape()
        if len(pshape)==1 or (len(pshape)==2 and int(pshape[1])==1):
            self.name = self.name or "binary_acc"   # clearly indicate binary accuracy being used
            self.tensor = binary_accuracy_op(predictions, targets)
        else:
            self.name = self.name or "acc"   	    # traditional categorical accuracy
            self.tensor = accuracy_op(predictions, targets)
        # Add a special name to that tensor, to be used by monitors
        self.tensor.m_name = self.name

accuracy = Accuracy

class Top_k(Metric):
    """ Top-k.

    Computes Top-k mean accuracy (whether the targets are in the top 'K'
    predictions).

    Examples:
        ```python
        # To be used with TFLearn estimators
        top5 = Top_k(k=5)
        regression = regression(net, metric=top5)
        ```

    Arguments:
        k: `int`. Number of top elements to look at for computing precision.
        name: The name to display.

    """

    def __init__(self, k=1, name=None):
        super(Top_k, self).__init__(name)
        self.name = "top" + str(k) if not name else name
        self.k = k

    def build(self, predictions, targets, inputs=None):
        """ Build top-k accuracy, comparing top-k predictions and targets. """
        self.built = True
        self.tensor = top_k_op(predictions, targets, self.k)
        # Add a special name to that tensor, to be used by monitors
        self.tensor.m_name = self.name

top_k = Top_k


class R2(Metric):
    """ Standard Error.

    Computes coefficient of determination. Useful to evaluate a linear
    regression.

    Examples:
        ```python
        # To be used with TFLearn estimators
        r2 = R2()
        regression = regression(net, metric=r2)
        ```

    Arguments:
        name: The name to display.

    """

    def __init__(self, name=None):
        super(R2, self).__init__(name)
        self.name = "R2" if not name else name

    def build(self, predictions, targets, inputs=None):
        """ Build standard error tensor. """
        self.built = True
        self.tensor = r2_op(predictions, targets)
        # Add a special name to that tensor, to be used by monitors
        self.tensor.m_name = self.name


class WeightedR2(Metric):
    """ Weighted Standard Error.

    Computes coefficient of determination. Useful to evaluate a linear
    regression.

    Examples:
        ```python
        # To be used with TFLearn estimators
        weighted_r2 = WeightedR2()
        regression = regression(net, metric=weighted_r2)
        ```

    Arguments:
        name: The name to display.

    """

    def __init__(self, name=None):
        super(WeightedR2, self).__init__(name)
        self.name = "R2" if not name else name

    def build(self, predictions, targets, inputs):
        """ Build standard error tensor. """
        self.built = True
        self.tensor = weighted_r2_op(predictions, targets, inputs)
        # Add a special name to that tensor, to be used by monitors
        self.tensor.m_name = self.name


class Prediction_Counts(Metric):
    """ Prints the count of each category of prediction that is present in the predictions.
    Can be useful to see, for example, to see if the model only gives one type of predictions,
    or if the predictions given are in the expected proportions """

    def __init__(self, inner_metric, name=None):
        super(Prediction_Counts, self).__init__(name)
        self.inner_metric = inner_metric

    def build(self, predictions, targets, inputs=None):
        """ Prints the number of each kind of prediction """
        self.built = True
        pshape = predictions.get_shape()
        self.inner_metric.build(predictions, targets, inputs)

        with tf.name_scope(self.name):
            if len(pshape) == 1 or (len(pshape) == 2 and int(pshape[1]) == 1):
                self.name = self.name or "binary_prediction_counts"
                y, idx, count = tf.unique_with_counts(tf.argmax(predictions))
                self.tensor = tf.Print(self.inner_metric, [y, count], name=self.inner_metric.name)
            else:
                self.name = self.name or "categorical_prediction_counts"
                y, idx, count = tf.unique_with_counts(tf.argmax(predictions, dimension=1))
                self.tensor = tf.Print(self.inner_metric.tensor, [y, count], name=self.inner_metric.name)

prediction_counts = Prediction_Counts


# ----------
# Metric ops
# ----------


def accuracy_op(predictions, targets):
    """ accuracy_op.

    An op that calculates mean accuracy, assuming predictiosn are targets
    are both one-hot encoded.

    Examples:
        ```python
        input_data = placeholder(shape=[None, 784])
        y_pred = my_network(input_data) # Apply some ops
        y_true = placeholder(shape=[None, 10]) # Labels
        acc_op = accuracy_op(y_pred, y_true)

        # Calculate accuracy by feeding data X and labels Y
        accuracy = sess.run(acc_op, feed_dict={input_data: X, y_true: Y})
        ```

    Arguments:
        predictions: `Tensor`.
        targets: `Tensor`.

    Returns:
        `Float`. The mean accuracy.

    """
    if not isinstance(targets, tf.Tensor):
        raise ValueError("mean_accuracy 'input' argument only accepts type "
                         "Tensor, '" + str(type(input)) + "' given.")

    with tf.name_scope('Accuracy'):
        correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(targets, 1))
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return acc


def binary_accuracy_op(predictions, targets):
    """ binary_accuracy_op.

    An op that calculates mean accuracy, assuming predictions are logits, and
    targets are binary encoded (and represented as int32).

    Examples:
        ```python
        input_data = placeholder(shape=[None, 784])
        y_pred = my_network(input_data) # Apply some ops
        y_true = placeholder(shape=[None, 10]) # Labels
        acc_op = binary_accuracy_op(y_pred, y_true)

        # Calculate accuracy by feeding data X and labels Y
        binary_accuracy = sess.run(acc_op, feed_dict={input_data: X, y_true: Y})
        ```

    Arguments:
        predictions: `Tensor` of `float` type.
        targets: `Tensor` of `float` type.

    Returns:
        `Float`. The mean accuracy.

    """
    if not isinstance(targets, tf.Tensor):
        raise ValueError("mean_accuracy 'input' argument only accepts type "
                         "Tensor, '" + str(type(input)) + "' given.")

    with tf.name_scope('BinaryAccuracy'):
        predictions = tf.cast(tf.greater(predictions, 0), tf.float32)
        correct_pred = tf.equal(predictions, tf.cast(targets, tf.float32))
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return acc


def top_k_op(predictions, targets, k=1):
    """ top_k_op.

    An op that calculates top-k mean accuracy.

    Examples:
        ```python
        input_data = placeholder(shape=[None, 784])
        y_pred = my_network(input_data) # Apply some ops
        y_true = placeholder(shape=[None, 10]) # Labels
        top3_op = top_k_op(y_pred, y_true, 3)

        # Calculate Top-3 accuracy by feeding data X and labels Y
        top3_accuracy = sess.run(top3_op, feed_dict={input_data: X, y_true: Y})
        ```

    Arguments:
        predictions: `Tensor`.
        targets: `Tensor`.
        k: `int`. Number of top elements to look at for computing precision.

    Returns:
        `Float`. The top-k mean accuracy.

    """
    with tf.name_scope('Top_' + str(k)):
        targets = tf.cast(targets, tf.int32)
        correct_pred = tf.nn.in_top_k(predictions, tf.argmax(targets, 1), k)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return acc


def r2_op(predictions, targets):
    """ r2_op.

    An op that calculates the standard error.

    Examples:
        ```python
        input_data = placeholder(shape=[None, 784])
        y_pred = my_network(input_data) # Apply some ops
        y_true = placeholder(shape=[None, 10]) # Labels
        stderr_op = r2_op(y_pred, y_true)

        # Calculate standard error by feeding data X and labels Y
        std_error = sess.run(stderr_op, feed_dict={input_data: X, y_true: Y})
        ```

    Arguments:
        predictions: `Tensor`.
        targets: `Tensor`.

    Returns:
        `Float`. The standard error.

    """
    with tf.name_scope('StandardError'):
        a = tf.reduce_sum(tf.square(predictions))
        b = tf.reduce_sum(tf.square(targets))
        return tf.divide(a, b)


def weighted_r2_op(predictions, targets, inputs):
    """ weighted_r2_op.

    An op that calculates the standard error.

    Examples:
        ```python
        input_data = placeholder(shape=[None, 784])
        y_pred = my_network(input_data) # Apply some ops
        y_true = placeholder(shape=[None, 10]) # Labels
        stderr_op = weighted_r2_op(y_pred, y_true, input_data)

        # Calculate standard error by feeding data X and labels Y
        std_error = sess.run(stderr_op, feed_dict={input_data: X, y_true: Y})
        ```

    Arguments:
        predictions: `Tensor`.
        targets: `Tensor`.
        inputs: `Tensor`.

    Returns:
        `Float`. The standard error.

    """
    with tf.name_scope('WeightedStandardError'):
        if hasattr(inputs, '__len__'):
            inputs = tf.add_n(inputs)
        if inputs.get_shape().as_list() != targets.get_shape().as_list():
            raise Exception("Weighted R2 metric requires Inputs and Targets to "
                            "have same shape.")
        a = tf.reduce_sum(tf.square(predictions - inputs))
        b = tf.reduce_sum(tf.square(targets - inputs))
        return tf.divide(a, b)
