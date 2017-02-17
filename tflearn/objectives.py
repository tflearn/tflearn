from __future__ import division, print_function, absolute_import

import tensorflow as tf

from .config import _EPSILON, _FLOATX
from .utils import get_from_module


def get(identifier):
    return get_from_module(identifier, globals(), 'objective')


def softmax_categorical_crossentropy(y_pred, y_true):
    """ Softmax Categorical Crossentropy.

    Computes softmax cross entropy between y_pred (logits) and
    y_true (labels).

    Measures the probability error in discrete classification tasks in which
    the classes are mutually exclusive (each entry is in exactly one class).
    For example, each CIFAR-10 image is labeled with one and only one label:
    an image can be a dog or a truck, but not both.

    **WARNING:** This op expects unscaled logits, since it performs a `softmax`
    on `y_pred` internally for efficiency.  Do not call this op with the
    output of `softmax`, as it will produce incorrect results.

    `y_pred` and `y_true` must have the same shape `[batch_size, num_classes]`
    and the same dtype (either `float32` or `float64`). It is also required
    that `y_true` (labels) are binary arrays (For example, class 2 out of a
    total of 5 different classes, will be define as [0., 1., 0., 0., 0.])

    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.

    """
    with tf.name_scope("SoftmaxCrossentropy"):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=y_pred, labels=y_true))


def categorical_crossentropy(y_pred, y_true):
    """ Categorical Crossentropy.

    Computes cross entropy between y_pred (logits) and y_true (labels).

    Measures the probability error in discrete classification tasks in which
    the classes are mutually exclusive (each entry is in exactly one class).
    For example, each CIFAR-10 image is labeled with one and only one label:
    an image can be a dog or a truck, but not both.

    `y_pred` and `y_true` must have the same shape `[batch_size, num_classes]`
    and the same dtype (either `float32` or `float64`). It is also required
    that `y_true` (labels) are binary arrays (For example, class 2 out of a
    total of 5 different classes, will be define as [0., 1., 0., 0., 0.])

    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.

    """
    with tf.name_scope("Crossentropy"):
        y_pred /= tf.reduce_sum(y_pred,
                                reduction_indices=len(y_pred.get_shape())-1,
                                keep_dims=True)
        # manual computation of crossentropy
        y_pred = tf.clip_by_value(y_pred, tf.cast(_EPSILON, dtype=_FLOATX),
                                  tf.cast(1.-_EPSILON, dtype=_FLOATX))
        cross_entropy = - tf.reduce_sum(y_true * tf.log(y_pred),
                               reduction_indices=len(y_pred.get_shape())-1)
        return tf.reduce_mean(cross_entropy)


def binary_crossentropy(y_pred, y_true):
    """ Binary Crossentropy.

    Computes sigmoid cross entropy between y_pred (logits) and y_true
    (labels).

    Measures the probability error in discrete classification tasks in which
    each class is independent and not mutually exclusive. For instance,
    one could perform multilabel classification where a picture can contain
    both an elephant and a dog at the same time.

    For brevity, let `x = logits`, `z = targets`.  The logistic loss is

      x - x * z + log(1 + exp(-x))

    To ensure stability and avoid overflow, the implementation uses

      max(x, 0) - x * z + log(1 + exp(-abs(x)))

    `y_pred` and `y_true` must have the same type and shape.

    Arguments:
        y_pred: `Tensor` of `float` type. Predicted values.
        y_true: `Tensor` of `float` type. Targets (labels).

    """
    with tf.name_scope("BinaryCrossentropy"):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=y_pred, labels=y_true))


def mean_square(y_pred, y_true):
    """ Mean Square Loss.

    Arguments:
        y_pred: `Tensor` of `float` type. Predicted values.
        y_true: `Tensor` of `float` type. Targets (labels).

    """
    with tf.name_scope("MeanSquare"):
        return tf.reduce_mean(tf.square(y_pred - y_true))


def hinge_loss(y_pred, y_true):
    """ Hinge Loss.

    Arguments:
        y_pred: `Tensor` of `float` type. Predicted values.
        y_true: `Tensor` of `float` type. Targets (labels).

    """
    with tf.name_scope("HingeLoss"):
        return tf.reduce_mean(tf.maximum(1. - y_true * y_pred, 0.))


def roc_auc_score(y_pred, y_true):
    """ ROC AUC Score.

    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.

    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.

    Measures overall performance for a full range of threshold levels.

    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.

    """
    with tf.name_scope("RocAucScore"):

        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)

        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p     = 3

        difference = tf.zeros_like(pos * neg) + pos - neg - gamma

        masked = tf.boolean_mask(difference, difference < 0.0)

        return tf.reduce_sum(tf.pow(-masked, p))


def weak_cross_entropy_2d(y_pred, y_true, num_classes=None, epsilon=0.0001,
                          head=None):
    """ Weak Crossentropy 2d.

    Calculate the semantic segmentation using weak softmax cross entropy loss.

    Given the prediction `y_pred` shaped as 2d image and the corresponding
    y_true, this calculated the widely used semantic segmentation loss.
    Using `tf.nn.softmax_cross_entropy_with_logits` is currently not supported.
    See https://github.com/tensorflow/tensorflow/issues/2327#issuecomment-224491229

    Arguments:
        y_pred: `tensor, float` - [batch_size, width, height, num_classes].
        y_true: `Labels tensor, int32` - [batch_size, width, height, num_classes].
            The ground truth of your data.
        num_classes: `int`. Number of classes.
        epsilon: `float`. Small number to add to `y_pred`.
        head: `numpy array` - [num_classes]. Weighting the loss of each class.

    Returns:
        Loss tensor of type float.
    """
    if num_classes is None:
        num_classes = y_true.get_shape().as_list()[-1]
        # This only works if shape of y_true is defined
        assert (num_classes is not None)

    with tf.name_scope("weakCrossEntropy2d"):
        y_pred = tf.reshape(y_pred, (-1, num_classes))
        y_pred = y_pred + tf.constant(epsilon, dtype=y_pred.dtype)
        y_true = tf.to_float(tf.reshape(y_true, (-1, num_classes)))

        softmax = tf.nn.softmax(y_pred)

        if head is not None:
            cross_entropy = -tf.reduce_sum(tf.multiply(y_true * tf.log(softmax),
                                                  head), reduction_indices=[1])
        else:
            cross_entropy = -tf.reduce_sum(y_true * tf.log(softmax),
                                           reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name="xentropy_mean")

    return cross_entropy_mean
