from __future__ import division, print_function, absolute_import

import math
import tensorflow as tf
from .utils import get_from_module


def get(identifier):
    if hasattr(identifier, '__call__'):
        return identifier
    else:
        return get_from_module(identifier, globals(), 'initialization')


def zeros(shape=None, dtype=tf.float32, seed=None):
    """ Zeros.

    Initialize a tensor with all elements set to zero.

    Arguments:
        shape: List of `int`. A shape to initialize a Tensor (optional).
        dtype: The tensor data type.

    Returns:
        The Initializer, or an initialized `Tensor` if a shape is specified.

    """
    if shape:
        return tf.zeros(shape, dtype=dtype)
    else:
        return tf.constant_initializer(0.)


def uniform(shape=None, minval=0, maxval=None, dtype=tf.float32, seed=None):
    """ Uniform.

    Initialization with random values from a uniform distribution.

    The generated values follow a uniform distribution in the range
    `[minval, maxval)`. The lower bound `minval` is included in the range,
    while the upper bound `maxval` is excluded.

    For floats, the default range is `[0, 1)`.  For ints, at least `maxval`
    must be specified explicitly.

    In the integer case, the random integers are slightly biased unless
    `maxval - minval` is an exact power of two.  The bias is small for values of
    `maxval - minval` significantly smaller than the range of the output (either
    `2**32` or `2**64`).

    Arguments:
        shape: List of `int`. A shape to initialize a Tensor (optional).
        dtype: The tensor data type. Only float are supported.
        seed: `int`. Used to create a random seed for the distribution.

    Returns:
        The Initializer, or an initialized `Tensor` if shape is specified.

    """
    if shape:
        return tf.random_uniform(shape, minval=minval, maxval=maxval,
                                 seed=seed, dtype=dtype)
    else:
        return tf.random_uniform_initializer(minval=minval, maxval=maxval,
                                             seed=seed, dtype=dtype)


def uniform_scaling(shape=None, factor=1.0, dtype=tf.float32, seed=None):
    """ Uniform Scaling.

    Initialization with random values from uniform distribution without scaling
    variance.

    When initializing a deep network, it is in principle advantageous to keep
    the scale of the input variance constant, so it does not explode or diminish
    by reaching the final layer. If the input is `x` and the operation `x * W`,
    and we want to initialize `W` uniformly at random, we need to pick `W` from

      [-sqrt(3) / sqrt(dim), sqrt(3) / sqrt(dim)]

    to keep the scale intact, where `dim = W.shape[0]` (the size of the input).
    A similar calculation for convolutional networks gives an analogous result
    with `dim` equal to the product of the first 3 dimensions.  When
    nonlinearities are present, we need to multiply this by a constant `factor`.
    See [Sussillo et al., 2014](https://arxiv.org/abs/1412.6558)
    ([pdf](http://arxiv.org/pdf/1412.6558.pdf)) for deeper motivation, experiments
    and the calculation of constants. In section 2.3 there, the constants were
    numerically computed: for a linear layer it's 1.0, relu: ~1.43, tanh: ~1.15.

    Arguments:
        shape: List of `int`. A shape to initialize a Tensor (optional).
        factor: `float`. A multiplicative factor by which the values will be
            scaled.
        dtype: The tensor data type. Only float are supported.
        seed: `int`. Used to create a random seed for the distribution.

    Returns:
        The Initializer, or an initialized `Tensor` if shape is specified.

    """
    if shape:
        input_size = 1.0
        for dim in shape[:-1]:
          input_size *= float(dim)
        max_val = math.sqrt(3 / input_size) * factor
        return tf.random_ops.random_uniform(shape, -max_val, max_val,
                                            dtype, seed=seed)
    else:
        return tf.uniform_unit_scaling_initializer(seed=seed, dtype=dtype)


def normal(shape=None, mean=0.0, stddev=0.02, dtype=tf.float32, seed=None):
    """ Normal.

    Initialization with random values from a normal distribution.

    Arguments:
        shape: List of `int`. A shape to initialize a Tensor (optional).
        mean: Same as `dtype`. The mean of the truncated normal distribution.
        stddev: Same as `dtype`. The standard deviation of the truncated
            normal distribution.
        dtype: The tensor data type.
        seed: `int`. Used to create a random seed for the distribution.

    Returns:
        The Initializer, or an initialized `Tensor` if shape is specified.

    """
    if shape:
        return tf.random_normal(shape, mean=mean, stddev=stddev, seed=seed,
                                dtype=dtype)
    else:
        return tf.random_normal_initializer(mean=mean, stddev=stddev,
                                            seed=seed, dtype=dtype)


def truncated_normal(shape=None, mean=0.0, stddev=0.02, dtype=tf.float32,
                     seed=None):
    """ Truncated Normal.

    Initialization with random values from a normal truncated distribution.

    The generated values follow a normal distribution with specified mean and
    standard deviation, except that values whose magnitude is more than 2 standard
    deviations from the mean are dropped and re-picked.

    Arguments:
        shape: List of `int`. A shape to initialize a Tensor (optional).
        mean: Same as `dtype`. The mean of the truncated normal distribution.
        stddev: Same as `dtype`. The standard deviation of the truncated
            normal distribution.
        dtype: The tensor data type.
        seed: `int`. Used to create a random seed for the distribution.

    Returns:
        The Initializer, or an initialized `Tensor` if shape is specified.

    """
    if shape:
        return tf.truncated_normal(shape=shape, mean=mean, stddev=stddev,
                                   seed=seed, dtype=dtype)
    else:
        return tf.truncated_normal_initializer(mean=mean, stddev=stddev,
                                               seed=seed, dtype=dtype)
