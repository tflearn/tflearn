from __future__ import division, print_function, absolute_import

import math
import tensorflow as tf
try:
    from tensorflow.contrib.layers.python.layers.initializers import \
        xavier_initializer
except Exception:
    xavier_initializer = None
try:
    from tensorflow.contrib.layers.python.layers.initializers import \
        variance_scaling_initializer
except Exception:
    variance_scaling_initializer = None
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


def xavier(uniform=True, seed=None, dtype=tf.float32):
    """ Xavier.

    Returns an initializer performing "Xavier" initialization for weights.

    This initializer is designed to keep the scale of the gradients roughly the
    same in all layers. In uniform distribution this ends up being the range:
    `x = sqrt(6. / (in + out)); [-x, x]` and for normal distribution a standard
    deviation of `sqrt(3. / (in + out))` is used.

    Arguments:
        uniform: Whether to use uniform or normal distributed random
            initialization.
        seed: A Python integer. Used to create random seeds. See
            `set_random_seed` for behavior.
        dtype: The data type. Only floating point types are supported.

    Returns:
        An initializer for a weight matrix.

    References:
        Understanding the difficulty of training deep feedforward neural
        networks. International conference on artificial intelligence and
        statistics. Xavier Glorot and Yoshua Bengio (2010).

    Links:
        [http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf]
        (http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    """
    if xavier_initializer is None:
        raise NotImplementedError("'xavier_initializer' not supported, "
                                  "please update TensorFlow.")
    return xavier_initializer(uniform=uniform, seed=seed, dtype=dtype)


def variance_scaling(factor=2.0, mode='FAN_IN', uniform=False, seed=None,
                     dtype=tf.float32):
    """ Variance Scaling.

    Returns an initializer that generates tensors without scaling variance.

    When initializing a deep network, it is in principle advantageous to keep
    the scale of the input variance constant, so it does not explode or diminish
    by reaching the final layer. This initializer use the following formula:

    ```
    if mode='FAN_IN': # Count only number of input connections.
      n = fan_in
    elif mode='FAN_OUT': # Count only number of output connections.
      n = fan_out
    elif mode='FAN_AVG': # Average number of inputs and output connections.
      n = (fan_in + fan_out)/2.0

      truncated_normal(shape, 0.0, stddev=sqrt(factor / n))
    ```

    To get http://arxiv.org/pdf/1502.01852v1.pdf use (Default):
    - factor=2.0 mode='FAN_IN' uniform=False

    To get http://arxiv.org/abs/1408.5093 use:
    - factor=1.0 mode='FAN_IN' uniform=True

    To get http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf use:
    - factor=1.0 mode='FAN_AVG' uniform=True.

    To get xavier_initializer use either:
    - factor=1.0 mode='FAN_AVG' uniform=True.
    - factor=1.0 mode='FAN_AVG' uniform=False.

    Arguments:
        factor: Float.  A multiplicative factor.
        mode: String.  'FAN_IN', 'FAN_OUT', 'FAN_AVG'.
        uniform: Whether to use uniform or normal distributed random
            initialization.
        seed: A Python integer. Used to create random seeds. See
            `set_random_seed` for behavior.
        dtype: The data type. Only floating point types are supported.

    Returns:
        An initializer that generates tensors with unit variance.

    Raises:
        ValueError: if `dtype` is not a floating point type.
        TypeError: if `mode` is not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG'].
    """
    if variance_scaling_initializer is None:
        raise NotImplementedError("'variance_scaling_initializer' not "
                                  "supported, please update TensorFlow.")
    return variance_scaling_initializer(factor=factor, mode=mode,
                                        uniform=uniform, seed=seed,
                                        dtype=dtype)
