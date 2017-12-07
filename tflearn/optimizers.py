from __future__ import division, print_function, absolute_import

import tensorflow as tf
from .utils import get_from_module


def get(identifier):
    return get_from_module(identifier, globals(), 'optimizer')


class Optimizer(object):
    """ Base Optimizer class.

    A basic class to create optimizers to be used with TFLearn estimators.
    First, The Optimizer class is initialized with given parameters,
    but no Tensor is created. In a second step, invoking `get_tensor` method
    will actually build the Tensorflow `Optimizer` Tensor, and return it.

    This way, a user can easily specifies an optimizer with non default
    parameters and learning rate decay, while TFLearn estimators will
    build the optimizer and a step tensor by itself.

    Arguments:
        learning_rate: `float`. Learning rate.
        use_locking: `bool`. If True use locks for update operation.
        name: `str`. The optimizer name.

    Attributes:
        tensor: `Optimizer`. The optimizer tensor.
        has_decay: `bool`. True if optimizer has a learning rate decay.

    """

    def __init__(self, learning_rate, use_locking, name):
        self.learning_rate = learning_rate
        self.use_locking = use_locking
        self.name = name
        self.tensor = None
        self.has_decay = False
        self.built = False

    def build(self, step_tensor=None):
        """ build optimizer tensor.

        This method creates the optimizer with specified parameters. It must
        be implemented for every `Optimizer`.

        Arguments:
            step_tensor: `tf.Tensor`. A variable holding the training step.
                Only necessary when optimizer has a learning rate decay.

        """
        raise NotImplementedError

    def get_tensor(self):
        """ get_tensor.

        A method to retrieve the optimizer tensor.

        Returns:
            The `Optimizer`.

        """
        if not self.built:
            self.build()
        return self.tensor

    def __call__(self):
        """ __call__

        A shortcut for `get_tensor`. Retrieve the optimizer tensor.

        Returns:
            The `Optimizer`.

        """
        return self.get_tensor()


class SGD(Optimizer):
    """ Stochastic Gradient Descent.

    SGD Optimizer accepts learning rate decay. When training a model,
    it is often recommended to lower the learning rate as the training
    progresses. The function returns the decayed learning rate.  It is
    computed as:

    ```python
    decayed_learning_rate = learning_rate *
                          decay_rate ^ (global_step / decay_steps)
    ```

    Examples:
        ```python
        # With TFLearn estimators.
        sgd = SGD(learning_rate=0.01, lr_decay=0.96, decay_step=100)
        regression = regression(net, optimizer=sgd)

        # Without TFLearn estimators (returns tf.Optimizer).
        sgd = SGD(learning_rate=0.01).get_tensor()
        ```

    Arguments:
        learning_rate: `float`. Learning rate.
        use_locking: `bool`. If True use locks for update operation.
        lr_decay: `float`. The learning rate decay to apply.
        decay_step: `int`. Apply decay every provided steps.
        staircase: `bool`. It `True` decay learning rate at discrete intervals.
        use_locking: `bool`. If True use locks for update operation.
        name: `str`. Optional name prefix for the operations created when
            applying gradients. Defaults to "GradientDescent".

    """

    def __init__(self, learning_rate=0.001, lr_decay=0., decay_step=100,
                 staircase=False, use_locking=False, name="SGD"):
        super(SGD, self).__init__(learning_rate, use_locking, name)
        self.lr_decay = lr_decay
        if self.lr_decay > 0.:
            self.has_decay = True
        self.decay_step = decay_step
        self.staircase = staircase

    def build(self, step_tensor=None):
        self.built = True
        if self.has_decay:
            if not step_tensor:
                raise Exception("Learning rate decay but no step_tensor "
                                "provided.")
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate, step_tensor,
                self.decay_step, self.lr_decay,
                staircase=self.staircase)
            tf.add_to_collection(tf.GraphKeys.LR_VARIABLES, self.learning_rate)
        self.tensor = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate,
            use_locking=self.use_locking,
            name=self.name)

# Shortcut
sgd = SGD


class RMSProp(Optimizer):
    """ RMSprop.

    Maintain a moving (discounted) average of the square of gradients.
    Divide gradient by the root of this average.

    Examples:
        ```python
        # With TFLearn estimators.
        rmsprop = RMSProp(learning_rate=0.1, decay=0.999)
        regression = regression(net, optimizer=rmsprop)

        # Without TFLearn estimators (returns tf.Optimizer).
        rmsprop = RMSProp(learning_rate=0.01, decay=0.999).get_tensor()
        # or
        rmsprop = RMSProp(learning_rate=0.01, decay=0.999)()

        ```

    Arguments:
        learning_rate: `float`. Learning rate.
        decay: `float`. Discounting factor for the history/coming gradient.
        momentum: `float`. Momentum.
        epsilon: `float`. Small value to avoid zero denominator.
        use_locking: `bool`. If True use locks for update operation.
        name: `str`. Optional name prefix for the operations created when
            applying gradients. Defaults to "RMSProp".

    """

    def __init__(self, learning_rate=0.001, decay=0.9, momentum=0.0,
                 epsilon=1e-10, use_locking=False, name="RMSProp"):
        super(RMSProp, self).__init__(learning_rate, use_locking, name)
        self.decay = decay
        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, step_tensor=None):
        self.built = True
        self.tensor = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate, decay=self.decay,
            momentum=self.momentum, epsilon=self.epsilon,
            use_locking=self.use_locking, name=self.name)

rmsprop = RMSProp


class Adam(Optimizer):
    """ Adam.

    The default value of 1e-8 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1.

    Examples:
        ```python
        # With TFLearn estimators
        adam = Adam(learning_rate=0.001, beta1=0.99)
        regression = regression(net, optimizer=adam)

        # Without TFLearn estimators (returns tf.Optimizer)
        adam = Adam(learning_rate=0.01).get_tensor()

        ```

    Arguments:
        learning_rate: `float`. Learning rate.
        beta1: `float`. The exponential decay rate for the 1st moment
            estimates.
        beta2: `float`. The exponential decay rate for the 2nd moment
            estimates.
        epsilon: `float`. A small constant for numerical stability.
        use_locking: `bool`. If True use locks for update operation.
        name: `str`. Optional name prefix for the operations created when
            applying gradients. Defaults to "Adam".

    References:
        Adam: A Method for Stochastic Optimization. Diederik Kingma,
        Jimmy Ba. ICLR 2015.

    Links:
        [Paper](http://arxiv.org/pdf/1412.6980v8.pdf)

    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, use_locking=False, name="Adam"):
        super(Adam, self).__init__(learning_rate, use_locking, name)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def build(self, step_tensor=None):
        self.built = True
        self.tensor = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, beta1=self.beta1,
            beta2=self.beta2, epsilon=self.epsilon,
            use_locking=self.use_locking, name=self.name)

adam = Adam


class Momentum(Optimizer):
    """ Momentum.

    Momentum Optimizer accepts learning rate decay. When training a model,
    it is often recommended to lower the learning rate as the training
    progresses. The function returns the decayed learning rate.  It is
    computed as:

    ```python
    decayed_learning_rate = learning_rate *
                          decay_rate ^ (global_step / decay_steps)
    ```

    Examples:
        ```python
        # With TFLearn estimators
        momentum = Momentum(learning_rate=0.01, lr_decay=0.96, decay_step=100)
        regression = regression(net, optimizer=momentum)

        # Without TFLearn estimators (returns tf.Optimizer)
        mm = Momentum(learning_rate=0.01, lr_decay=0.96).get_tensor()
        ```

    Arguments:
        learning_rate: `float`. Learning rate.
        momentum: `float`. Momentum.
        lr_decay: `float`. The learning rate decay to apply.
        decay_step: `int`. Apply decay every provided steps.
        staircase: `bool`. It `True` decay learning rate at discrete intervals.
        use_locking: `bool`. If True use locks for update operation.
        name: `str`. Optional name prefix for the operations created when
            applying gradients. Defaults to "Momentum".

    """

    def __init__(self, learning_rate=0.001, momentum=0.9, lr_decay=0.,
                 decay_step=100, staircase=False, use_locking=False,
                 name="Momentum"):
        super(Momentum, self).__init__(learning_rate, use_locking, name)
        self.momentum = momentum
        self.lr_decay = lr_decay
        if self.lr_decay > 0.:
            self.has_decay = True
        self.decay_step = decay_step
        self.staircase = staircase

    def build(self, step_tensor=None):
        self.built = True
        if self.has_decay:
            if not step_tensor:
                raise Exception("Learning rate decay but no step_tensor "
                                "provided.")
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate, step_tensor,
                self.decay_step, self.lr_decay,
                staircase=self.staircase)
            tf.add_to_collection(tf.GraphKeys.LR_VARIABLES, self.learning_rate)
        self.tensor = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            use_locking=self.use_locking,
            name=self.name)

momentum = Momentum


class AdaGrad(Optimizer):
    """ AdaGrad.

    Examples:
        ```python
        # With TFLearn estimators
        adagrad = AdaGrad(learning_rate=0.01, initial_accumulator_value=0.01)
        regression = regression(net, optimizer=adagrad)

        # Without TFLearn estimators (returns tf.Optimizer)
        adagrad = AdaGrad(learning_rate=0.01).get_tensor()
        ```

    Arguments:
        learning_rate: `float`. Learning rate.
        initial_accumulator_value: `float`. Starting value for the
            accumulators, must be positive
        use_locking: `bool`. If True use locks for update operation.
        name: `str`. Optional name prefix for the operations created when
            applying gradients. Defaults to "AdaGrad".

    References:
        Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization. J. Duchi, E. Hazan & Y. Singer. Journal of Machine
        Learning Research 12 (2011) 2121-2159.

    Links:
        [Paper](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

    """

    def __init__(self, learning_rate=0.001, initial_accumulator_value=0.1,
                 use_locking=False, name="AdaGrad"):
        super(AdaGrad, self).__init__(learning_rate, use_locking, name)
        self.initial_accumulator_value = initial_accumulator_value

    def build(self, step_tensor=None):
        self.built = True
        self.tensor = tf.train.AdagradOptimizer(
            self.learning_rate,
            initial_accumulator_value=self.initial_accumulator_value,
            use_locking=self.use_locking, name=self.name)

adagrad = AdaGrad


class Ftrl(Optimizer):
    """ Ftrl Proximal.

    The Ftrl-proximal algorithm, abbreviated for Follow-the-regularized-leader,
    is described in the paper below.

    It can give a good performance vs. sparsity tradeoff.

    Ftrl-proximal uses its own global base learning rate and can behave like
    Adagrad with `learning_rate_power=-0.5`, or like gradient descent with
    `learning_rate_power=0.0`.

    Examples:
        ```python
        # With TFLearn estimators.
        ftrl = Ftrl(learning_rate=0.01, learning_rate_power=-0.1)
        regression = regression(net, optimizer=ftrl)

        # Without TFLearn estimators (returns tf.Optimizer).
        ftrl = Ftrl(learning_rate=0.01).get_tensor()
        ```

    Arguments:
        learning_rate: `float`. Learning rate.
        learning_rate_power: `float`. Must be less or equal to zero.
        initial_accumulator_value: `float`. The starting value for accumulators.
            Only positive values are allowed.
        l1_regularization_strength: `float`. Must be less or equal to zero.
        l2_regularization_strength: `float`. Must be less or equal to zero.
        use_locking: `bool`. If True use locks for update operation.
        name: `str`. Optional name prefix for the operations created when
            applying gradients. Defaults to "Ftrl".

    Links:
        [Ad Click Prediction: a View from the Trenches](https://www.eecs.tufts.
        edu/~dsculley/papers/ad-click-prediction.pdf)

    """

    def __init__(self, learning_rate=3.0, learning_rate_power=-0.5,
                 initial_accumulator_value=0.1, l1_regularization_strength=0.0,
                 l2_regularization_strength=0.0, use_locking=False,
                 name="Ftrl"):
        super(Ftrl, self).__init__(learning_rate, use_locking, name)
        self.learning_rate_power = learning_rate_power
        self.initial_accumulator_value = initial_accumulator_value
        self.l1_regularization_strength = l1_regularization_strength
        self.l2_regularization_strength = l2_regularization_strength

    def build(self, step_tensor=None):
        self.built = True
        with tf.device('/cpu:0'):
            self.tensor = tf.train.FtrlOptimizer(
                self.learning_rate,
                learning_rate_power=self.learning_rate_power,
                initial_accumulator_value=self.initial_accumulator_value,
                l1_regularization_strength=self.l1_regularization_strength,
                l2_regularization_strength=self.l2_regularization_strength,
                use_locking=self.use_locking, name=self.name)

ftrl = Ftrl


class AdaDelta(Optimizer):
    """ AdaDelta.

    Construct a new Adadelta optimizer.

    Arguments:
        learning_rate: A `Tensor` or a floating point value. The learning rate.
        rho: A `Tensor` or a floating point value. The decay rate.
        epsilon: A `Tensor` or a floating point value.  A constant epsilon used
            to better conditioning the grad update.
        use_locking: If `True` use locks for update operations.
        name: Optional name prefix for the operations created when applying
            gradients.  Defaults to "Adadelta".

    References:
        ADADELTA: An Adaptive Learning Rate Method, Matthew D. Zeiler, 2012.

    Links:
        [http://arxiv.org/abs/1212.5701](http://arxiv.org/abs/1212.5701)

    """

    def __init__(self, learning_rate=0.001, rho=0.1, epsilon=1e-08,
                 use_locking=False, name="AdaDelta"):
        super(AdaDelta, self).__init__(learning_rate, use_locking, name)
        self.rho = rho
        self.epsilon = epsilon

    def build(self, step_tensor=None):
        self.built = True
        self.tensor = tf.train.AdadeltaOptimizer(
            self.learning_rate,
            rho=self.rho, epsilon=self.epsilon,
            use_locking=self.use_locking, name=self.name)

adadelta = AdaDelta


class ProximalAdaGrad(Optimizer):
    """ ProximalAdaGrad.

    Examples:
        ```python
        # With TFLearn estimators
        proxi_adagrad = ProximalAdaGrad(learning_rate=0.01,
                                        l2_regularization_strength=0.01,
                                        initial_accumulator_value=0.01)
        regression = regression(net, optimizer=proxi_adagrad)

        # Without TFLearn estimators (returns tf.Optimizer)
        adagrad = ProximalAdaGrad(learning_rate=0.01).get_tensor()
        ```

    Arguments:
        learning_rate: `float`. Learning rate.
        initial_accumulator_value: `float`. Starting value for the
            accumulators, must be positive
        use_locking: `bool`. If True use locks for update operation.
        name: `str`. Optional name prefix for the operations created when
            applying gradients. Defaults to "AdaGrad".

    References:
        Efficient Learning using Forward-Backward Splitting. J. Duchi, Yoram
        Singer, 2009.

    Links:
        [Paper](http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf)

    """

    def __init__(self, learning_rate=0.001, initial_accumulator_value=0.1,
                 use_locking=False, name="AdaGrad"):
        super(ProximalAdaGrad, self).__init__(learning_rate, use_locking, name)
        self.initial_accumulator_value = initial_accumulator_value

    def build(self, step_tensor=None):
        self.built = True
        self.tensor = tf.train.AdagradOptimizer(
            self.learning_rate,
            initial_accumulator_value=self.initial_accumulator_value,
            use_locking=self.use_locking, name=self.name)

proximaladagrad = ProximalAdaGrad


class Nesterov(Optimizer):
    """ Nesterov.

    The main difference between classical momentum and nesterov is:
    In classical momentum you first correct your velocity and 
    then make a big step according to that velocity (and then repeat), 
    but in Nesterov momentum you first making a step into velocity 
    direction and then make a correction to a velocity vector based on
    new location (then repeat).
    See [Sutskever et. al., 2013](
            http://jmlr.org/proceedings/papers/v28/sutskever13.pdf)

    Examples:
        ```python
        # With TFLearn estimators
        nesterov = Nesterov(learning_rate=0.01, lr_decay=0.96, decay_step=100)
        regression = regression(net, optimizer=nesterov)

        # Without TFLearn estimators (returns tf.Optimizer)
        mm = Neserov(learning_rate=0.01, lr_decay=0.96).get_tensor()
        ```

    Arguments:
        learning_rate: `float`. Learning rate.
        momentum: `float`. Momentum.
        lr_decay: `float`. The learning rate decay to apply.
        decay_step: `int`. Apply decay every provided steps.
        staircase: `bool`. It `True` decay learning rate at discrete intervals.
        use_locking: `bool`. If True use locks for update operation.
        name: `str`. Optional name prefix for the operations created when
            applying gradients. Defaults to "Momentum".

    """

    def __init__(self, learning_rate=0.001, momentum=0.9, lr_decay=0.,
                 decay_step=100, staircase=False, use_locking=False,
                 name="Nesterov"):
        super(Nesterov, self).__init__(learning_rate, use_locking, name)
        self.momentum = momentum
        self.lr_decay = lr_decay
        if self.lr_decay > 0.:
            self.has_decay = True
        self.decay_step = decay_step
        self.staircase = staircase

    def build(self, step_tensor=None):
        self.built = True
        if self.has_decay:
            if not step_tensor:
                raise Exception("Learning rate decay but no step_tensor "
                                "provided.")
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate, step_tensor,
                self.decay_step, self.lr_decay,
                staircase=self.staircase)
            tf.add_to_collection(tf.GraphKeys.LR_VARIABLES, self.learning_rate)
        self.tensor = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            use_locking=self.use_locking,
            name=self.name,use_nesterov=True)

nesterov = Nesterov
