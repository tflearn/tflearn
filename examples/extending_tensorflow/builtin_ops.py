from __future__ import division, print_function, absolute_import

"""
This tutorial will introduce how to combine TFLearn built-in ops with any
Tensorflow graph.
"""

import tensorflow as tf
import tflearn

# ----------------------------------
# Using TFLearn built-in ops example
# ----------------------------------

# Using MNIST Dataset
import tflearn.datasets.mnist as mnist
trainX, trainY, testX, testY = mnist.load_data(one_hot=True)

# User defined placeholders
with tf.Graph().as_default():

    # Model variables
    X = tf.placeholder("float", [None, 784])
    Y = tf.placeholder("float", [None, 10])

    W1 = tf.Variable(tf.random_normal([784, 256]))
    W2 = tf.Variable(tf.random_normal([256, 256]))
    W3 = tf.Variable(tf.random_normal([256, 10]))
    b1 = tf.Variable(tf.random_normal([256]))
    b2 = tf.Variable(tf.random_normal([256]))
    b3 = tf.Variable(tf.random_normal([10]))

    # Multilayer perceptron
    def dnn(x):
        # Using TFLearn PReLU activations ops
        x = tflearn.prelu(tf.add(tf.matmul(x, W1), b1))
        tflearn.summaries.monitor_activation(x) # Monitor activation
        x = tflearn.prelu(tf.add(tf.matmul(x, W2), b2))
        tflearn.summaries.monitor_activation(x) # Monitor activation
        x = tf.nn.softmax(tf.add(tf.matmul(x, W3), b3))
        return x

    net = dnn(X)

    # Using objective ops from TFLearn to compute crossentropy
    loss = tflearn.categorical_crossentropy(net, Y)

    # Using metric ops from TFLearn to compute accuracy
    acc = tflearn.metrics.accuracy_op(net, Y)

    # Using TFLearn SGD Optimizer class
    optimizer = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=200)
    # Because of lr decay, it is required to first build the Optimizer with
    # the step tensor that will monitor training step.
    # (Note: When using TFLearn estimators wrapper, build is self managed,
    # so only using above `Optimizer` class as `DNN` optimizer arg is enough).
    step = tflearn.variable("step", initializer='zeros', shape=[])
    optimizer.build(step_tensor=step)
    optim_tensor = optimizer.get_tensor()

    # Using TFLearn Trainer
    # Define a training op (op for backprop, only need 1 in this model)
    trainop = tflearn.TrainOp(loss=loss, optimizer=optim_tensor,
                              metric=acc, batch_size=128,
                              step_tensor=step)

    # Create Trainer, providing all training ops. Tensorboard logs stored
    # in /tmp/tflearn_logs/. It is possible to change verbose level for more
    # details logs about gradients, variables etc...
    trainer = tflearn.Trainer(train_ops=trainop, tensorboard_verbose=0)
    # Training for 10 epochs.
    trainer.fit({X: trainX, Y: trainY}, val_feed_dicts={X: testX, Y: testY},
                n_epoch=10, show_metric=True)
