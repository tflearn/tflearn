"""
This example introduces the use of TFLearn functions to easily summarize
variables into tensorboard.

TFLearn can summarize:
- Loss / Accuracy: The model loss and accuracy over training steps.
- Activations: Histogram of operation output values.(Requires to add each
    activation to monitor into tf.Graphkeys.ACTIVATIONS collection).
- Gradients: Histogram of trainable variables gradient.
- Weights: Histogram of trainable variables weights.
- Weight Decay: Decay of trainable variables with regularizer. (Requires
    to add each decay into tf.Graphkeys.REGULARIZATION_LOSSES collection)
- Sparsity: Sparsity of trainable variables.

It is useful to also be able to periodically monitor various variables
during training, e.g. confusion matrix entries or AUC metrics. This
can be done using "validation_monitors", an argument to regression or
TrainOp; this argument takes a list of Tensor variables, and passes
them to the trainer, where they are evaluated each time a validation
step happens. The evaluation results are then summarized, and saved
for tensorboard visualization.

Summaries are monitored according to the following verbose levels:
- 0: Loss & Metric (Best speed).
- 1: Loss, Metric & Gradients.
- 2: Loss, Metric, Gradients & Weights.
- 3: Loss, Metric, Gradients, Weights, Activations & Sparsity (Best
     Visualization).

Note: If you are using TFLearn layers, summaries are automatically handled,
so you do not need to manually add them.

"""

import tensorflow as tf
import tflearn

# Loading MNIST dataset
import tflearn.datasets.mnist as mnist
trainX, trainY, testX, testY = mnist.load_data(one_hot=True)

# Define a dnn using Tensorflow
with tf.Graph().as_default():

    # Model variables
    X = tf.placeholder("float", [None, 784])
    Y = tf.placeholder("float", [None, 10])

    # Multilayer perceptron, with `tanh` functions activation monitor
    def dnn(x):
        with tf.name_scope('Layer1'):
            W1 = tf.Variable(tf.random_normal([784, 256]), name="W1")
            b1 = tf.Variable(tf.random_normal([256]), name="b1")
            x = tf.nn.tanh(tf.add(tf.matmul(x, W1), b1))
            # Add this `tanh` op to activations collection or monitoring
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            # Add weights regularizer (Regul. summary automatically added)
            tflearn.add_weights_regularizer(W1, 'L2', weight_decay=0.001)

        with tf.name_scope('Layer2'):
            W2 = tf.Variable(tf.random_normal([256, 256]), name="W2")
            b2 = tf.Variable(tf.random_normal([256]), name="b2")
            x = tf.nn.tanh(tf.add(tf.matmul(x, W2), b2))
            # Add this `tanh` op to activations collection or monitoring
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            # Add weights regularizer (Regul. summary automatically added)
            tflearn.add_weights_regularizer(W2, 'L2', weight_decay=0.001)

        with tf.name_scope('Layer3'):
            W3 = tf.Variable(tf.random_normal([256, 10]), name="W3")
            b3 = tf.Variable(tf.random_normal([10]), name="b3")
            x = tf.add(tf.matmul(x, W3), b3)

        return x

    net = dnn(X)
        
    with tf.name_scope('Summaries'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net,labels=Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1)), tf.float32),
            name="acc")

    # construct two varaibles to add as additional "valiation monitors"
    # these varaibles are evaluated each time validation happens (eg at a snapshot)
    # and the results are summarized and output to the tensorboard events file,
    # together with the accuracy and loss plots.
    #
    # Here, we generate a dummy variable given by the sum over the current
    # network tensor, and a constant variable.  In practice, the validation
    # monitor may present useful information, like confusion matrix
    # entries, or an AUC metric.
    with tf.name_scope('CustomMonitor'):
        test_var = tf.reduce_sum(tf.cast(net, tf.float32), name="test_var")
        test_const = tf.constant(32.0, name="custom_constant")
        # Define a train op
    trainop = tflearn.TrainOp(loss=loss, optimizer=optimizer,
                            validation_monitors=[test_var, test_const],
                            metric=accuracy, batch_size=128)

    # Tensorboard logs stored in /tmp/tflearn_logs/. Using verbose level 2.
    trainer = tflearn.Trainer(train_ops=trainop,
                              tensorboard_dir='/tmp/tflearn_logs/',
                              tensorboard_verbose=2)
    # Training for 10 epochs.
    trainer.fit({X: trainX, Y: trainY}, val_feed_dicts={X: testX, Y: testY},
                n_epoch=10, show_metric=True, run_id='Summaries_example')

    # Run the following command to start tensorboard:
    # >> tensorboard /tmp/tflearn_logs/
    # Navigate with your web browser to http://0.0.0.0:6006/
