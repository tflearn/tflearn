from __future__ import division, print_function, absolute_import

import tensorflow.compat.v1 as tf
import tflearn
import unittest
import os

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

class TestValidationMonitors(unittest.TestCase):
    """
    Testing Validation Monitors
    """

    def test_vm1(self):

        # Temp skip test
        return

        with tf.Graph().as_default():
            # Data loading and preprocessing
            import tflearn.datasets.mnist as mnist
            X, Y, testX, testY = mnist.load_data(one_hot=True)
            X = X.reshape([-1, 28, 28, 1])
            testX = testX.reshape([-1, 28, 28, 1])
            X = X[:10, :, :, :]
            Y = Y[:10, :]
            
            # Building convolutional network
            network = input_data(shape=[None, 28, 28, 1], name='input')
            network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
            network = max_pool_2d(network, 2)
            network = local_response_normalization(network)
            network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
            network = max_pool_2d(network, 2)
            network = local_response_normalization(network)
            network = fully_connected(network, 128, activation='tanh')
            network = dropout(network, 0.8)
            network = fully_connected(network, 256, activation='tanh')
            network = dropout(network, 0.8)
    
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
                test_var = tf.reduce_sum(tf.cast(network, tf.float32), name="test_var")
                test_const = tf.constant(32.0, name="custom_constant")
    
            print ("network=%s, test_var=%s" % (network, test_var))
            network = fully_connected(network, 10, activation='softmax')
            network = regression(network, optimizer='adam', learning_rate=0.01,
                                 loss='categorical_crossentropy', name='target', validation_monitors=[test_var, test_const])
            
            # Training
            model = tflearn.DNN(network, tensorboard_verbose=3)
            model.fit({'input': X}, {'target': Y}, n_epoch=1,
                       validation_set=({'input': testX}, {'target': testY}),
                       snapshot_step=10, show_metric=True, run_id='convnet_mnist')
            
            # check for validation monitor variables
            ats = tf.get_collection("Adam_testing_summaries")
            print ("ats=%s" % ats)
            self.assertTrue(len(ats)==4)	# should be four variables being summarized: [loss, test_var, test_const, accuracy]
            
            session = model.session
            print ("session=%s" % session)
            trainer = model.trainer
            print ("train_ops = %s" % trainer.train_ops)
            top = trainer.train_ops[0]
            vmtset = top.validation_monitors_T
            print ("validation_monitors_T = %s" % vmtset)
            with model.session.as_default():
                ats_var_val = tflearn.variables.get_value(vmtset[0])
                ats_const_val = tflearn.variables.get_value(vmtset[1])
            print ("summary values: var=%s, const=%s" % (ats_var_val, ats_const_val))
            self.assertTrue(ats_const_val==32)	# test to make sure the constant made it through
    
            # TBD: parse the recorded tensorboard events and ensure the validation monitor variables show up there
    
class TestValidationBatch(unittest.TestCase):
    """
    Testing Validation Batch size specification
    """

    def test_vbs1(self):

        # Temp skip test
        return

        with tf.Graph().as_default():
            # Data loading and preprocessing
            import tflearn.datasets.mnist as mnist
            X, Y, testX, testY = mnist.load_data(one_hot=True)
            X = X.reshape([-1, 28, 28, 1])
            testX = testX.reshape([-1, 28, 28, 1])
            X = X[:20, :, :, :]
            Y = Y[:20, :]
            testX = testX[:10, :, :, :]
            testY = testY[:10, :]
            
            # Building convolutional network
            network = input_data(shape=[None, 28, 28, 1], name='input')
            network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
            network = max_pool_2d(network, 2)
            network = local_response_normalization(network)
            network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
            network = max_pool_2d(network, 2)
            network = local_response_normalization(network)
            network = fully_connected(network, 128, activation='tanh')
            network = dropout(network, 0.8)
            network = fully_connected(network, 256, activation='tanh')
            network = dropout(network, 0.8)
            network = fully_connected(network, 10, activation='softmax')
            network = regression(network, optimizer='adam', learning_rate=0.01,
                                 loss='categorical_crossentropy', name='target')
            
            # Training
            model = tflearn.DNN(network, tensorboard_verbose=3)
            model.fit({'input': X}, {'target': Y}, n_epoch=1,
                      batch_size=10,
                      validation_set=({'input': testX}, {'target': testY}),
                      validation_batch_size=5,
                      snapshot_step=10, show_metric=True, run_id='convnet_mnist_vbs')
    
            self.assertEqual(model.train_ops[0].validation_batch_size, 5)
            self.assertEqual(model.train_ops[0].batch_size, 10)
    
if __name__ == "__main__":
    unittest.main()
