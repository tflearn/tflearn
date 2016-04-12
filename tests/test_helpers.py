import tensorflow as tf
import tflearn
import unittest
import os

class TestHelpers(unittest.TestCase):
    """
    Testing helper functions from tflearn/helpers
    """

    def test_variable(self):
        # Bulk Tests
        with tf.Graph().as_default():
            W = tflearn.variable(name='W1', shape=[784, 256],
                     initializer='uniform_scaling',
                     regularizer='L2')
            W = tflearn.variable(name='W2', shape=[784, 256],
                     initializer='uniform_scaling',
                     regularizer='L2')

    def test_regularizer(self):
        # Bulk Tests
        with tf.Graph().as_default():
            x = tf.placeholder("float", [None, 4])
            W = tf.Variable(tf.random_normal([4, 4]))
            x = tf.nn.tanh(tf.matmul(x, W))
            tflearn.add_weights_regularizer(W, 'L2', weight_decay=0.001)

    def test_summarizer(self):
        # Bulk Tests
        with tf.Graph().as_default():
            x = tf.placeholder("float", [None, 4])
            W = tf.Variable(tf.random_normal([4, 4]))
            x = tf.nn.tanh(tf.matmul(x, W))
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
            import tflearn.helpers.summarizer as s
            s.summarize_variables([W])
            s.summarize_activations(tf.get_collection(tf.GraphKeys.ACTIVATIONS))
            s.summarize(x, 'histogram', "test_summary")

if __name__ == "__main__":
    unittest.main()
