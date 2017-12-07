'''
    This file contains test cases for tflearn
'''

import tensorflow as tf
import tflearn
import unittest

class TestActivations(unittest.TestCase):
    '''
        This class contains test cases for the functions in tflearn/activations.py
    '''
    PLACES = 4 # Number of places to match when testing floating point values

    def test_linear(self):
        f = tflearn.linear

        # Case 1
        x = tf.placeholder(tf.float32, shape=())
        self.assertEqual(f(x), x)

        # Case 2
        x = tf.placeholder(tf.int64, shape=())
        self.assertEqual(f(x), x)

    def test_tanh(self):
        f = tflearn.tanh
        x = tf.placeholder(tf.float32, shape=())
        
        with tf.Session() as sess:
            # Case 1
            self.assertEqual(sess.run(f(x), feed_dict={x:0}), 0)

            # Case 2
            self.assertAlmostEqual(sess.run(f(x), feed_dict={x:0.5}),
                0.4621, places=TestActivations.PLACES)

            # Case 3
            self.assertAlmostEqual(sess.run(f(x), feed_dict={x:-0.25}),
                -0.2449, places=TestActivations.PLACES)

    def test_leaky_relu(self):
        f = lambda x: tflearn.leaky_relu(x, alpha=0.2)
        x = tf.placeholder(tf.float32, shape=())

        with tf.Session() as sess:
            # Case 1
            self.assertEqual(sess.run(f(x), feed_dict={x:0}), 0)

            # Case 2
            self.assertAlmostEqual(sess.run(f(x), feed_dict={x:1}),
                1, places=TestActivations.PLACES)

            # Case 3
            self.assertAlmostEqual(sess.run(f(x), feed_dict={x:-1}),
                -0.2, places=TestActivations.PLACES)

            # Case 4
            self.assertAlmostEqual(sess.run(f(x), feed_dict={x:-5}),
                -1, places=TestActivations.PLACES)

    def test_apply_activation(self):
        lrelu_02 = lambda x: tflearn.leaky_relu(x, alpha=0.2)
        x = tf.constant(-0.25, tf.float32)

        with tf.Session() as sess:
            # Case 1: 'linear'
            self.assertEqual(
                sess.run(tflearn.activation(x, 'linear')),
                -0.25)

            # Case 2: 'relu'
            self.assertEqual(
                sess.run(tflearn.activation(x, 'relu')),
                0)

            # Case 3: 'leaky_relu'
            self.assertAlmostEqual(
                sess.run(tflearn.activation(x, 'leaky_relu')),
                -0.025, places=TestActivations.PLACES)

            # Case 4: 'tanh'
            self.assertAlmostEqual(
                sess.run(tflearn.activation(x, 'tanh')),
                -0.2449, places=TestActivations.PLACES)

            # Case 5: lrelu_02 (callable)
            self.assertAlmostEqual(
                sess.run(tflearn.activation(x, lrelu_02)),
                -0.05, places=TestActivations.PLACES)

if __name__ == "__main__":
    unittest.main()