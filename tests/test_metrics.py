import tflearn
import unittest

import numpy as np
import tensorflow.compat.v1 as tf

class TestMetrics(unittest.TestCase):
    """
    Testing metric functions from tflearn/metrics
    """

    def test_binary_accuracy(self):
        with tf.Graph().as_default():
            input_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            y_true = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            ba = tflearn.metrics.accuracy()
            ba.build(input_data, y_true)
            acc_op = ba.tensor
    
            X = np.array([1,-1,1,1,-1,-1]).reshape([-1, 1])
            Y = np.array([1,0,1,0,0,1]).reshape([-1, 1])
            with tf.Session() as sess:
                binary_accuracy = sess.run(acc_op, feed_dict={input_data: X, y_true: Y})
                print ("binary_accuracy = %s" % binary_accuracy)
            self.assertEqual(acc_op.m_name, "binary_acc")
            self.assertLess(abs(binary_accuracy-4.0/6), 0.0001)

    def test_categorical_accuracy(self):
        with tf.Graph().as_default():
            input_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
            y_true = tf.placeholder(shape=[None, 2], dtype=tf.float32)
            ba = tflearn.metrics.accuracy()
            ba.build(input_data, y_true)
            acc_op = ba.tensor
    
            X = np.array([1,-1, -1, 1, 0.5, 0]).reshape([-1, 2])
            Y = np.array([1, 0,  0, 1, 0,   1]).reshape([-1, 2])
            with tf.Session() as sess:
                accuracy = sess.run(acc_op, feed_dict={input_data: X, y_true: Y})
                print ("categorical accuracy = %s" % accuracy)
            self.assertEqual(acc_op.m_name, "acc")
            self.assertLess(abs(accuracy - 2.0/3), 0.0001)

            X = np.array([1,-1, -1, 1, 0.5, 0]).reshape([-1, 2])
            Y = np.array([1, 0,  0, 1, 1,   0]).reshape([-1, 2])
            with tf.Session() as sess:
                accuracy = sess.run(acc_op, feed_dict={input_data: X, y_true: Y})
                print ("categorical accuracy = %s" % accuracy)
            self.assertEqual(accuracy, 1.0)

if __name__ == "__main__":
    unittest.main()
