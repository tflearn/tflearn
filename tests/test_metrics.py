import tflearn
import unittest

import numpy as np
import tensorflow as tf

class TestMetrics(unittest.TestCase):
    """
    Testing metric functions from tflearn/metrics
    """

    def test_accuracy(self):
        input_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        y_true = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        ba = tflearn.metrics.binary_accuracy()
        ba.build(input_data, y_true)
        acc_op = ba.tensor

        X = np.array([1,0,1,1,0,0]).reshape([-1, 1])
        Y = np.array([1,0,1,0,0,1]).reshape([-1, 1])
        with tf.Session() as sess:
            binary_accuracy = sess.run(acc_op, feed_dict={input_data: X, y_true: Y})
            print ("binary_accuracy = %s" % binary_accuracy)
        self.assertTrue(abs(binary_accuracy-4.0/6)< 0.0001)

if __name__ == "__main__":
    unittest.main()
