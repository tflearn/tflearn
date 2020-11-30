import tflearn
import unittest

import numpy as np
import tensorflow.compat.v1 as tf


class TestObjectives(unittest.TestCase):
    """
    Testing objective functions from tflearn/objectives
    """

    def test_weak_cross_entropy_2d(self):
        """
        Test tflearn.objectives.weak_cross_entropy_2d
        """
        num_classes = 2
        batch_size = 3
        height, width = 5, 5
        shape = (batch_size, height, width, num_classes)
        y_pred = np.random.random(shape).astype(np.float32)
        target = np.random.randint(0, num_classes, np.prod(shape[:-1]))
        # convert to one-hot encoding
        y_true = np.eye(num_classes)[target].reshape(shape)

        with tf.Graph().as_default():
            y_pred = tf.convert_to_tensor(y_pred)
            y_true = tf.convert_to_tensor(y_true)

            loss = tflearn.objectives.weak_cross_entropy_2d(y_pred, y_true)

            with tf.Session() as sess:
                res = sess.run(loss)

        self.assertGreater(res, 0.)
        self.assertLess(res, 1.)

if __name__ == "__main__":
    unittest.main()
