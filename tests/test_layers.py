import tensorflow.compat.v1 as tf
import tflearn
import unittest
import os

class TestLayers(unittest.TestCase):
    """
    Testing layers from tflearn/layers
    """

    def test_core_layers(self):

        X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
        Y_nand = [[1.], [1.], [1.], [0.]]
        Y_or = [[0.], [1.], [1.], [1.]]

        # Graph definition
        with tf.Graph().as_default():
            # Building a network with 2 optimizers
            g = tflearn.input_data(shape=[None, 2])

            # Nand operator definition
            g_nand = tflearn.fully_connected(g, 32, activation='linear')
            g_nand = tflearn.fully_connected(g_nand, 32, activation='linear')
            g_nand = tflearn.fully_connected(g_nand, 1, activation='sigmoid')
            g_nand = tflearn.regression(g_nand, optimizer='sgd',
                                        learning_rate=2.,
                                        loss='binary_crossentropy')
            # Or operator definition
            g_or = tflearn.fully_connected(g, 32, activation='linear')
            g_or = tflearn.fully_connected(g_or, 32, activation='linear')
            g_or = tflearn.fully_connected(g_or, 1, activation='sigmoid')
            g_or = tflearn.regression(g_or, optimizer='sgd',
                                      learning_rate=2.,
                                      loss='binary_crossentropy')
            # XOR merging Nand and Or operators
            g_xor = tflearn.merge([g_nand, g_or], mode='elemwise_mul')

            # Training
            m = tflearn.DNN(g_xor)
            m.fit(X, [Y_nand, Y_or], n_epoch=400, snapshot_epoch=False)

            # Testing
            self.assertLess(m.predict([[0., 0.]])[0][0], 0.01)
            self.assertGreater(m.predict([[0., 1.]])[0][0], 0.9)
            self.assertGreater(m.predict([[1., 0.]])[0][0], 0.9)
            self.assertLess(m.predict([[1., 1.]])[0][0], 0.01)

        # Bulk Tests
        with tf.Graph().as_default():
            net = tflearn.input_data(shape=[None, 2])
            net = tflearn.flatten(net)
            net = tflearn.reshape(net, new_shape=[-1])
            net = tflearn.activation(net, 'relu')
            net = tflearn.dropout(net, 0.5)
            net = tflearn.single_unit(net)

    def test_conv_layers(self):

        X = [[0., 0., 0., 0.], [1., 1., 1., 1.], [0., 0., 1., 0.], [1., 1., 1., 0.]]
        Y = [[1., 0.], [0., 1.], [1., 0.], [0., 1.]]

        with tf.Graph().as_default():
            g = tflearn.input_data(shape=[None, 4])
            g = tflearn.reshape(g, new_shape=[-1, 2, 2, 1])
            g = tflearn.conv_2d(g, 4, 2, activation='relu')
            g = tflearn.max_pool_2d(g, 2)
            g = tflearn.fully_connected(g, 2, activation='softmax')
            g = tflearn.regression(g, optimizer='sgd', learning_rate=1.)

            m = tflearn.DNN(g)
            m.fit(X, Y, n_epoch=100, snapshot_epoch=False)
            # TODO: Fix test
            #self.assertGreater(m.predict([[1., 0., 0., 0.]])[0][0], 0.5)

        # Bulk Tests
        with tf.Graph().as_default():
            g = tflearn.input_data(shape=[None, 4])
            g = tflearn.reshape(g, new_shape=[-1, 2, 2, 1])
            g = tflearn.conv_2d(g, 4, 2)
            g = tflearn.conv_2d(g, 4, 1)
            g = tflearn.conv_2d_transpose(g, 4, 2, [2, 2])
            g = tflearn.max_pool_2d(g, 2)

    def test_recurrent_layers(self):

        X = [[1, 3, 5, 7], [2, 4, 8, 10], [1, 5, 9, 11], [2, 6, 8, 0]]
        Y = [[0., 1.], [1., 0.], [0., 1.], [1., 0.]]

        with tf.Graph().as_default():
            g = tflearn.input_data(shape=[None, 4])
            g = tflearn.embedding(g, input_dim=12, output_dim=4)
            g = tflearn.lstm(g, 6)
            g = tflearn.fully_connected(g, 2, activation='softmax')
            g = tflearn.regression(g, optimizer='sgd', learning_rate=1.)

            m = tflearn.DNN(g)
            m.fit(X, Y, n_epoch=300, snapshot_epoch=False)
            self.assertGreater(m.predict([[5, 9, 11, 1]])[0][1], 0.9)

    def test_regression_placeholder(self):
        '''
        Check that regression does not duplicate placeholders
        '''

        with tf.Graph().as_default():

            g = tflearn.input_data(shape=[None, 2])
            g_nand = tflearn.fully_connected(g, 1, activation='linear')
            with tf.name_scope("Y"):
                Y_in = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="Y")
            tflearn.regression(g_nand, optimizer='sgd',
                               placeholder=Y_in,
                               learning_rate=2.,
                               loss='binary_crossentropy', 
                               op_name="regression1",
                               name="Y")
            # for this test, just use the same default trainable_vars
            # in practice, this should be different for the two regressions
            tflearn.regression(g_nand, optimizer='adam',
                               placeholder=Y_in,
                               learning_rate=2.,
                               loss='binary_crossentropy', 
                               op_name="regression2",
                               name="Y")

            self.assertEqual(len(tf.get_collection(tf.GraphKeys.TARGETS)), 1)

    def test_feed_dict_no_None(self):

        X = [[0., 0., 0., 0.], [1., 1., 1., 1.], [0., 0., 1., 0.], [1., 1., 1., 0.]]
        Y = [[1., 0.], [0., 1.], [1., 0.], [0., 1.]]

        with tf.Graph().as_default():
            g = tflearn.input_data(shape=[None, 4], name="X_in")
            g = tflearn.reshape(g, new_shape=[-1, 2, 2, 1])
            g = tflearn.conv_2d(g, 4, 2)
            g = tflearn.conv_2d(g, 4, 1)
            g = tflearn.max_pool_2d(g, 2)
            g = tflearn.fully_connected(g, 2, activation='softmax')
            g = tflearn.regression(g, optimizer='sgd', learning_rate=1.)

            m = tflearn.DNN(g)

            def do_fit():
                m.fit({"X_in": X, 'non_existent': X}, Y, n_epoch=30, snapshot_epoch=False)
            self.assertRaisesRegexp(Exception, "Feed dict asks for variable named 'non_existent' but no such variable is known to exist", do_fit)

if __name__ == "__main__":
    unittest.main()
