import tensorflow as tf
import tflearn
import unittest
import os

class TestModelsLoadingScope(unittest.TestCase):
    """
    Testing loading scope, using DNN
    """

    def test_dnn_loading_scope(self):

        with tf.Graph().as_default():
            X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]
            Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]
            input = tflearn.input_data(shape=[None])
            linear = tflearn.single_unit(input)
            regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
                                            metric='R2', learning_rate=0.01)
            m = tflearn.DNN(regression)
            # Testing fit and predict
            m.fit(X, Y, n_epoch=1000, show_metric=True, snapshot_epoch=False)
            res = m.predict([3.2])[0]
            self.assertGreater(res, 1.3, "DNN test (linear regression) failed! with score: " + str(res) + " expected > 1.3")
            self.assertLess(res, 1.8, "DNN test (linear regression) failed! with score: " + str(res) + " expected < 1.8")

            # Testing save method
            m.save("test_dnn.tflearn")
            self.assertTrue(os.path.exists("test_dnn.tflearn.index"))

        # Testing loading, with change of variable scope (saved with no scope, now loading into scopeA)
        with tf.Graph().as_default():	# start with clear graph
            with tf.variable_scope("scopeA") as scope:
                input = tflearn.input_data(shape=[None])
                linear = tflearn.single_unit(input)
                regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
                                                metric='R2', learning_rate=0.01)
                m = tflearn.DNN(regression)
                def try_load():
                    m.load("test_dnn.tflearn")
                self.assertRaises(tf.errors.NotFoundError, try_load)	# fails, since names in file don't have "scopeA"

                m.load("test_dnn.tflearn", variable_name_map=("scopeA/", ""))	# succeeds, because variable names are rewritten
                res = m.predict([3.2])[0]
                self.assertGreater(res, 1.3, "DNN test (linear regression) failed after loading model! score: " + str(res) + " expected > 1.3")
                self.assertLess(res, 1.8, "DNN test (linear regression) failed after loading model! score: " + str(res) + " expected < 1.8")

if __name__ == "__main__":
    unittest.main()
