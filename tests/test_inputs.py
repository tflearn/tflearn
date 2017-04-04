'''
    This file contains test cases for tflearn
'''

import tensorflow as tf
import tflearn
import unittest

class TestInputs(unittest.TestCase):
    '''
        This class contains test cases for serval input types
    '''
    INPUT_DATA_1 = [ [ 1 ], [ 2 ], [ 3 ], [ 4 ], [ 5 ] ]
    INPUT_DATA_2 = [ [ 6 ], [ 7 ], [ 8 ], [ 9 ], [ 10 ] ]
    TARGET = [ [ 14 ], [ 18 ], [ 22 ], [ 26 ], [ 30 ] ]   # (input1 + input2) * 2

    def test_list_inputs(self):
        """Test input a list
        """
        with tf.Graph().as_default():
            model, inputs, target = self.build_simple_model()
            model.fit([ inpData for _, _, inpData in inputs ], target, batch_size = 1)

    def test_dict_inputs(self):
        """Test input a dict with layer name
        """
        with tf.Graph().as_default():
            model, inputs, target = self.build_simple_model()
            model.fit({ name: inpData for name, _, inpData in inputs }, target, batch_size = 1)

    def test_dict_withtensor_inputs(self):
        """Test input a dict with placeholder
        """
        with tf.Graph().as_default():
            model, inputs, target = self.build_simple_model()
            model.fit({ placeholder: inpData for _, placeholder, inpData in inputs }, target, batch_size = 1)

    def build_simple_model(self):
        """Build a simple model for test
        Returns:
            DNN, [ (input layer name, input placeholder, input data) ], Target data
        """
        inputPlaceholder1, inputPlaceholder2 = \
            tf.placeholder(tf.float32, (1, 1), name = "input1"), tf.placeholder(tf.float32, (1, 1), name = "input2")
        input1 = tflearn.input_data(placeholder = inputPlaceholder1)
        input2 = tflearn.input_data(placeholder = inputPlaceholder2)
        network = tflearn.merge([ input1, input2 ], "sum")
        network = tflearn.reshape(network, (1, 1))
        network = tflearn.fully_connected(network, 1)
        network = tflearn.regression(network)
        return (
            tflearn.DNN(network),
            [ ("input1:0", inputPlaceholder1, self.INPUT_DATA_1), ("input2:0", inputPlaceholder2, self.INPUT_DATA_2) ],
            self.TARGET,
        )

if __name__ == "__main__":
    unittest.main()
