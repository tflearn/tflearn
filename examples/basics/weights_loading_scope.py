'''
Demonstrate that weights saved with models in one scope, can be loaded 
into models being used in a different scope.

This allows multiple models to be run, and combined models to load
weights from separately trained models.
'''

from __future__ import division, print_function, absolute_import

import re
import tflearn
import tensorflow as tf
import tflearn.datasets.mnist as mnist

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

#-----------------------------------------------------------------------------

class Model1(object):
    '''
    convnet MNIST
    '''
    def __init__(self):
        network = tflearn.input_data(shape=[None, 784], name="input")
        network = self.make_core_network(network)
        network = regression(network, optimizer='adam', learning_rate=0.01,
                             loss='categorical_crossentropy', name='target')
        
        model = tflearn.DNN(network, tensorboard_verbose=0)
        self.model = model

    @staticmethod
    def make_core_network(network):
        network = tflearn.reshape(network, [-1, 28, 28, 1], name="reshape")
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
        return network

    def train(self, X, Y, testX, testY, n_epoch=1, snapshot_step=1000):
        # Training
        self.model.fit({'input': X}, {'target': Y}, n_epoch=n_epoch,
                       validation_set=({'input': testX}, {'target': testY}),
                       snapshot_step=snapshot_step,
                       show_metric=True, run_id='convnet_mnist')
        
class Model2(object):
    '''
    dnn MNIST
    '''
    def __init__(self):
        # Building deep neural network
        network = tflearn.input_data(shape=[None, 784], name="input")
        network = self.make_core_network(network)

        # Regression using SGD with learning rate decay and Top-3 accuracy
        sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
        top_k = tflearn.metrics.Top_k(3)

        network = tflearn.regression(network, optimizer=sgd, metric=top_k,
                                 loss='categorical_crossentropy', name="target")
        model = tflearn.DNN(network, tensorboard_verbose=0)
        self.model = model

    @staticmethod
    def make_core_network(network):
        dense1 = tflearn.fully_connected(network, 64, activation='tanh',
                                         regularizer='L2', weight_decay=0.001, name="dense1")
        dropout1 = tflearn.dropout(dense1, 0.8)
        dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh',
                                         regularizer='L2', weight_decay=0.001, name="dense2")
        dropout2 = tflearn.dropout(dense2, 0.8)
        softmax = tflearn.fully_connected(dropout2, 10, activation='softmax', name="softmax")
        return softmax

    def train(self, X, Y, testX, testY, n_epoch=1, snapshot_step=1000):
        # Training
        self.model.fit(X, Y, n_epoch=n_epoch, validation_set=(testX, testY),
                       snapshot_step=snapshot_step,
                       show_metric=True, run_id="dense_model")
        
class Model12(object):
    '''
    Combination of two networks
    '''
    def __init__(self):
        inputs = tflearn.input_data(shape=[None, 784], name="input")

        with tf.variable_scope("scope1") as scope:
            net_conv = Model1.make_core_network(inputs)	# shape (?, 10)
        with tf.variable_scope("scope2") as scope:
            net_dnn = Model2.make_core_network(inputs)	# shape (?, 10)

        network = tf.concat([net_conv, net_dnn], 1, name="concat")	# shape (?, 20)
        network = tflearn.fully_connected(network, 10, activation="softmax")
        network = regression(network, optimizer='adam', learning_rate=0.01,
                             loss='categorical_crossentropy', name='target')

        self.model = tflearn.DNN(network, tensorboard_verbose=0)

    def load_from_two(self, m1fn, m2fn):
        self.model.load(m1fn, scope_for_restore="scope1", weights_only=True)
        self.model.load(m2fn, scope_for_restore="scope2", weights_only=True, create_new_session=False)

    def train(self, X, Y, testX, testY, n_epoch=1, snapshot_step=1000):
        # Training
        self.model.fit(X, Y, n_epoch=n_epoch, validation_set=(testX, testY),
                       snapshot_step=snapshot_step,
                       show_metric=True, run_id="model12")

#-----------------------------------------------------------------------------

X, Y, testX, testY = mnist.load_data(one_hot=True)

def prepare_model1_weights_file():
    tf.reset_default_graph()
    m1 = Model1()
    m1.train(X, Y, testX, testY, 2)
    m1.model.save("model1.tfl")

def prepare_model1_weights_file_in_scopeQ():
    tf.reset_default_graph()
    with tf.variable_scope("scopeQ") as scope:
        m1 = Model1()
    m1.model.fit({"scopeQ/input": X}, {"scopeQ/target": Y}, n_epoch=1, validation_set=0.1, show_metric=True, run_id="model1_scopeQ")
    m1.model.save("model1_scopeQ.tfl")

def prepare_model2_weights_file():
    tf.reset_default_graph()
    m2 = Model2()
    m2.train(X, Y, testX, testY, 1)
    m2.model.save("model2.tfl")

def demonstrate_loading_weights_into_different_scope():
    print("="*60 + " Demonstrate loading weights saved in scopeQ, into variables now in scopeA")
    tf.reset_default_graph()
    with tf.variable_scope("scopeA") as scope:
        m1a = Model1()
        print ("=" * 60 + " Trying to load model1 weights from scopeQ into scopeA")
        m1a.model.load("model1_scopeQ.tfl", variable_name_map=("scopeA", "scopeQ"), verbose=True)

def demonstrate_loading_weights_into_different_scope_using_custom_function():
    print("="*60 + " Demonstrate loading weights saved in scopeQ, into variables now in scopeA, using custom map function")
    tf.reset_default_graph()
    def vname_map(ename):	# variables were saved in scopeA, but we want to load into scopeQ
        name_in_file = ename.replace("scopeA", "scopeQ")
        print ("%s -> %s" % (ename, name_in_file))
        return name_in_file
    with tf.variable_scope("scopeA") as scope:
        m1a = Model1()
        print ("=" * 60 + " Trying to load model1 weights from scopeQ into scopeA")
        m1a.model.load("model1_scopeQ.tfl", variable_name_map=vname_map, verbose=True)

def demonstrate_loading_two_instances_of_model1():
    print("="*60 + " Demonstrate loading weights from model1 into two instances of model1 in scopeA and scopeB")
    tf.reset_default_graph()
    with tf.variable_scope("scopeA") as scope:
        m1a = Model1()
        print ("-" * 40 + " Trying to load model1 weights: should fail")
        try:
            m1a.model.load("model1.tfl", weights_only=True)
        except Exception as err:
            print ("Loading failed, with error as expected, because variables are in scopeA")
            print ("error: %s" % str(err))
        print ("-" * 40)

        print ("=" * 60 + " Trying to load model1 weights: should succeed")
        m1a.model.load("model1.tfl", scope_for_restore="scopeA", verbose=True, weights_only=True)

    with tf.variable_scope("scopeB") as scope:
        m1b = Model1()
        m1b.model.load("model1.tfl", scope_for_restore="scopeB", verbose=True, weights_only=True)
    print ("="*60 + " Successfully restored weights to two instances of model1, in different scopes")
            
def demonstrate_combined_model1_and_model2_network():
    print("="*60 + " Demonstrate loading weights from model1 and model2 into new mashup network model12")
    print ("-"*40 + " Creating mashup of model1 and model2 networks")
    tf.reset_default_graph()
    m12 = Model12()
    print ("-"*60 + " Loading model1 and model2 weights into mashup")
    m12.load_from_two("model1.tfl", "model2.tfl")
    print ("-"*60 + " Training mashup")
    m12.train(X, Y, testX, testY, 1)
    print ("-"*60 + " Saving mashup weights")
    m12.model.save("model12.tfl")
    print ("-"*60 + " Done")

print("="*77)
prepare_model1_weights_file()
prepare_model2_weights_file()
prepare_model1_weights_file_in_scopeQ()
print("-"*77)
print("-"*77)

demonstrate_loading_weights_into_different_scope()
demonstrate_loading_weights_into_different_scope_using_custom_function()
demonstrate_loading_two_instances_of_model1()
demonstrate_combined_model1_and_model2_network()

print("="*77)
