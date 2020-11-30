from __future__ import division, print_function, absolute_import

import tensorflow.compat.v1 as tf

import tflearn
from ..utils import to_list
from .. import data_flow
from .. import metrics
from .trainer import evaluate_flow


class Evaluator(object):

    """ Evaluator.

    A class used for performing predictions and evaluate a model performance.

    Arguments:
        tensors: list of `Tensor`. A list of tensors to perform predictions.
        model: `str`. The model weights path (Optional).
        session: `Session`. The session to run the prediction (Optional).

    """

    def __init__(self, tensors, model=None, session=None):
        self.tensors = to_list(tensors)
        self.graph = self.tensors[0].graph
        self.model = model
        self.dprep_collection = tf.get_collection(tf.GraphKeys.DATA_PREP)
        self.inputs = tf.get_collection(tf.GraphKeys.INPUTS)

        with self.graph.as_default():
            self.session = tf.Session()
            if session: self.session = session
            self.saver = tf.train.Saver()
            if model: self.saver.restore(self.session, model)

    def predict(self, feed_dict):
        """ predict.

        Run data through the provided network and return the result value.

        Arguments:
            feed_dict: `dict`. Feed data dictionary, with placeholders as
                keys, and data as values.

        Returns:
            An `array`. In case of multiple tensors to predict, each tensor's
            prediction result is concatenated.

        """
        with self.graph.as_default():
            # Data Preprocessing
            dprep_dict = dict()
            for i in range(len(self.inputs)):
                # Support for custom inputs not using dprep/daug
                if len(self.dprep_collection) > i:
                    if self.dprep_collection[i] is not None:
                        dprep_dict[self.inputs[i]] = self.dprep_collection[i]
            # Apply pre-processing
            if len(dprep_dict) > 0:
                for k in dprep_dict:
                    feed_dict[k] = dprep_dict[k].apply(feed_dict[k])

            # Prediction for each tensor
            tflearn.is_training(False, self.session)
            prediction = []
            if len(self.tensors) == 1:
                return self.session.run(self.tensors[0], feed_dict=feed_dict)
            else:
                for output in self.tensors:
                    o_pred = self.session.run(output, feed_dict=feed_dict).tolist()
                    for i, val in enumerate(o_pred): # Reshape pred per sample
                        if len(self.tensors) > 1:
                            if not len(prediction) > i: prediction.append([])
                            prediction[i].append(val)
                return prediction

    def evaluate(self, feed_dict, ops, batch_size=128):
        """ Evaluate.

        Evaluate a list of tensors over a whole dataset. Generally,
        'ops' argument are average performance metrics (such as average mean,
        top-3, etc...)

        Arguments:
            feed_dict: `dict`. The feed dictionary of data.
            ops: list of `Tensors`. The tensors to evaluate.
            batch_size: `int`. A batch size.

        Returns:
            The mean average result per tensor over all batches.

        """
        tflearn.is_training(False, self.session)
        coord = tf.train.Coordinator()
        inputs = tf.get_collection(tf.GraphKeys.INPUTS)
        # Data Preprocessing
        dprep_dict = {}
        dprep_collection = tf.get_collection(tf.GraphKeys.DATA_PREP)
        for i in range(len(inputs)):
            # Support for custom inputs not using dprep/daug
            if len(dprep_collection) > i:
                if dprep_collection[i] is not None:
                    dprep_dict[inputs[i]] = dprep_collection[i]
        # Data Flow
        df = data_flow.FeedDictFlow(feed_dict, coord,
                                    batch_size=batch_size,
                                    dprep_dict=dprep_dict,
                                    daug_dict=None,
                                    index_array=None,
                                    num_threads=1)

        return evaluate_flow(self.session, ops, df)
