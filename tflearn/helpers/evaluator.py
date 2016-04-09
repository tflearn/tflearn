from __future__ import division, print_function, absolute_import

import tensorflow as tf

import tflearn
from ..utils import to_list
from .. import metrics
from .trainer import evaluate as eval


class Evaluator(object):

    """ Evaluator.

    A class used for performing predictions or evaluate a model performances.

    Arguments:
        tensors: list of `Tensor`. A list of tensors to perform predictions.
        model: `str`. The model weights path (Optional).
        session: `Session`. The session to run the prediction (Optional).

    """

    def __init__(self, tensors, model=None, session=None):
        self.tensors = to_list(tensors)
        self.graph = self.tensors[0].graph
        self.model = model

        with self.graph.as_default():
            self.session = tf.Session()
            if session: self.session = session
            self.saver = tf.train.Saver()
            if model: self.saver.restore(self.session, model)

    def predict(self, feed_dict):
        """ predict.

        Run data through each tensor's network, and return prediction value.

        Arguments:
            feed_dict: `dict`. Feed data dictionary, with placeholders as
                keys, and data as values.

        Returns:
            An `array`. In case of multiple tensors to predict, array is a
            concatanation of each tensor prediction result.

        """
        with self.graph.as_default():
            tflearn.is_training(False, self.session)
            prediction = []
            for output in self.tensors:
                o_pred = self.session.run(output, feed_dict=feed_dict).tolist()
                for i, val in enumerate(o_pred): # Reshape pred per sample
                    if len(self.tensors) > 1:
                        if not len(prediction) > i: prediction.append([])
                        prediction[i].append(val)
                    else:
                        prediction.append(val)
            return prediction

    def evaluate(self, feed_dict, metric='accuracy', batch_size=128):
        raise NotImplementedError
