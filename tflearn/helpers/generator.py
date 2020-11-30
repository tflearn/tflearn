from __future__ import division, print_function, absolute_import

import tensorflow.compat.v1 as tf

from ..utils import to_list


class SequenceGenerator(object):

    def __init__(self, net_outputs, model=None, session=None):
        self.net_outputs = to_list(net_outputs)
        self.graph = net_outputs[0].graph
        self.model = model

        with self.graph.as_default():
            self.session = tf.Session()
            if session: self.session = session
            self.saver = tf.train.Saver()
            if model: self.saver.restore(self.session, model)

    def predict(self, feed_dict):
        with self.graph.as_default():
            prediction = []
            for output in self.net_outputs:
                o_pred = self.session.run(output, feed_dict=feed_dict).tolist()
                for i, val in enumerate(o_pred): # Reshape pred per sample
                    if len(self.net_outputs) > 1:
                        if not len(prediction) > i: prediction.append([])
                        prediction[i].append(val)
                    else:
                        prediction.append(val)
            return prediction

    def generate(self):
        raise NotImplementedError
