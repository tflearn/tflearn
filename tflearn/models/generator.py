from __future__ import division, print_function, absolute_import

import sys
import numpy as np
import tensorflow as tf

from ..helpers.trainer import Trainer, evaluate as eval
from ..helpers.evaluator import Evaluator
from ..utils import feed_dict_builder, is_none


class SequenceGenerator(object):
    """ Sequence Generator Model.

    A deep neural network model for generating sequences.

    Arguments:
        network: `Tensor`. Neural network to be used.
        dictionary: `dict`. A dictionary associating each sample with a key (
            usually integers). For example: {'a': 0, 'b': 1, 'c': 2, ...}.
        seq_maxlen: `int`. The maximum length of a sequence.
        tensorboard_verbose: `int`. Summary verbose level, it accepts
            different levels of tensorboard logs:
            ```python
            0 - Loss, Accuracy (Best Speed).
            1 - Loss, Accuracy, Gradients.
            2 - Loss, Accuracy, Gradients, Weights.
            3 - Loss, Accuracy, Gradients, Weights, Activations, Sparsity.
                (Best visualization)
            ```
        tensorboard_dir: `str`. Directory to store tensorboard logs.
            Default: "/tmp/tflearn_logs/"
        checkpoint_path: `str`. Path to store model checkpoints. If None,
            no model checkpoint will be saved. Default: None.
        max_checkpoints: `int` or None. Maximum amount of checkpoints. If
            None, no limit. Default: None.
        session: `Session`. A session for running ops. If None, a new one will
            be created. Note: When providing a session, variables must have been
            initialized already, otherwise an error will be raised.

    Attributes:
        trainer: `Trainer`. Handle model training.
        predictor: `Predictor`. Handle model prediction.
        session: `Session`. The current model session.

    """

    def __init__(self, network, dictionary=None, seq_maxlen=25,
                 clip_gradients=0.0, tensorboard_verbose=0,
                 tensorboard_dir="/tmp/tflearn_logs/",
                 checkpoint_path=None, max_checkpoints=None,
                 session=None):
        assert isinstance(network, tf.Tensor), "'network' arg is not a Tensor!"
        self.net = network
        self.train_ops = tf.get_collection(tf.GraphKeys.TRAIN_OPS)
        self.trainer = Trainer(self.train_ops,
                               clip_gradients=clip_gradients,
                               tensorboard_dir=tensorboard_dir,
                               tensorboard_verbose=tensorboard_verbose,
                               checkpoint_path=checkpoint_path,
                               max_checkpoints=max_checkpoints,
                               session=session)
        self.session = self.trainer.session
        self.inputs = tf.get_collection(tf.GraphKeys.INPUTS)
        self.targets = tf.get_collection(tf.GraphKeys.TARGETS)
        self.predictor = Evaluator([self.net],
                                   session=self.session)
        self.dic = dictionary
        self.rev_dic = reverse_dictionary(dictionary)
        self.seq_maxlen = seq_maxlen

    def fit(self, X_inputs, Y_targets, n_epoch=10, validation_set=None,
            show_metric=False, batch_size=None, shuffle=None,
            snapshot_epoch=True, snapshot_step=None, excl_trainops=None,
            run_id=None):
        """ Fit.

        Train model, feeding X_inputs and Y_targets to the network.

        NOTE: When not feeding dicts, data assignations is made by
            input/estimator layers creation order (For example, the second
            input layer created will be feeded by the second value of
            X_inputs list).

        Examples:
            ```python
            model.fit(X, Y) # Single input and output
            model.fit({'input1': X}, {'output1': Y}) # Single input and output
            model.fit([X1, X2], Y) # Mutliple inputs, Single output

            # validate with X_val and [Y1_val, Y2_val]
            model.fit(X, [Y1, Y2], validation_set=(X_val, [Y1_val, Y2_val]))
            # 10% of training data used for validation
            model.fit(X, Y, validation_set=0.1)
            ```

        Arguments:
            X_inputs: array, `list` of array (if multiple inputs) or `dict`
                (with inputs layer name as keys). Data to feed to train
                model.
            Y_targets: array, `list` of array (if multiple inputs) or `dict`
                (with estimators layer name as keys). Targets (Labels) to
                feed to train model. Usually set as the next element of a
                sequence, i.e. for x[0] => y[0] = x[1].
            n_epoch: `int`. Number of epoch to run. Default: None.
            validation_set: `tuple`. Represents data used for validation.
                `tuple` holds data and targets (provided as same type as
                X_inputs and Y_targets). Additionally, it also accepts
                `float` (<1) to performs a data split over training data.
            show_metric: `bool`. Display or not accuracy at every step.
            batch_size: `int` or None. If `int`, overrides all network
                estimators 'batch_size' by this value.
            shuffle: `bool` or None. If `bool`, overrides all network
                estimators 'shuffle' by this value.
            snapshot_epoch: `bool`. If True, it will snapshot model at the end
                of every epoch. (Snapshot a model will evaluate this model
                on validation set, as well as create a checkpoint if
                'checkpoint_path' specified).
            snapshot_step: `int` or None. If `int`, it will snapshot model
                every 'snapshot_step' steps.
            excl_trainops: `list` of `TrainOp`. A list of train ops to
                exclude from training process (TrainOps can be retrieve
                through `tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)`).
            run_id: `str`. Give a name for this run. (Useful for Tensorboard).

        """
        if batch_size:
            for train_op in self.train_ops:
                train_op.batch_size = batch_size

        valX, valY = None, None
        if validation_set:
            if isinstance(validation_set, float):
                valX = validation_set
                valY = validation_set
            else:
                valX = validation_set[0]
                valY = validation_set[1]

        # For simplicity we build sync dict synchronously but
        # Trainer support asynchronous feed dict allocation
        feed_dict = feed_dict_builder(X_inputs, Y_targets, self.inputs,
                                      self.targets)
        feed_dicts = [feed_dict for i in self.train_ops]

        val_feed_dicts = None
        if not (is_none(valX) or is_none(valY)):
            if isinstance(valX, float):
                val_feed_dicts = valX
            else:
                val_feed_dict = feed_dict_builder(valX, valY, self.inputs,
                                                  self.targets)
                val_feed_dicts = [val_feed_dict for i in self.train_ops]

        # Retrieve data preprocesing and augmentation
        dprep_dict, daug_dict = {}, {}
        dprep_collection = tf.get_collection(tf.GraphKeys.DATA_PREP)
        daug_collection = tf.get_collection(tf.GraphKeys.DATA_AUG)
        for i in range(len(self.inputs)):
            if dprep_collection[i] is not None:
                dprep_dict[self.inputs[i]] = dprep_collection[i]
            if daug_collection[i] is not None:
                daug_dict[self.inputs[i]] = daug_collection[i]

        self.trainer.fit(feed_dicts, val_feed_dicts=val_feed_dicts,
                         n_epoch=n_epoch,
                         show_metric=show_metric,
                         snapshot_step=snapshot_step,
                         snapshot_epoch=snapshot_epoch,
                         shuffle_all=shuffle,
                         dprep_dict=dprep_dict,
                         daug_dict=daug_dict,
                         excl_trainops=excl_trainops,
                         run_id=run_id)
        self.predictor = Evaluator([self.net],
                                   session=self.trainer.session)

    def _predict(self, X):
        feed_dict = feed_dict_builder(X, None, self.inputs, None)
        return self.predictor.predict(feed_dict)

    def generate(self, seq_length, temperature=0.5, seq_seed=None,
                 display=False):
        """ Generate.

        Generate a sequence. Temperature is controlling the novelty of
        the created sequence, a temperature near 0 will looks like samples
        used for training, while the higher the temperature, the more novelty.
        For optimal results, it is suggested to set sequence seed as some
        random sequence samples from training dataset.

        Arguments:
            seq_length: `int`. The generated sequence length.
            temperature: `float`. Novelty rate.
            seq_seed: `sequence`. A sequence used as a seed to generate a
                new sequence. Suggested to be a sequence from data used for
                training.
            display: `bool`. If True, print sequence as it is generated.

        Returns:
            The generated sequence.

        """

        generated = seq_seed[:]
        sequence = seq_seed[:]
        whole_sequence = seq_seed[:]

        if display: sys.stdout.write(str(generated))

        for i in range(seq_length):
            x = np.zeros((1, self.seq_maxlen, len(self.dic)))
            for t, char in enumerate(sequence):
                x[0, t, self.dic[char]] = 1.

            preds = self._predict(x)[0]
            next_index = _sample(preds, temperature)
            next_char = self.rev_dic[next_index]

            if type(sequence) == str:
                generated += next_char
                sequence = sequence[1:] + next_char
                whole_sequence += next_char
            else:
                generated.append(next_char)
                sequence = sequence[1:]
                sequence.append(next_char)
                whole_sequence.append(next_char)

            if display:
                sys.stdout.write(str(next_char))
                sys.stdout.flush()

        if display: print()

        return whole_sequence

    def save(self, model_file):
        """ Save.

        Save model weights.

        Arguments:
            model_file: `str`. Model path.

        """
        self.trainer.save(model_file)

    def load(self, model_file, **optargs):
        """ Load.

        Restore model weights.

        Arguments:
            model_file: `str`. Model path.
            optargs: optional extra arguments for trainer.restore (see helpers/trainer.py)
                     These optional arguments may be used to limit the scope of
                     variables restored, and to control whether a new session is
                     created for the restored variables.

        """
        self.trainer.restore(model_file, **optargs)
        self.session = self.trainer.session
        self.predictor = Evaluator([self.net],
                                   session=self.session,
                                   model=None)
        for d in tf.get_collection(tf.GraphKeys.DATA_PREP):
            if d: d.restore_params(self.session)

    def get_weights(self, weight_tensor):
        """ Get weights.

        Get a variable weights.

        Examples:
            sgen = SequenceGenerator(...)
            w = sgen.get_weights(denselayer.W) -- get a dense layer weights

        Arguments:
            weight_tensor: `tf.Tensor`. A Variable.

        Returns:
            `np.array`. The provided variable weights.
        """
        return weight_tensor.eval(self.trainer.session)

    def set_weights(self, tensor, weights):
        """ Set Weights.

        Assign a tensor variable a given value.

        Arguments:
            tensor: `Tensor`. The tensor variable to assign value.
            weights: The value to be assigned.

        """
        op = tf.assign(tensor, weights)
        self.trainer.session.run(op)

    def evaluate(self, X, Y, batch_size=128):
        """ Evaluate.

        Evaluate model on given samples.

        Arguments:
            X: array, `list` of array (if multiple inputs) or `dict`
                (with inputs layer name as keys). Data to feed to train
                model.
            Y: array, `list` of array (if multiple inputs) or `dict`
                (with estimators layer name as keys). Targets (Labels) to
                feed to train model. Usually set as the next element of a
                sequence, i.e. for x[0] => y[0] = x[1].
            batch_size: `int`. The batch size. Default: 128.

        Returns:
            The metric score.

        """
        feed_dict = feed_dict_builder(X, Y, self.inputs, self.targets)
        return eval(self.trainer.session, self.net, feed_dict, batch_size)


def reverse_dictionary(dic):
    # Build reverse dict
    rev_dic = {}
    for key in dic:
        rev_dic[dic[key]] = key
    return rev_dic


def _sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))
