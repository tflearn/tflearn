from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np

from ..helpers.trainer import Trainer
from ..helpers.evaluator import Evaluator
from ..utils import feed_dict_builder, is_none, get_tensor_parents_placeholders


class DNN(object):
    """ Deep Neural Network Model.

    TODO: complete description

    Arguments:
        network: `Tensor`. Neural network to be used.
        tensorboard_verbose: `int`. Summary verbose level, it accepts
            different levels of tensorboard logs:
            ```python
            0: Loss, Accuracy (Best Speed).
            1: Loss, Accuracy, Gradients.
            2: Loss, Accuracy, Gradients, Weights.
            3: Loss, Accuracy, Gradients, Weights, Activations, Sparsity.
                (Best visualization)
            ```
        tensorboard_dir: `str`. Directory to store tensorboard logs.
            Default: "/tmp/tflearn_logs/"
        checkpoint_path: `str`. Path to store model checkpoints. If None,
            no model checkpoint will be saved. Default: None.
        best_checkpoint_path: `str`. Path to store the model when the validation rate reaches its
            highest point of the current training session and also is above best_val_accuracy. Default: None.
        max_checkpoints: `int` or None. Maximum amount of checkpoints. If
            None, no limit. Default: None.
        session: `Session`. A session for running ops. If None, a new one will
            be created. Note: When providing a session, variables must have been
            initialized already, otherwise an error will be raised.
        best_val_accuracy: `float` The minimum validation accuracy that needs to be
            achieved before a model weight's are saved to the best_checkpoint_path. This
            allows the user to skip early saves and also set a minimum save point when continuing
            to train a reloaded model. Default: 0.0.

    Attributes:
        trainer: `Trainer`. Handle model training.
        predictor: `Predictor`. Handle model prediction.
        session: `Session`. The current model session.

    """

    def __init__(self, network, clip_gradients=5.0, tensorboard_verbose=0,
                 tensorboard_dir="/tmp/tflearn_logs/", checkpoint_path=None,
                 best_checkpoint_path=None, max_checkpoints=None, session=None,
                 best_val_accuracy=0.0):
        assert isinstance(network, tf.Tensor), "'network' arg is not a Tensor!"
        self.net = network
        self.train_ops = tf.get_collection(tf.GraphKeys.TRAIN_OPS)
        self.trainer = Trainer(self.train_ops,
                               clip_gradients=clip_gradients,
                               tensorboard_dir=tensorboard_dir,
                               tensorboard_verbose=tensorboard_verbose,
                               checkpoint_path=checkpoint_path,
                               best_checkpoint_path=best_checkpoint_path,
                               max_checkpoints=max_checkpoints,
                               session=session,
                               best_val_accuracy=best_val_accuracy)
        self.session = self.trainer.session

        self.inputs = tf.get_collection(tf.GraphKeys.INPUTS)
        if len(self.inputs) == 0:
            raise Exception("No input data! Please add an 'input_data' layer "
                            "to your model (or add your input data "
                            "placeholder to tf.GraphKeys.INPUTS collection).")
        # verif_inputs = get_tensor_parents_placeholders(network)
        # if len(self.inputs) != len(verif_inputs):
        #     print("WARNING: TFLearn detected " + str(len(verif_inputs)) +
        #           " input placeholders, but tf collection '" +
        #           tf.GraphKeys.INPUTS + "' only contains " +
        #           str(len(self.inputs)) + ". If you define placeholders "
        #           "outside of TFLearn wrappers, make sure to add them to "
        #           "that collection.")

        self.targets = tf.get_collection(tf.GraphKeys.TARGETS)
        # TODO: error tracking when targets are actually used
        # if len(self.targets) == 0:
        #     raise Exception("No target data! Please add a 'regression' layer "
        #                     "to your model (or add your target data "
        #                     "placeholder to tf.GraphKeys.TARGETS collection).")
        self.predictor = Evaluator([self.net],
                                   session=self.session)

    def fit(self, X_inputs, Y_targets, n_epoch=10, validation_set=None,
            show_metric=False, batch_size=None, shuffle=None,
            snapshot_epoch=True, snapshot_step=None, excl_trainops=None,
            validation_batch_size=None, run_id=None, callbacks=[]):
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
                feed to train model.
            n_epoch: `int`. Number of epoch to run. Default: None.
            validation_set: `tuple`. Represents data used for validation.
                `tuple` holds data and targets (provided as same type as
                X_inputs and Y_targets). Additionally, it also accepts
                `float` (<1) to performs a data split over training data.
            show_metric: `bool`. Display or not accuracy at every step.
            batch_size: `int` or None. If `int`, overrides all network
                estimators 'batch_size' by this value.  Also overrides
                `validation_batch_size` if `int`, and if `validation_batch_size`
                is None.
            validation_batch_size: `int` or None. If `int`, overrides all network
                estimators 'validation_batch_size' by this value.
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
            callbacks: `Callback` or `list`. Custom callbacks to use in the
                training life cycle

        """
        if len(self.train_ops) == 0:
            raise Exception('tf collection "' + tf.GraphKeys.TRAIN_OPS + '" '
                            'is empty! Please make sure you are using '
                            '`regression` layer in your network.')

        if batch_size:
            for train_op in self.train_ops:
                train_op.batch_size = batch_size

        if batch_size is not None and validation_batch_size is None:
            validation_batch_size = batch_size

        if validation_batch_size:
            for train_op in self.train_ops:
                train_op.validation_batch_size = validation_batch_size

        valX, valY = None, None
        if validation_set:
            if isinstance(validation_set, float):
                valX = validation_set
                valY = validation_set
            elif type(validation_set) not in [tuple, list]:
                raise ValueError("validation_set must be a tuple or list: ("
                                 "valX, valY), " + str(type(validation_set))
                                 + " is not compatible!")
            else:
                valX = validation_set[0]
                valY = validation_set[1]

        # For simplicity we build sync dict synchronously but Trainer support
        # asynchronous feed dict allocation.
        # TODO: check memory impact for large data and multiple optimizers
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
            # Support for custom inputs not using dprep/daug
            if len(dprep_collection) > i:
                if dprep_collection[i] is not None:
                    dprep_dict[self.inputs[i]] = dprep_collection[i]
            if len(daug_collection) > i:
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
                         run_id=run_id,
                         callbacks=callbacks)

    def fit_batch(self, X_inputs, Y_targets):

        # For simplicity we build sync dict synchronously but Trainer support
        # asynchronous feed dict allocation.
        # TODO: check memory impact for large data and multiple optimizers
        feed_dict = feed_dict_builder(X_inputs, Y_targets, self.inputs,
                                      self.targets)
        feed_dicts = [feed_dict for i in self.train_ops]

        # Retrieve data preprocesing and augmentation
        dprep_dict, daug_dict = {}, {}
        dprep_collection = tf.get_collection(tf.GraphKeys.DATA_PREP)
        daug_collection = tf.get_collection(tf.GraphKeys.DATA_AUG)
        for i in range(len(self.inputs)):
            # Support for custom inputs not using dprep/daug
            if len(dprep_collection) > i:
                if dprep_collection[i] is not None:
                    dprep_dict[self.inputs[i]] = dprep_collection[i]
            if len(daug_collection) > i:
                if daug_collection[i] is not None:
                    daug_dict[self.inputs[i]] = daug_collection[i]
        return self.trainer.fit_batch(feed_dicts,
                                      dprep_dict=dprep_dict,
                                      daug_dict=daug_dict)

    def predict(self, X):
        """ Predict.

        Model prediction for given input data.

        Arguments:
            X: array, `list` of array (if multiple inputs) or `dict`
                (with inputs layer name as keys). Data to feed for prediction.

        Returns:
            array or `list` of array. The predicted probabilities.

        """
        feed_dict = feed_dict_builder(X, None, self.inputs, None)
        return self.predictor.predict(feed_dict)

    def predict_label(self, X):
        """ Predict Label.

        Predict class labels for input X.

        Arguments:
            X: array, `list` of array (if multiple inputs) or `dict`
                (with inputs layer name as keys). Data to feed for prediction.

        Returns:
            array or `list` of array. The predicted classes index array, sorted
            by descendant probability value.

        """
        feed_dict = feed_dict_builder(X, None, self.inputs, None)
        labels = np.argsort(self.predictor.predict(feed_dict))
        if labels.ndim == 1:
            return labels[::-1]
        else:
            return labels[:, ::-1]

    def save(self, model_file):
        """ Save.

        Save model weights.

        Arguments:
            model_file: `str`. Model path.

        """
        #with self.graph.as_default():
        self.trainer.save(model_file)

    def load(self, model_file, weights_only=False, **optargs):
        """ Load.

        Restore model weights.

        Arguments:
            model_file: `str`. Model path.
            weights_only: `bool`. If True, only weights will be restored (
                and not intermediate variable, such as step counter, moving
                averages...). Note that if you are using batch normalization,
                averages will not be restored as well.
            optargs: optional extra arguments for trainer.restore (see helpers/trainer.py)
                     These optional arguments may be used to limit the scope of
                     variables restored, and to control whether a new session is 
                     created for the restored variables.
        """
        self.trainer.restore(model_file, weights_only, **optargs)
        self.session = self.trainer.session
        self.predictor = Evaluator([self.net],
                                   session=self.session,
                                   model=None)
        for d in tf.get_collection(tf.GraphKeys.DATA_PREP):
            if d: d.restore_params(self.session)

    def get_weights(self, weight_tensor):
        """ Get Weights.

        Get a variable weights.

        Examples:
            ```
            dnn = DNNTrainer(...)
            w = dnn.get_weights(denselayer.W) # get a dense layer weights
            w = dnn.get_weights(convlayer.b) # get a conv layer biases
            ```

        Arguments:
            weight_tensor: `Tensor`. A Variable.

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

        Evaluate model metric(s) on given samples.

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
            The metric(s) score.

        """
        feed_dict = feed_dict_builder(X, Y, self.inputs, self.targets)
        ops = [o.metric for o in self.train_ops]
        return self.predictor.evaluate(feed_dict, ops, batch_size)

    def get_train_vars(self):
        ret = list()
        for tr_op in self.train_ops:
            ret = ret + tr_op.train_vars
        return ret
