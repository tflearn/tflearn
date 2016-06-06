# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.python.training import optimizer as tf_optimizer

import tflearn
from .. import callbacks
from ..config import init_training_mode
from ..utils import to_list, id_generator, check_dir_name, standarize_dict, \
    get_dict_first_element, make_batches, slice_array, check_scope_path, \
    check_restore_tensor
from .. import data_flow

from .summarizer import summaries, summarize, summarize_gradients, \
    summarize_variables, summarize_activations


class Trainer(object):
    """ Trainer.

    Generic class to handle any TensorFlow graph training. It requires
    the use of `TrainOp` to specify all optimization parameters.

    Arguments:
        train_ops: list of `TrainOp`. A list of a network training
            operations for performing optimizations.
        graph: `tf.Graph`. The TensorFlow graph to use. Default: default tf
            graph.
        clip_gradients: `float`. Clip gradient. Default: 5.0.
        tensorboard_dir: `str`. Tensorboard log directory.
            Default: "/tmp/tflearn_logs/".
        tensorboard_verbose: `int`. Verbose level. It supports:
            ```python
            0 - Loss, Accuracy. (Best Speed)
            1 - Loss, Accuracy, Gradients.
            2 - Loss, Accuracy, Gradients, Weights.
            3 - Loss, Accuracy, Gradients, Weights, Activations, Sparsity.
                (Best Visualization)
            ```
        checkpoint_path: `str`. Path to store model checkpoints. If None,
            no model checkpoint will be saved. Default: None.
        max_checkpoints: `int` or None. Maximum amount of checkpoints. If
            None, no limit. Default: None.
        keep_checkpoint_every_n_hours: `float`. Number of hours between each
            model checkpoints.
        random_seed: `int`. Random seed, for test reproductivity.
            Default: None.
        session: `Session`. A session for running ops. If None, a new one will
            be created. Note: When providing a session, variables must have been
            initialized already, otherwise an error will be raised.

    """

    def __init__(self, train_ops, graph=None, clip_gradients=5.0,
                 tensorboard_dir="/tmp/tflearn_logs/",
                 tensorboard_verbose=0, checkpoint_path=None,
                 max_checkpoints=None,
                 keep_checkpoint_every_n_hours=10000.0, random_seed=None,
                 session=None):

        self.graph = tf.get_default_graph()
        if graph:
            self.graph = graph

        with self.graph.as_default():

            init_training_mode()

            train_ops = to_list(train_ops)
            duplicate_identical_ops(train_ops)

            if random_seed:
                tf.set_random_seed(random_seed)
            self.restored = False
            self.tensorboard_dir = check_dir_name(tensorboard_dir)
            self.training_step = 0

            self.train_ops = to_list(train_ops)
            self.validate_trainop_names()

            self.global_loss = None
            self.global_step = tf.Variable(0., name='Global_Step',
                                           trainable=False)
            self.incr_global_step = tf.assign(self.global_step,
                                              tf.add(self.global_step, 1))

            config = None
            tflearn_conf = tf.get_collection(tf.GraphKeys.GRAPH_CONFIG)
            if tflearn_conf:
                config = tflearn_conf[0]

            if not session:
                self.session = tf.Session(config=config)
            else:
                self.session = session
                self.restored = True

            self.coord = tf.train.Coordinator()

            for i, train_op in enumerate(self.train_ops):

                # For display simplicity in Tensorboard, if only one optmizer,
                # we don't display its name
                if len(train_ops) == 1:
                    train_op.scope_name = ""

                train_op.initialize_training_ops(i, self.session,
                                                 tensorboard_verbose,
                                                 clip_gradients)

            # Saver for saving a model
            self.saver = tf.train.Saver(
                max_to_keep=max_checkpoints,
                keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)
            # Saver for restoring a model (With exclude variable list)
            all_vars = tf.get_collection(tf.GraphKeys.VARIABLES)
            excl_vars = tf.get_collection(tf.GraphKeys.EXCL_RESTORE_VARS)
            to_restore = [item for item in all_vars
                          if check_restore_tensor(item, excl_vars)]
            self.restorer = tf.train.Saver(
                var_list=to_restore,
                max_to_keep=max_checkpoints,
                keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

            self.checkpoint_path = checkpoint_path

            if not self.restored:
                init = tf.initialize_all_variables()
                self.session.run(init)

    def fit(self, feed_dicts, n_epoch=10, val_feed_dicts=None, show_metric=False,
            snapshot_step=None, snapshot_epoch=True, shuffle_all=None,
            dprep_dict=None, daug_dict=None, excl_trainops=None, run_id=None):
        """ fit.

        Train network with feeded data dicts.

        Examples:
            ```python
            # 1 Optimizer
            trainer.fit(feed_dicts={input1: X, output1: Y},
                        val_feed_dicts={input1: X, output1: Y})
            trainer.fit(feed_dicts={input1: X1, input2: X2, output1: Y},
                        val_feed_dicts=0.1) # 10% of data used for validation

            # 2 Optimizers
            trainer.fit(feed_dicts=[{in1: X1, out1:Y}, {in2: X2, out2:Y2}],
                        val_feed_dicts=[{in1: X1, out1:Y}, {in2: X2, out2:Y2}])
            ```

        Arguments:
            feed_dicts: `dict` or list of `dict`. The dictionary to feed
                data to the network. It follows Tensorflow feed dict
                specifications: '{placeholder: data}'. In case of multiple
                optimizers, a list of dict is expected, that will
                respectively feed optimizers.
            n_epoch: `int`. Number of epoch to runs.
            val_feed_dicts: `dict`, list of `dict`, `float` or list of
                `float`. The data used for validation. Feed dict are
                following the same specification as `feed_dicts` above. It
                is also possible to provide a `float` for splitting training
                data for validation (Note that this will shuffle data).
            show_metric: `bool`. If True, accuracy will be calculated and
                displayed at every step. Might give slower training.
            snapshot_step: `int`. If not None, the network will be snapshot
                every provided step (calculate validation loss/accuracy and
                save model, if a `checkpoint_path` is specified in `Trainer`).
            snapshot_epoch: `bool`. If True, snapshot the network at the end
                of every epoch.
            shuffle_all: `bool`. If True, shuffle all data batches (overrides
                `TrainOp` shuffle parameter behavior).
            dprep_dict: `dict` with `Placeholder` as key and
                `DataPreprocessing` as value. Apply realtime data
                preprocessing to the given placeholders (Applied at training
                and testing time).
            daug_dict: `dict` with `Placeholder` as key and
                `DataAugmentation` as value. Apply realtime data
                augmentation to the given placeholders (Only applied at
                training time).
            excl_trainops: `list` of `TrainOp`. A list of train ops to
                exclude from training process.
            run_id: `str`. A name for the current run. Used for Tensorboard
                display. If no name provided, a random one will be generated.

        """

        if not run_id:
            run_id = id_generator(6)
        print("---------------------------------")
        print("Run id: " + run_id)
        print("Log directory: " + self.tensorboard_dir)

        original_train_ops = list(self.train_ops)
        # Remove excluded train_ops
        for t in self.train_ops:
            if excl_trainops and t in excl_trainops:
                self.train_ops.remove(t)

        # shuffle is an override for simplicty, it will overrides every
        # training op batch shuffling
        if isinstance(shuffle_all, bool):
            for t in self.train_ops: t.shuffle = shuffle_all

        with self.graph.as_default():

            try:
                self.summ_writer = tf.train.SummaryWriter(
                    self.tensorboard_dir + run_id, self.session.graph)
            except Exception: # TF 0.7
                self.summ_writer = tf.train.SummaryWriter(
                    self.tensorboard_dir + run_id, self.session.graph_def)

            feed_dicts = to_list(feed_dicts)
            for d in feed_dicts: standarize_dict(d)
            val_feed_dicts = to_list(val_feed_dicts)
            if val_feed_dicts:
                [standarize_dict(d) for d in val_feed_dicts if not
                 isinstance(d, float)]

            termlogger = callbacks.TermLogger(self.training_step)
            modelsaver = callbacks.ModelSaver(self.save,
                                              self.training_step,
                                              self.checkpoint_path,
                                              snapshot_epoch)

            for i, train_op in enumerate(self.train_ops):
                vd = val_feed_dicts[i] if val_feed_dicts else None
                # Prepare all train_ops for fitting
                train_op.initialize_fit(feed_dicts[i], vd, dprep_dict,
                                        daug_dict, show_metric,
                                        self.summ_writer, self.coord)

                # Prepare TermLogger for training diplay
                metric_term_name = None
                if train_op.metric is not None:
                    if hasattr(train_op.metric, 'm_name'):
                        metric_term_name = train_op.metric.m_name
                    else:
                        metric_term_name = train_op.metric.name.split(':')[0]
                termlogger.add(train_op.n_train_samples,
                               val_size=train_op.n_val_samples,
                               metric_name=metric_term_name,
                               name=train_op.name)

            max_batches_len = np.max([t.n_batches for t in self.train_ops])

            termlogger.on_train_begin()
            modelsaver.on_epoch_begin()

            try:
                for epoch in range(n_epoch):

                    termlogger.on_epoch_begin()
                    modelsaver.on_epoch_begin()

                    # Global epoch are defined as loop over all data (whatever
                    # which data input), so one epoch loop in a multi-inputs
                    # model is equal to max(data_input) size.
                    for batch_step in range(max_batches_len):

                        self.training_step += 1
                        termlogger.on_batch_begin()
                        modelsaver.on_batch_begin()

                        global_loss, global_acc = 0., 0.

                        for i, train_op in enumerate(self.train_ops):

                            termlogger.on_sub_epoch_begin()
                            modelsaver.on_sub_batch_begin()

                            snapshot = train_op._train(self.training_step,
                                                       snapshot_epoch,
                                                       snapshot_step,
                                                       show_metric)
                            global_loss += train_op.loss_value
                            if train_op.acc_value and global_acc:
                                global_acc += train_op.acc_value / len(
                                    self.train_ops)
                            else:
                                global_acc = None

                            data_status = train_op.train_dflow.data_status
                            # Optimizer batch end
                            termlogger.on_sub_batch_end(i, data_status.epoch,
                                                        data_status.current_iter,
                                                        train_op.loss_value,
                                                        train_op.acc_value,
                                                        train_op.val_loss,
                                                        train_op.val_acc)
                            modelsaver.on_sub_batch_end()

                        # All optimizers batch end
                        self.session.run(self.incr_global_step)
                        termlogger.on_batch_end(global_loss, global_acc,
                                                snapshot)
                        modelsaver.on_batch_end(snapshot)

                    # Epoch end
                    termlogger.on_epoch_end()
                    modelsaver.on_epoch_end()

            finally:
                termlogger.on_train_end()
                modelsaver.on_train_end()
                for t in self.train_ops:
                    t.train_dflow.interrupt()
                # Set back train_ops
                self.train_ops = original_train_ops

    def save(self, model_file, global_step=None):
        """ save.

        Save a Tensorflow model

        Arguments:
            model_file: `str`. Saving path of tensorflow model
            global_step: `float`. The training step to append to the
                model file name (optional).

        """
        # Temp workaround for tensorflow 0.7.0 dict proto serialization issue
        try:
            # Try latest api
            l = tf.get_collection_ref("summary_tags")
        except Exception:
            l = tf.get_collection("summary_tags")
        l_stags = list(l)
        del l[:]

        try:
            # Try latest api
            l1 = tf.get_collection_ref(tf.GraphKeys.DATA_PREP)
            l2 = tf.get_collection_ref(tf.GraphKeys.DATA_AUG)
        except Exception:
            l1 = tf.get_collection(tf.GraphKeys.DATA_PREP)
            l2 = tf.get_collection(tf.GraphKeys.DATA_AUG)
        l1_dtags = list(l1)
        l2_dtags = list(l2)
        del l1[:]
        del l2[:]

        try: # Do not save exclude variables
            l3 = tf.get_collection_ref(tf.GraphKeys.EXCL_RESTORE_VARS)
        except Exception:
            l3 = tf.get_collection(tf.GraphKeys.EXCL_RESTORE_VARS)
        l3_tags = list(l3)
        del l3[:]

        # Temp workaround for tensorflow 0.7.0 relative path issue
        if model_file[0] not in ['/', '~']: model_file = './' + model_file

        self.saver.save(self.session, model_file, global_step=global_step)

        # 0.7 workaround, restore values
        for t in l_stags:
            tf.add_to_collection("summary_tags", t)
        for t in l1_dtags:
            tf.add_to_collection(tf.GraphKeys.DATA_PREP, t)
        for t in l2_dtags:
            tf.add_to_collection(tf.GraphKeys.DATA_AUG, t)
        for t in l3_tags:
            tf.add_to_collection(tf.GraphKeys.EXCL_RESTORE_VARS, t)

    def restore(self, model_file):
        """ restore.

        Restore a Tensorflow model

        Arguments:
            model_file: path of tensorflow model to restore

        """
        self.close_session()
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())
        self.restorer.restore(self.session, model_file)
        for o in self.train_ops:
            o.session = self.session
        self.restored = True
        self.training_step = int(self.global_step.eval(self.session))

    def close_session(self):
        """ Close session """
        self.session.close()

    def validate_trainop_names(self):
        """ Give names to all TrainOp, handle no names and duplicated names """
        t_len = len(self.train_ops)
        # Rename optimizers without name
        for i in range(t_len):
            if not self.train_ops[i].name:
                self.train_ops[i].name = 'Optimizer'
                self.train_ops[i].scope_name = 'Optimizer'
        # Handle duplicate names
        for i in range(t_len):
            dupl = 0
            for j in range(i+1, t_len):
                if not self.train_ops[i].name:
                    break
                if self.train_ops[i].name == self.train_ops[j].name:
                    if dupl == 0:
                        self.train_ops[i].name += '_' + str(dupl)
                        self.train_ops[i].scope_name = self.train_ops[i].name
                    dupl += 1
                    self.train_ops[j].name += '_' + str(dupl)
                    self.train_ops[j].scope_name = self.train_ops[j].name


class TrainOp(object):
    """ TrainOp.

    TrainOp represents a set of operation used for optimizing a network.

    A TrainOp is meant to hold all training parameters of an optimizer.
    `Trainer` class will then instantiate them all specifically considering all
    optimizers of the network (set names, scopes... set optimization ops...).

    Arguments:
        loss: `Tensor`. Loss operation to evaluate network cost.
            Optimizer will use this cost function to train network.
        optimizer: `Optimizer`. Tensorflow Optimizer. The optimizer to
            use to train network.
        metric:  `Tensor`. The metric tensor to be used for evaluation.
        batch_size: `int`. Batch size for data feeded to this optimizer.
            Default: 64.
        ema: `float`. Exponential moving averages.
        trainable_vars: list of `tf.Variable`. List of trainable variables to
            use for training. Default: all trainable variables.
        shuffle: `bool`. Shuffle data.
        step_tensor: `tf.Tensor`. A variable holding training step. If not
            provided, it will be created. Early defining the step tensor
            might be useful for network creation, such as for learning rate
            decay.
        input_vars: list of `Variable`. The input data for this training op
            to be transformed by data augmentation. Default:
            tf.GraphKeys.INPUTS collection. (optional) Only necessary when
            using data augmentation.
        data_preprocessing: A `DataPreprocessing` subclass object to manage
            real-time data pre-processing when training and predicting (such
            as zero center data, std normalization...).
        data_augmentation: `DataAugmentation`. A `DataAugmentation` subclass
            object to manage real-time data augmentation while training (
            such as random image crop, random image flip, random sequence
            reverse...).
        name: `str`. A name for this class (optional).
        graph: `tf.Graph`. Tensorflow Graph to use for training. Default:
            default tf graph.

    """

    def __init__(self, loss, optimizer, metric=None, batch_size=64, ema=0.,
                 trainable_vars=None, shuffle=True, step_tensor=None,
                 name=None, graph=None):
        self.graph = tf.get_default_graph()
        if graph:
            self.graph = graph

        self.name = name
        self.scope_name = name

        # Ops
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric
        self.metric_summ_name = ""
        if metric is not None:
            self.metric_summ_name = metric.name.split('/')[0]
        self.grad = None
        self.apply_grad = None
        self.summ_op = None
        self.val_summary_op = None

        self.train_vars = trainable_vars
        self.shuffle = shuffle

        self.batch_size = batch_size
        self.n_batches = 0

        self.ema = ema

        self.feed_dict = None
        self.val_feed_dict = None
        self.loss_value = None
        self.val_loss = None
        self.acc_value = None
        self.val_acc = None

        if step_tensor is None:
            with self.graph.as_default():
                self.training_steps = tf.Variable(0., name="Training_step",
                                                  trainable=False)
        else:
            self.training_steps = step_tensor

        # Building
        if not isinstance(self.loss, tf.Tensor):
            raise ValueError("Unknown Loss type")

        if not isinstance(self.optimizer, tf_optimizer.Optimizer):
            raise ValueError("Unknown Optimizer")

        if self.train_vars is None:
            self.train_vars = tf.trainable_variables()
        else:
            self.train_var = to_list(self.train_vars)

        self.train = None

    def initialize_training_ops(self, i, session, tensorboard_verbose,
                                clip_gradients):
        """ initialize_training_ops.

        Initialize all ops used for training. Because a network can have
        multiple optimizers, an id 'i' is allocated to differentiate them.
        This is meant to be used by `Trainer` when initializing all train ops.

        Arguments:
            i: `int`. This optimizer training process ID.
            session: `tf.Session`. The session used to train the network.
            tensorboard_verbose: `int`. Logs verbose. Supports:
                ```
                0 - Loss, Accuracy.
                1 - Loss, Accuracy, Gradients.
                2 - Loss, Accuracy, Gradients, Weights.
                3 - Loss, Accuracy, Gradients, Weights, Activations, Sparsity..
                ```
            clip_gradients: `float`. Option for clipping gradients.
        """
        self.session = session

        # Variables holding mean validation loss and accuracy, assigned after
        # each model evaluation (by batch). For visualization in Tensorboard.
        self.val_loss_T = tf.Variable(0., name='val_loss', trainable=False)
        self.val_acc_T = tf.Variable(0., name='val_acc', trainable=False)

        # Creating the accuracy moving average, for better visualization.
        if self.metric is not None:
            self.acc_averages = \
                tf.train.ExponentialMovingAverage(0.9, self.training_steps,
                                                  name='moving_avg')
            acc_avg_op = self.acc_averages.apply([self.metric])
        else:
            acc_avg_op = tf.no_op()

        # Compute total loss, which is the loss of all optimizers plus the
        # loss of all regularizers. Then, we summarize those losses for
        # visualization in Tensorboard.
        with tf.name_scope(self.name):
            lss = [self.loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = tf.add_n(lss, name="Total_Loss")
            loss_avg_op = summaries.add_loss_summaries(
                total_loss,
                self.loss,
                regul_losses_collection_key=tf.GraphKeys.REGULARIZATION_LOSSES,
                name_prefix=self.scope_name,
                summaries_collection_key=self.name + "_training_summaries",
                exp_moving_avg=0.9,
                ema_num_updates=self.training_steps)

            # Compute gradients operations
            with tf.control_dependencies([loss_avg_op, acc_avg_op]):
                self.grad = tf.gradients(total_loss, self.train_vars)
                if clip_gradients > 0.0:
                    self.grad, self.grad_norm = \
                        tf.clip_by_global_norm(self.grad, clip_gradients)

            self.grad = list(zip(self.grad, self.train_vars))
            self.apply_grad = self.optimizer.apply_gradients(
                    grads_and_vars=self.grad,
                    global_step=self.training_steps,
                    name="apply_grad_op_" + str(i))

            # Create other useful summary (weights, grads, activations...)
            # according to 'tensorboard_verbose' level.
            self.create_summaries(tensorboard_verbose)

            # Track the moving averages of trainable variables
            if self.ema > 0.:
                var_averages = tf.train.ExponentialMovingAverage(
                        self.ema, self.training_steps)
                var_averages_op = var_averages.apply(self.train_vars)

                with tf.control_dependencies([var_averages_op]):
                    with tf.control_dependencies([self.apply_grad]):
                        self.train = tf.no_op(name="train_op_" + str(i))
            else:
                with tf.control_dependencies([self.apply_grad]):
                    self.train = tf.no_op(name="train_op_" + str(i))

    def initialize_fit(self, feed_dict, val_feed_dict, dprep_dict, daug_dict,
                       show_metric, summ_writer, coord):
        """ initialize_fit.

        Initialize data for feeding the training process. It is meant to
        be used by `Trainer` before starting to fit data.

        Arguments:
            feed_dict: `dict`. The data dictionary to feed.
            val_feed_dict: `dict` or `float`. The validation data dictionary to
                feed or validation split.
            dprep_dict: `dict`. Data Preprocessing dict (with placeholder as
                key and corresponding `DataPreprocessing` object as value).
            daug_dict: `dict`. Data Augmentation dict (with placeholder as
                key and corresponding `DataAugmentation` object as value).
            show_metric: `bool`. If True, display accuracy at every step.
            summ_writer: `SummaryWriter`. The summary writer to use for
                Tensorboard logging.

        """
        self.summary_writer = summ_writer
        self.feed_dict = feed_dict
        self.val_feed_dict = val_feed_dict
        self.n_train_samples = len(get_dict_first_element(feed_dict))

        self.index_array = np.arange(self.n_train_samples)
        self.n_val_samples = 0
        # Validation Split
        #TODO: Optional per key validation split
        if isinstance(val_feed_dict, float):
            split_at = int(self.n_train_samples * (1 - val_feed_dict))
            # Shuffle Data
            np.random.shuffle(self.index_array)
            self.val_index_array = self.index_array[split_at:]
            self.index_array = self.index_array[:split_at]
            self.n_train_samples = len(self.index_array)
            self.n_val_samples = len(self.val_index_array)
            val_feed_dict = feed_dict
        elif val_feed_dict is not None:
            self.val_index_array = None
            self.n_val_samples = len(get_dict_first_element(val_feed_dict))

        if dprep_dict:
            for k in dprep_dict:
                assert feed_dict[k] is not None, \
                    "Unknown DataPreprocessing dict key!"
                dprep_dict[k].initialize(feed_dict[k], self.session)
        self.train_dflow = data_flow.FeedDictFlow(feed_dict, coord,
                                                  continuous=True,
                                                  batch_size=self.batch_size,
                                                  dprep_dict=dprep_dict,
                                                  daug_dict=daug_dict,
                                                  index_array=self.index_array,
                                                  num_threads=1,
                                                  shuffle=self.shuffle)

        self.n_batches = len(self.train_dflow.batches)
        self.train_dflow.start()
        # TODO: Optimize data_flow to not start/restart threads (cost time)
        # every time testing
        if val_feed_dict:
            self.test_dflow = data_flow.FeedDictFlow(val_feed_dict, coord,
                                                     batch_size=self.batch_size,
                                                     dprep_dict=dprep_dict,
                                                     daug_dict=None,
                                                     index_array=self.val_index_array,
                                                     num_threads=1)

        self.create_testing_summaries(show_metric, self.metric_summ_name,
                                      val_feed_dict)

    def _train(self, training_step, snapshot_epoch, snapshot_step,
               show_metric):
        """ Training process for this optimizer.

        Arguments:
            training_step: `int`. The global step.
            snapshot_epoch: `bool`. If True, snapshot network at each epoch.
            snapshot_step: `int`. If not None, snapshot network given 'step'.
            show_metric: `bool`. If True, display accuracy at every step.

        """
        self.loss_value, self.acc_value = None, None
        self.val_loss, self.val_acc = None, None
        train_summ_str, test_summ_str = None, None
        snapshot = False
        epoch = self.train_dflow.data_status.epoch

        feed_batch = self.train_dflow.next()
        tflearn.is_training(True, session=self.session)
        _, train_summ_str = self.session.run([self.train, self.summ_op],
                                             feed_batch)

        # Retrieve loss value from summary string
        sname = "- Loss/" + self.scope_name
        self.loss_value = summaries.get_value_from_summary_string(
            sname, train_summ_str)

        if show_metric and self.metric is not None:
            # Retrieve accuracy value from summary string
            sname = "- " + self.metric_summ_name + "/" + self.scope_name
            self.acc_value = summaries.get_value_from_summary_string(
                sname, train_summ_str)

        if epoch != self.train_dflow.data_status.epoch:
            if snapshot_epoch:
                snapshot = True

        # Check if step reached snapshot step
        if snapshot_step:
            if training_step % snapshot_step == 0:
                snapshot = True

        # Calculate validation
        if snapshot and self.val_feed_dict:
            tflearn.is_training(False, session=self.session)
            # Evaluation returns the mean over all batches.
            eval_ops = [self.loss]
            if show_metric and self.metric is not None:
                eval_ops.append(self.metric)
            e = evaluate_flow(self.session, eval_ops, self.test_dflow)
            self.val_loss = e[0]
            if show_metric and self.metric is not None:
                self.val_acc = e[1]

            # Set evaluation results to variables, to be summarized.
            if show_metric:
                update_val_op = [tf.assign(self.val_loss_T, self.val_loss),
                                 tf.assign(self.val_acc_T, self.val_acc)]
            else:
                update_val_op = tf.assign(self.val_loss_T, self.val_loss)
            self.session.run(update_val_op)

            # Run summary operation.
            test_summ_str = self.session.run(self.val_summary_op)

        # Write to Tensorboard
        #TODO: Delete?
        n_step = self.training_steps.eval(session=self.session)
        if n_step > 1:
            if train_summ_str:
                self.summary_writer.add_summary(
                    train_summ_str, n_step)
            if test_summ_str:
                self.summary_writer.add_summary(
                    test_summ_str, n_step)

        return snapshot

    def duplicate(self):
        """ Returns a duplicated `TrainOp` """
        return TrainOp(self.loss, optimizer=self.optimizer,
                       batch_size=self.batch_size, ema=self.ema,
                       metric=self.metric,
                       trainable_vars=self.train_vars,
                       shuffle=self.shuffle)

    def create_summaries(self, verbose=2):
        """ Create summaries with `verbose` level """

        summ_collection = self.name + "_training_summaries"

        if verbose in [3]:
            # Summarize activations
            activations = tf.get_collection(tf.GraphKeys.ACTIVATIONS)
            summarize_activations(activations, summ_collection)
        if verbose in [2, 3]:
            # Summarize variable weights
            summarize_variables(self.train_vars, summ_collection)
        if verbose in [1, 2, 3]:
            # Summarize gradients
            summarize_gradients(self.grad, summ_collection)

        self.summ_op = tf.merge_summary(tf.get_collection(summ_collection))

    def create_testing_summaries(self, show_metric=False,
                                 metric_name="Accuracy", validation_set=None):
        """ Create accuracy and validation summaries """

        tr_summ_collection = self.name + "_training_summaries"
        te_summ_collection = self.name + "_testing_summaries"

        mn = metric_name.replace('/Mean:0/', '')

        if show_metric and self.metric is not None:
            # Summarize Raw Accuracy
            sname = "- " + mn + "/" + self.scope_name + " (raw)"
            summarize(self.metric, "scalar", sname, tr_summ_collection)
            # Summarize Accuracy's moving averages
            sname = "- " + mn + "/" + self.scope_name
            self.summ_op = summarize(self.acc_averages.average(self.metric),
                                     "scalar", sname, tr_summ_collection)

        if validation_set is not None:
            # Summarive Validation Loss
            loss_val_name = "- Loss/" + self.scope_name + "/Validation"
            loss_val_name = check_scope_path(loss_val_name)
            self.val_summary_op = summarize(self.val_loss_T, "scalar",
                                            loss_val_name, te_summ_collection)
            if show_metric and self.metric is not None:
                # Summarize Validation Accuracy
                acc_val_name = "- " + mn + "/" + self.scope_name + "/Validation"
                acc_val_name = check_scope_path(acc_val_name)
                self.val_summary_op = summarize(self.val_acc_T, "scalar",
                                                acc_val_name,
                                                te_summ_collection)


def duplicate_identical_ops(ops):
    """ Duplicate identical `TrainOp` """
    for i in range(len(ops)):
        for j in range(i+1, len(ops)):
            if ops[i] == ops[j]:
                ops[j] = ops[i].duplicate()


def get_current_batch_size(feed_batch, dataflow):
    if hasattr(feed_batch, 'iteritems'):
        iterator = feed_batch.iteritems
    else:
        iterator = feed_batch.items
    for k, v in iterator():
        if k.get_shape()[0].value == None:
            if type(v) is list:
              return len(v)
            else:
              return int(v.shape[0])
    return dataflow.batch_size


def evaluate_flow(session, ops_to_evaluate, dataflow):
        if not isinstance(ops_to_evaluate, list):
            ops_to_evaluate = [ops_to_evaluate]
        tflearn.is_training(False, session)
        dataflow.reset()
        dataflow.start()
        res = [0. for i in ops_to_evaluate]
        feed_batch = dataflow.next()

        while feed_batch:
            r = session.run(ops_to_evaluate, feed_batch)
            current_batch_size = get_current_batch_size(feed_batch, dataflow)
            for i in range(len(r)):
                res[i] += r[i] * current_batch_size
            feed_batch = dataflow.next()
        res = [r / dataflow.n_samples for r in res]
        return res


def evaluate(session, op_to_evaluate, feed_dict, batch_size):
        """ evaluate.

        Evaluate an operation with provided data dict using a batch size
        to save GPU memory.

        Args:
            session: `tf.Session`. Session for running operations.
            op_to_evaluate: `tf.Op`. Operation to be evaluated.
            feed_dict: `dict`. Data dictionary to feed op_to_evaluate.
            batch_size: `int`. Batch size to be used for evaluation.

        Ret:
            `float`. op_to_evaluate mean over all batches.

        """
        tflearn.is_training(False, session)
        n_test_samples = len(get_dict_first_element(feed_dict))
        batches = make_batches(n_test_samples, batch_size)
        index_array = np.arange(n_test_samples)
        avg = 0.0
        for i, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            feed_batch = {}
            for key in feed_dict:
                # Make batch for multi-dimensional data
                if np.ndim(feed_dict[key]) > 0:
                    feed_batch[key] = slice_array(feed_dict[key], batch_ids)
                else:
                    feed_batch[key] = feed_dict[key]
            avg += session.run(op_to_evaluate, feed_batch) / len(batches)
        return avg
