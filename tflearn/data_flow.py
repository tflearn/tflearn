# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import threading
try:
    # Python 2
    import Queue as queue
except Exception:
    # Python 3
    import queue

import tensorflow as tf
from . import utils


class DataFlow(object):
    """ Data Flow.

    Base class for using real time pre-processing and controlling data flow.
    Supports pipelining for faster computation.

    Arguments:
        coord: `Coordinator`. A Tensorflow coordinator.
        num_threads: `int`. Total number of simultaneous threads to process data.
        max_queue: `int`. Maximum number of data stored in a queue.
        shuffle: `bool`. If True, data will be shuffle.
        continuous: `bool`. If True, when an epoch is over, same data will be
            feeded again.
        ensure_data_order: `bool`. Ensure that data order is keeped when using
            'next' to retrieve data (Processing will be slower).
        dprep_dict: dict. Optional data pre-processing parameter for performing
            real time data pre-processing. Keys must be placeholders and values
            `DataPreprocessing` subclass object.
        daug_dict: dict. Optional data augmentation parameter for performing
            real time data augmentation. Keys must be placeholders and values
            `DataAugmentation` subclass object.

    """

    def __init__(self, coord, num_threads=8, max_queue=32, shuffle=False,
                 continuous=False, ensure_data_order=False,
                 dprep_dict=None, daug_dict=None):
        self.coord = coord
        self.num_threads = num_threads
        self.max_queue = max_queue
        self.shuffle = shuffle
        self.continuous = continuous
        if ensure_data_order:
            self.num_threads = 1
            self.max_queue = 1
        self.dprep_dict = dprep_dict
        self.daug_dict = daug_dict
        self.interrupted = False


class FeedDictFlow(DataFlow):

    """ FeedDictFlow.

    Generate a stream of batches from a dataset. It uses two queues, one for
    generating batch of data ids, and the other one to load data and apply pre
    processing. If continuous is `True`, data flow will never ends until `stop`
    is invoked, or `coord` interrupt threads.

    Arguments:
        feed_dict: `dict`. A TensorFlow formatted feed dict (with placeholders
            as keys and data as values).
        coord: `Coordinator`. A Tensorflow coordinator.
        num_threads: `int`. Total number of simultaneous threads to process data.
        max_queue: `int`. Maximum number of data stored in a queue.
        shuffle: `bool`. If True, data will be shuffle.
        continuous: `bool`. If True, when an epoch is over, same data will be
            feeded again.
        ensure_data_order: `bool`. Ensure that data order is keeped when using
            'next' to retrieve data (Processing will be slower).
        dprep_dict: dict. Optional data pre-processing parameter for performing
            real time data pre-processing. Keys must be placeholders and values
            `DataPreprocessing` subclass object.
        daug_dict: dict. Optional data augmentation parameter for performing
            real time data augmentation. Keys must be placeholders and values
            `DataAugmentation` subclass object.
        index_array: `list`. An optional list of index to be used instead of
            using the whole dataset indexes (Useful for validation split).

    """

    def __init__(self, feed_dict, coord, batch_size=128, num_threads=8,
                 max_queue=32, shuffle=False, continuous=False,
                 ensure_data_order=False, dprep_dict=None, daug_dict=None,
                 index_array=None):
        super(FeedDictFlow, self).__init__(coord, num_threads, max_queue,
                                           shuffle, continuous,
                                           ensure_data_order,
                                           dprep_dict,
                                           daug_dict)
        self.feed_dict = feed_dict
        self.batch_size = batch_size
        self.n_samples = len(utils.get_dict_first_element(feed_dict))

        # Queue holding batch ids
        self.batch_ids_queue = queue.Queue(self.max_queue)
        # Queue holding data ready feed dicts
        self.feed_dict_queue = queue.Queue(self.max_queue)

        # Create samples index array
        self.index_array = np.arange(self.n_samples)
        if index_array is not None:
            self.index_array = index_array
            self.n_samples = len(index_array)

        # Create batches
        self.batches = self.make_batches()
        self.reset_batches()

        # Data Recording
        self.data_status = DataFlowStatus(self.batch_size, self.n_samples)

    def next(self, timeout=None):
        """ next.

        Get the next feed dict.

        Returns:
            A TensorFlow feed dict, or 'False' if it has no more data.

        """
        self.data_status.update()
        return self.feed_dict_queue.get(timeout=timeout)

    def start(self, reset_status=True):
        """ start.

        Arguments:
            reset_status: `bool`. If True, `DataStatus` will be reset.

        Returns:

        """
        # Start to process data and fill queues
        self.clear_queues()
        self.interrupted = False
        # Reset Data Status
        if reset_status:
            self.data_status.reset()
        # Only a single thread needed for batches ids
        bi_threads = [threading.Thread(target=self.fill_batch_ids_queue)]
        # Multiple threads available for feed batch pre-processing
        fd_threads = [threading.Thread(target=self.fill_feed_dict_queue)
                      for i in range(self.num_threads)]
        self.threads = bi_threads + fd_threads
        for t in self.threads:
            t.daemon = True
            t.start()

    def stop(self):
        """ stop.

        Stop the queue from creating more feed_dict.

        """
        # Send stop signal to processing queue
        for i in range(self.num_threads):
            self.batch_ids_queue.put(False)
        # Launch a Thread to wait for processing scripts to finish
        t = threading.Thread(target=self.wait_for_threads)
        t.daemon = True
        t.start()

    def reset(self):
        """ reset.

        Reset batch index.
        """
        self.batch_index = -1

    def interrupt(self):
        # Send interruption signal to processing queue
        self.interrupted = True
        self.clear_queues()

    def fill_feed_dict_queue(self):
        while not self.coord.should_stop() and not self.interrupted:
            batch_ids = self.batch_ids_queue.get()
            if batch_ids is False:
                break
            data = self.retrieve_data(batch_ids)
            # Apply augmentation according to daug dict
            if self.daug_dict:
                for k in self.daug_dict:
                    data[k] = self.daug_dict[k].apply(data[k])
            # Apply preprocessing according to dprep dict
            if self.dprep_dict:
                for k in self.dprep_dict:
                    data[k] = self.dprep_dict[k].apply(data[k])
            #all prepped, put the data into the queue
            self.feed_dict_queue.put(data)

    def fill_batch_ids_queue(self):
        while not self.coord.should_stop() and not self.interrupted:
            ids = self.next_batch_ids()
            if ids is False:
                break
            self.batch_ids_queue.put(ids)

    def next_batch_ids(self):

        self.batch_index += 1
        if self.batch_index == len(self.batches):
            if not self.continuous:
                self.stop()
                return False
            self.reset_batches()

        batch_start, batch_end = self.batches[self.batch_index]
        return self.index_array[batch_start:batch_end]

    def retrieve_data(self, batch_ids):
        feed_batch = {}
        for key in self.feed_dict:
            feed_batch[key] = \
                    utils.slice_array(self.feed_dict[key], batch_ids)
        return feed_batch

    def reset_batches(self):
        if self.shuffle:
            self.shuffle_samples()
            # Generate new batches
            self.batches = self.make_batches()
        self.batch_index = -1

    def make_batches(self):
        return utils.make_batches(self.n_samples, self.batch_size)

    def shuffle_samples(self):
        np.random.shuffle(self.index_array)

    def wait_for_threads(self):
        # Wait for threads to finish computation (max 120s)
        self.coord.join(self.threads)
        # Send end signal to indicate no more data in feed queue
        self.feed_dict_queue.put(False)

    def clear_queues(self):
        """ clear_queues.

        Clear queues.

        """
        while not self.feed_dict_queue.empty():
            self.feed_dict_queue.get()
        while not self.batch_ids_queue.empty():
            self.batch_ids_queue.get()


class TFRecordsFlow(DataFlow):

    def __init__(self, coord):
        super(TFRecordsFlow, self).__init__(coord)
        raise NotImplementedError


class DataFlowStatus(object):
    """ Data Flow Status

    Simple class for recording how many data have been processed.

    """

    def __init__(self, batch_size, n_samples):
        self.step = 0
        self.epoch = 0
        self.current_iter = 0
        self.batch_size = batch_size
        self.n_samples = n_samples

    def update(self):
        self.step += 1
        self.current_iter = min(self.step * self.batch_size, self.n_samples)

        if self.current_iter == self.n_samples:
            self.epoch += 1
            self.step = 0

    def reset(self):
        self.step = 0
        self.epoch = 0


class ArrayFlow(object):
    """ ArrayFlow.

    Convert array samples to tensors and store them in a queue.

    Arguments:
        X: `array`. The features data array.
        Y: `array`. The targets data array.
        multi_inputs: `bool`. Set to True if X has multiple input sources (i.e.
            X is a list of arrays).
        batch_size: `int`. The batch size.
        shuffle: `bool`. If True, data will be shuffled.

    Returns:
        The `X` and `Y` data tensors or a list(`X`) and `Y` data tensors if
        multi_inputs is True.

    """
    def __init__(self, X, Y, multi_inputs=False, batch_size=32, shuffle=True):
        # Handle multiple inputs
        if not multi_inputs:
            X = [X]
        X = [np.array(x) for x in X]
        self.X = X
        self.Xlen = len(X[0])
        Y = np.array(Y)
        self.Y = Y
        # Create X placeholders
        self.tensorX = [tf.placeholder(
                dtype=tf.float32,
                shape=[None] + list(utils.get_incoming_shape(x)[1:]))
            for x in X]
        # Create Y placeholders
        self.tensorY = tf.placeholder(
            dtype=tf.float32,
            shape=[None] + list(utils.get_incoming_shape(Y)[1:]))
        # FIFO Queue for feeding data
        self.queue = tf.FIFOQueue(
            dtypes=[x.dtype for x in self.tensorX] + [self.tensorY.dtype],
            capacity=batch_size * 8)
        self.enqueue_op = self.queue.enqueue(self.tensorX + [self.tensorY])
        self.batch_size = batch_size
        self.multi_inputs = multi_inputs
        self.shuffle = shuffle

    def iterate(self, X, Y, batch_size):
        while True:
            # Shuffle array if specified
            if self.shuffle:
                idxs = np.arange(0, len(X[0]))
                np.random.shuffle(idxs)
                X = [x[idxs] for x in X]
                Y = Y[idxs]
            # Split array by batch
            for batch_idx in range(0, self.Xlen, batch_size):
                batchX = [x[batch_idx:batch_idx + batch_size] for x in X]
                batchY = Y[batch_idx:batch_idx + batch_size]
                yield batchX, batchY

    def get(self):
        # get data from the queue
        dequeue = self.queue.dequeue()
        if self.multi_inputs:
            return dequeue[:-1], dequeue[-1]
        else:
            return dequeue[0], dequeue[1]

    def launch_threads(self, session, num_threads=1):
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=self.thread_main, args=(session,))
            t.daemon = True
            t.start()
            threads.append(t)
        return threads

    def thread_main(self, sess):
        for dataX, dataY in self.iterate(self.X, self.Y, self.batch_size):
            feed_dict = {self.tensorY: dataY}
            for i, x in enumerate(self.tensorX):
                feed_dict[x] = dataX[i]
            sess.run(self.enqueue_op, feed_dict=feed_dict)
