from __future__ import division, print_function, absolute_import

import tensorflow.compat.v1 as tf

from .variables import variable

# -------------------
# Basic Configuration
# -------------------


def init_graph(seed=None, log_device=False, num_cores=0, gpu_memory_fraction=0,
               soft_placement=True):
    """ init_graph.

    Initialize a graph with specific parameters.

    Arguments:
        seed: `int`. Set the graph random seed.
        log_device: `bool`. Log device placement or not.
        num_cores: Number of CPU cores to be used. Default: All.
        gpu_memory_fraction: A value between 0 and 1 that indicates what
            fraction of the available GPU memory to pre-allocate for each
            process. 1 means to pre-allocate all of the GPU memory,
            0.5 means the process allocates ~50% of the available GPU
            memory. Default: Use all GPU's available memory.
        soft_placement: `bool`. Whether soft placement is allowed. If true,
            an op will be placed on CPU if:
                1. there's no GPU implementation for the OP
                    or
                2. no GPU devices are known or registered
                    or
                3. need to co-locate with reftype input(s) which are from CPU.
    """
    if seed: tf.set_random_seed(seed)
    gs = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    config = tf.ConfigProto(log_device_placement=log_device,
                            inter_op_parallelism_threads=num_cores,
                            intra_op_parallelism_threads=num_cores,
                            gpu_options=gs,
                            allow_soft_placement=soft_placement)
    tf.add_to_collection(tf.GraphKeys.GRAPH_CONFIG, config)

    return config


# ------------------
# Training Mode
# ------------------
"""
Because some ops have different behavior at training and testing time (such
as dropout, or batch normalization), TFLearn implements a boolean variable
`is_training` to indicates the network if it is used for training or not.
This variable is stored in the tf.collection `is_training`, and is the
unique element of it.
The two operations to update that variable (set it to True or False),
are stored in another tf.collection `is_training_ops` with 2 elemens:
[set_training_mode_op, set_predicting_mode_op]. So invoking the first element
will enable training mode, while the second one will enable predicting mode.
"""


def is_training(is_training=False,  session=None):
    """ is_training.

    Set the graph training mode.

    This is meant to be used to control ops that have different output at
    training and testing time., such as dropout or batch normalization,

    Examples:
        ```
        >> # Retrieve variable responsible for managing training mode
        >> training_mode = tflearn.get_training_mode()
        >> # Define a conditional op
        >> my_conditional_op = tf.cond(training_mode, if_yes_op, if_no_op)
        >> # Set training mode to True
        >> tflearn.is_training(True)
        >> session.run(my_conditional_op)
        if_yes_op
        >> # Set training mode to False
        >> tflearn.is_training(False)
        >> session.run(my_conditional_op)
        if_no_op
        ```

    Returns:
        A `bool`, True if training, False else.

    """
    if not session:
        session = tf.get_default_session()
    init_training_mode()
    if is_training:
        tf.get_collection('is_training_ops')[0].eval(session=session)
    else:
        tf.get_collection('is_training_ops')[1].eval(session=session)


def get_training_mode():
    """ get_training_mode.

    Returns variable in-use to set training mode.

    Returns:
        A `Variable`, the training mode holder.

    """
    init_training_mode()
    coll = tf.get_collection('is_training')
    return coll[0]


def init_training_mode():
    """  init_training_mode.

    Creates `is_training` variable and its ops if they haven't be created
    yet. This op is required if you are using layers such as dropout or
    batch normalization independently of TFLearn models (DNN or Trainer class).

    """
    # 'is_training' collection stores the training mode variable
    coll = tf.get_collection('is_training')
    if len(coll) == 0:
        tr_var = variable(
            "is_training", dtype=tf.bool, shape=[],
            initializer=tf.constant_initializer(False),
            trainable=False)
        tf.add_to_collection('is_training', tr_var)
        # 'is_training_ops' stores the ops to update training mode variable
        a = tf.assign(tr_var, True)
        b = tf.assign(tr_var, False)
        tf.add_to_collection('is_training_ops', a)
        tf.add_to_collection('is_training_ops', b)


_FLOATX = tf.float32
_EPSILON = 1e-10

