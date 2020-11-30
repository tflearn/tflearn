from __future__ import division, print_function, absolute_import

import tensorflow.compat.v1 as tf
from tensorflow.core.framework import summary_pb2

from .utils import format_scope_name


def monitor_activation(tensor):
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, tensor)


def get_summary(stype, tag, value=None, collection_key=None,
                break_if_exists=False):
    """ get_summary.

    Create or retrieve a summary. It keep tracks of all graph summaries
    through summary_tags collection. If a summary tags already exists,
    it will return that summary tensor or raise an error (according to
    'break_if_exists').

    Arguments:
        stype: `str`. Summary type: 'histogram', 'scalar' or 'image'.
        tag: `str`. The summary tag (name).
        value: `Tensor`. The summary initialization value. Default: None.
        collection_key: `str`. If specified, the created summary will be
            added to that collection (optional).
        break_if_exists: `bool`. If True, if a summary with same tag already
            exists, it will raise an exception (instead of returning that
            existing summary).

    Returns:
        The summary `Tensor`.

    """
    summ = next((item for item in tf.get_collection("summary_tags") if
                 item["tag"] == tag), None)

    if not summ:
        if value is None:
            raise Exception("Summary doesn't exist, a value must be "
                            "specified to initialize it.")
        if stype == "histogram":
            summ = tf.summary.histogram(tag, value)
        elif stype == "scalar":
            summ = tf.summary.scalar(tag, value)
        elif stype == "image":
            pass  # TODO: create summary
        else:
            raise ValueError("Unknow summary type: '" + str(stype) + "'")
        tf.add_to_collection("summary_tags", {"tag": tag, "tensor": summ})
        if collection_key:
            tf.add_to_collection(collection_key, summ)
    elif break_if_exists:
        raise ValueError("Error: Summary tag already exists! (to ignore this "
                         "error, set add_summary() parameter 'break_if_exists'"
                         " to False)")
    else:
        summ = summ["tensor"]

    return summ


def add_activations_summary(activation_ops, name_prefix="", name_suffix="",
                            collection_key=None):
    """ add_activations_summary.

    Add histogram summary for given activations.

    Arguments:
        activation_ops: A list of `Tensor`. The activations to summarize.
        name_prefix: `str`. A prefix to add to summary scope.
        name_suffix: `str`. A suffix to add to summary scope.
        collection_key: `str`. A collection to store the summaries.

    Returns:
        The list of created activation summaries.
    """

    summ = []
    for ao in activation_ops:
        ao_name = ao.op.name
        summ_name = format_scope_name(ao_name, name_prefix,
                                      "Activations/" + name_suffix)
        summ_exists = summary_exists(summ_name)
        if summ_exists is not None:
            tf.add_to_collection(collection_key, summ_exists)
        else:
            get_summary("histogram", summ_name, ao, collection_key)

        summ_name = format_scope_name(ao_name, name_prefix,
                                      "Sparsity/" + name_suffix)
        summ_exists = summary_exists(summ_name)
        if summ_exists is not None:
            tf.add_to_collection(collection_key, summ_exists)
            summ.append(summ_exists)
        else:
            summ.append(get_summary("scalar", summ_name,
                                    tf.nn.zero_fraction(ao), collection_key))
    return summ


def add_gradients_summary(grads, name_prefix="", name_suffix="",
                          collection_key=None):
    """ add_gradients_summary.

    Add histogram summary for given gradients.

    Arguments:
        grads: A list of `Tensor`. The gradients to summarize.
        name_prefix: `str`. A prefix to add to summary scope.
        name_suffix: `str`. A suffix to add to summary scope.
        collection_key: `str`. A collection to store the summaries.

    Returns:
        The list of created gradient summaries.

    """

    # Add histograms for gradients.
    summ = []
    for grad, var in grads:
        if grad is not None:
            summ_name = format_scope_name(var.op.name, name_prefix,
                                          "Gradients/" + name_suffix)
            summ_exists = summary_exists(summ_name)
            if summ_exists is not None:
                tf.add_to_collection(collection_key, summ_exists)
                summ.append(summ_exists)
            else:
                summ.append(get_summary("histogram", summ_name, grad,
                                        collection_key))
    return summ


def add_trainable_vars_summary(variables, name_prefix="", name_suffix="",
                               collection_key=None):
    """ add_trainable_vars_summary.

    Add histogram summary for given variables weights.

    Arguments:
        variables: A list of `Variable`. The variables to summarize.
        name_prefix: `str`. A prefix to add to summary scope.
        name_suffix: `str`. A suffix to add to summary scope.
        collection_key: `str`. A collection to store the summaries.

    Returns:
        The list of created weights summaries.

    """

    # Add histograms for trainable variables.
    summ = []
    for var in variables:
        summ_name = format_scope_name(var.op.name, name_prefix, name_suffix)
        summ_exists = summary_exists(summ_name)
        if summ_exists is not None:
            tf.add_to_collection(collection_key, summ_exists)
            summ.append(summ_exists)
        else:
            summ.append(get_summary("histogram", summ_name, var, collection_key))
    return summ


def get_value_from_summary_string(tag, summary_str):
    """ get_value_from_summary_string.

    Retrieve a summary value from a summary string.

    Arguments:
        tag: `str`. The summary tag (name).
        summary_str: `str`. The summary string to look in.

    Returns:
        A `float`. The retrieved value.

    Raises:
        `Exception` if tag not found.

    """

    # Compatibility hotfix for the seq2seq example
    if tag == u'acc:0/':
        tag = u'acc_0/'

    # Fix for TF 0.12
    if tag[-1] == '/':
        tag = tag[:-1]
    summ = summary_pb2.Summary()
    summ.ParseFromString(summary_str)

    for row in summ.value:
        if row.tag.endswith(tag):
            return float(row.simple_value)

    raise ValueError("Tag: " + tag + " cannot be found in summaries list.")


def add_loss_summaries(total_loss, loss, regul_losses_collection_key,
                       name_prefix="", summaries_collection_key=None,
                       exp_moving_avg=0.9, ema_num_updates=None):
    """ add_loss_summaries.

    Add scalar summaries (raw and averages) for given losses.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Arguments:
        total_loss: `Tensor`. The total loss (Regression loss +
            regularization losses).
        loss: `Tensor`. Regression loss.
        name_prefix: `str`. A prefix to add to the summary name.
        regul_losses_collection_key: `str`. A collection name to retrieve
            regularization losses.
        exp_moving_avg: `float`. Exponential moving average.
        ema_num_updates: `int`. Step to be used with exp moving avg.

    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(decay=exp_moving_avg,
                                                      num_updates=ema_num_updates,
                                                      name='moving_avg')
    other_losses = tf.get_collection(regul_losses_collection_key)

    # Attach a scalar summmary to all individual losses and the total loss;
    # do the same for the averaged version of the losses.
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.

    # Only add total loss, if it has more than one loss...
    if len(other_losses) > 0 and total_loss is not None:
        loss_averages_op = loss_averages.apply(
            [total_loss] + [loss] + other_losses)
        summ_name = "Loss_var_loss/" + name_prefix
        get_summary("scalar", summ_name, loss_averages.average(total_loss),
                    summaries_collection_key)
        get_summary("scalar", summ_name + 'raw', total_loss,
                    summaries_collection_key)
    elif total_loss is not None:
        loss_averages_op = loss_averages.apply([loss] + other_losses)
    else:
        loss_averages_op = loss_averages.apply([loss])

    # For tflearn wrapper visibility
    summ_name = "Loss/" + name_prefix
    get_summary("scalar", summ_name, loss_averages.average(loss),
                summaries_collection_key)
    get_summary("scalar", summ_name + 'raw', loss, summaries_collection_key)

    for wdl in other_losses:
        # No prefix, we store every variable into their own scope
        summ_name = wdl.op.name
        get_summary("scalar", summ_name, loss_averages.average(wdl),
                    summaries_collection_key)
        get_summary("scalar", summ_name + 'raw', wdl,
                    summaries_collection_key)

    return loss_averages_op


def summary_exists(tag):
    """ summary_exists.

    Check if a summary exists.

    Arguments:
        tag: `str`. The summary name.

    Returns:
        A `bool`. Whether the summary exists or not.

    """
    return next(
        (item['tensor'] for item in tf.get_collection("summary_tags") if
         item["tag"] == tag), None)
