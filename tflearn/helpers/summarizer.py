from __future__ import division, print_function, absolute_import

import tensorflow as tf
from .. import summaries

# Fix for TF 0.12
try:
    tf012 = True
    merge_summary = tf.summary.merge
except Exception:
    tf012 = False
    merge_summary = tf.merge_summary

"""
Summarizer contains some useful functions to help summarize variables,
activations etc... in Tensorboard.
"""


def summarize_all(train_vars, grads, activations,
                  summary_collection="tflearn_summ"):
    summarize_variables(train_vars, summary_collection)
    summarize_gradients(grads, summary_collection)
    return summarize_activations(activations, summary_collection)


def summarize_variables(train_vars=None, summary_collection="tflearn_summ"):
    """ summarize_variables.

    Arguemnts:
        train_vars: list of `Variable`. The variable weights to monitor.
        summary_collection: A collection to add this summary to and
            also used for returning a merged summary over all its elements.
            Default: 'tflearn_summ'.

    Returns:
        `Tensor`. Merge of all summary in 'summary_collection'

    """
    if not train_vars: train_vars = tf.trainable_variables()
    summaries.add_trainable_vars_summary(train_vars, "", "", summary_collection)
    return merge_summary(tf.get_collection(summary_collection))


def summarize_activations(activations, summary_collection="tflearn_summ"):
    """ summarize_activations.

    Arguemnts:
        activations: list of `Tensor`. The activations to monitor.
        summary_collection: A collection to add this summary to and
            also used for returning a merged summary over all its elements.
            Default: 'tflearn_summ'.

    Returns:
        `Tensor`. Merge of all summary in 'summary_collection'

    """
    summaries.add_activations_summary(activations, "", "", summary_collection)
    return merge_summary(tf.get_collection(summary_collection))


def summarize_gradients(grads, summary_collection="tflearn_summ"):
    """ summarize_gradients.

    Arguemnts:
        grads: list of `Tensor`. The gradients to monitor.
        summary_collection: A collection to add this summary to and
            also used for returning a merged summary over all its elements.
            Default: 'tflearn_summ'.

    Returns:
        `Tensor`. Merge of all summary in 'summary_collection'

    """
    summaries.add_gradients_summary(grads, "", "", summary_collection)
    return merge_summary(tf.get_collection(summary_collection))


def summarize(value, type, name, summary_collection="tflearn_summ"):
    """ summarize.

    A custom summarization op.

    Arguemnts:
        value: `Tensor`. The tensor value to monitor.
        type: `str` among 'histogram', 'scalar'. The data monitoring type.
        name: `str`. A name for this summary.
        summary_collection: A collection to add this summary to and
            also used for returning a merged summary over all its elements.
            Default: 'tflearn_summ'.

    Returns:
        `Tensor`. Merge of all summary in 'summary_collection'.

    """
    if tf012:
        name = name.replace(':', '_')
    summaries.get_summary(type, name, value, summary_collection)
    return merge_summary(tf.get_collection(summary_collection))
