# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import six
import string
import random
try:
    import h5py
    H5PY_SUPPORTED = True
except Exception as e:
    print("hdf5 not supported (please install/reinstall h5py)")
    H5PY_SUPPORTED = False
import numpy as np
import tensorflow as tf

import tflearn.variables as vs


def get_from_module(identifier, module_params, module_name, instantiate=False, kwargs=None):
    if isinstance(identifier, six.string_types):
        res = module_params.get(identifier)
        if not res:
            res = module_params.get(identifier.lower())
            if not res:
                raise Exception('Invalid ' + str(module_name) + ': ' + str(identifier))
        if instantiate and not kwargs:
            return res()
        elif instantiate and kwargs:
            return res(**kwargs)
        else:
            return res
    return identifier


# ------------------
#  Ops utils
# ------------------


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")


def get_tensor_parents_placeholders(tensor):
    """ Get all placeholders that is depending the given tensor. """
    placeholders_list = []
    if tensor.op.type == 'Placeholder':
        placeholders_list.append(tensor)
    if tensor.op:
        for t in tensor.op.inputs:
            if not 'read:0' in t.name:
                placeholders_list += get_tensor_parents_placeholders(t)
    return list(set(placeholders_list))


def get_tensor_parents(tensor):
    """ Get all calculation and data parent tensors (Not read). """
    parents_list = []
    parents_list.append(tensor)
    if tensor.op:
        for t in tensor.op.inputs:
            if not 'read:0' in t.name:
                parents_list += get_tensor_parents(t)
    return parents_list


def get_all_tensor_parents(tensor):
    """ Get all parents tensors. """
    parents_list = []
    parents_list.append(tensor)
    if tensor.op:
        for t in tensor.op.inputs:
            parents_list += get_tensor_parents(t)
    return list(set(parents_list))


def get_tensor_children_placeholders(tensor):
    """ Get all placeholders that is depending the given tensor. """
    placeholders_list = []
    if tensor.op.type == 'Placeholder':
        placeholders_list.append(tensor)
    if tensor.op:
        for t in tensor.op.outputs:
            if not 'read:0' in t.name:
                placeholders_list += get_tensor_children_placeholders(t)
    return list(set(placeholders_list))


def get_tensor_children(tensor):
    """ Get all calculation and data parent tensors (Not read). """
    children_list = []
    children_list.append(tensor)
    if tensor.op:
        for t in tensor.op.outputs:
            if not 'read:0' in t.name:
                children_list += get_tensor_children(t)
    return list(set(children_list))


def get_all_tensor_children(tensor):
    """ Get all parents tensors. """
    children_list = []
    children_list.append(tensor)
    if tensor.op:
        for t in tensor.op.outputs:
            children_list += get_all_tensor_children(t)
    return list(set(children_list))

# ------------------
#  Other utils
# ------------------


def to_list(data):
    if data is None: return None
    if type(data) in [tuple, list]:
        return data
    return [data]


def standarize_data(data):
    if data is None: return None
    if type(data) in [tuple, list]:
        return [np.asarray(x) for x in data]
    if type(data) is dict:
        return data
    return [np.asarray(data)]


def standarize_dict(d):
    for key in d:
        if isinstance(d[key], list):
            d[key] = np.asarray(d[key])


def del_duplicated(l):
    res = []
    for e in l:
        if e not in res:
            res.append(e)
    return res
    #return list(np.unique(np.array(l)))


def make_batches(samples_size, batch_size):
    nb_batch = int(np.ceil(samples_size/float(batch_size)))
    return [(i*batch_size, min(samples_size, (i+1)*batch_size)) for i in range(0, nb_batch)]


def slice_array(X, start=None, stop=None):
    if type(X) == list:
        if hasattr(start, '__len__'):
            return [x[start] for x in X]
        else:
            return [x[start:stop] for x in X]
    if H5PY_SUPPORTED:
        if type(X) == h5py.Dataset:
            return [X[i] for i in start]
    if hasattr(start, '__len__'):
        return X[start]
    else:
        return X[start:stop]


def get_dict_first_element(input_dict):
    for key in input_dict:
        return input_dict[key]


def get_tensor_with_parent_name(tensor):
    """ Get a tensor name with its parent tensor's name as prefix. """
    tensor_name = tensor.name
    if tensor.op.inputs[0].name is not None:
        return tensor.op.inputs[0].name + "_" + tensor_name
    return tensor_name


def format_scope_name(scope_name, prefix, suffix):
    """ Add a predix and a suffix to a scope name. """
    if prefix is not "":
        if not prefix[-1] == "/":
            prefix += "/"
    if suffix is not "":
        if not suffix[0] == "/":
            suffix = "/" + suffix
    return prefix + scope_name + suffix


def check_scope_path(scope_name):
    scope_name = scope_name.replace("//", "/")
    return scope_name


def feed_dict_builder(X, Y, net_inputs, net_targets):
    """ Format provided data to a dictionary format compatible with
    Tensorflow data feeding. It match all X and Y data provided with
    net_inputs and net_targets provided placeholders. In case of inputs
    data list, matching is made respectively.

    Examples:
        ```python
        # Building feed dictionary
        >> feed_dict = feed_dict_builder(X, Y, input1, output1)
        >> {input1: X, output1: Y}

        >> feed_dict = feed_dict_builder({input1: X}, Y, input1, output1)
        >> {input1: X, output1: Y}

        >> feed_dict = feed_dict_builder([X1, X2], Y, [in1, in2], out1)
        >> {in1: X1, in2: X2, output1: Y}

        # For validation split:
        >> val_feed_dict = feed_dict_builder(0.1, 0.1, input1, output1)
        >> {input1: 0.1, output1: 0.1}
        ```

    Arguments:
        X: `array` or `dict`. The input data.
        Y: `array`, `dict` or `float`. The targets (labels).
        net_inputs: `list`. The network data inputs `Placeholders`.
        net_targets: `list`. The network targets `Placeholders`.

    Returns:
        `dict`. A Tensorflow-ready dictionary to feed data.

    Raises:
        Exception if X and net_inputs or Y and net_targets list length doesn't
        match.

    """

    feed_dict = {}

    if not (is_none(X) or is_none(net_inputs)):
        # If input data are not a dict, we match them by creation order
        if not isinstance(X, dict):
            # If validation split, copy that value to the whole placeholders
            if isinstance(X, float): X = [X for _i in net_inputs]
            elif len(net_inputs) > 1:
                if np.ndim(X) < 2:
                    raise ValueError("Multiple inputs but only one data "
                                     "feeded. Please verify number of inputs "
                                     "and data provided match.")
                elif len(X) != len(net_inputs):
                    raise Exception(str(len(X)) + " inputs feeded, "
                                    "but expected: " + str(len(net_inputs)) +
                                    ". If you are using notebooks, please "
                                    "make sure that you didn't run graph "
                                    "construction cell multiple time, "
                                    "or try to enclose your graph within "
                                    "`with tf.Graph().as_default():`")
            else:
                X = [X]
            for i, x in enumerate(X):
                feed_dict[net_inputs[i]] = x
        else:
            # If a dict is provided
            for key, val in X.items():
                # Do nothing if dict already fits {placeholder: data} template
                if isinstance(key, tf.Tensor):
                    continue
                else: # Else retrieve placeholder with its name
                    feed_dict[vs.get_inputs_placeholder_by_name(key)] = val

    if not (is_none(Y) or is_none(net_targets)):
        if not isinstance(Y, dict):
            # Verify network has targets
            if len(net_targets) == 0:
                return feed_dict
            # If validation split, copy that value to every target placeholder.
            if isinstance(Y, float):
                Y = [Y for _t in net_targets]
            elif len(net_targets) > 1:
                if np.ndim(Y) < 2:
                    raise ValueError("Multiple outputs but only one data "
                                     "feeded. Please verify number of outputs "
                                     "and data provided match.")
                elif len(Y) != len(net_targets):
                    raise Exception(str(len(Y)) + " outputs feeded, "
                                    "but expected: " + str(len(net_targets)))
            else: Y = [Y]
            for i, y in enumerate(Y):
                feed_dict[net_targets[i]] = y
        else:
            # If a dict is provided
            for key, val in Y.items():
                # Do nothing if dict already fits {placeholder: data} template
                if isinstance(key, tf.Tensor):
                    continue
                else: # Else retrieve placeholder with its name
                    feed_dict[vs.get_targets_placeholder_by_name(key)] = val

    return feed_dict


def is_none(val):
    # Check if a value is None or not, required because `np.array is None` is
    # ambiguous and raise Exception.
    if type(val) is np.array:
        return False
    else:
        return val is None


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def check_dir_name(dir_path):
    if isinstance(dir_path, str):
        if len(dir_path) > 0:
            if dir_path[-1] != "/":
                dir_path += "/"
        return dir_path
    else:
        raise ValueError("Incorrect string format for directory path.")


# ----------------------------
# Parameter formatting helpers
# ----------------------------


# Auto format kernel
def autoformat_kernel_2d(strides):
    if type(strides) is int:
        return [1, strides, strides, 1]
    elif type(strides) in [tuple, list]:
        if len(strides) == 2:
            return [1, strides[0], strides[1], 1]
        elif len(strides) == 4:
            return [strides[0], strides[1], strides[2], strides[3]]
        else:
            if type(strides) in [tuple, list]:
                raise Exception("strides length error: " + str(len(strides))
                                + ", only a length of 2 or 4 is supported.")
            else:
                raise Exception("strides format error: " + str(type(strides)))


# Auto format filter size
# Output shape: (rows, cols, input_depth, out_depth)
def autoformat_filter_conv2d(fsize, in_depth, out_depth):
    if type(fsize) is int:
        return [fsize, fsize, in_depth, out_depth]
    elif type(fsize) in [tuple, list]:
        if len(fsize) == 2:
            return [fsize[0], fsize[1], in_depth, out_depth]
        else:
            if type(fsize) in [tuple, list]:
                raise Exception("filter length error: " + str(len(fsize))
                                + ", only a length of 2 or 4 is supported.")
            else:
                raise Exception("filter format error: " + str(type(fsize)))


# Auto format padding
def autoformat_padding(padding):
    if padding in ['same', 'SAME', 'valid', 'VALID']:
        return str.upper(padding)
    else:
        raise Exception("Unknow padding! Accepted values: 'same', 'valid'.")
