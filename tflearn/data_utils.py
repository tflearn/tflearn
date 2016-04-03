# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import os
import random
import numpy as np
from PIL import Image
import pickle

"""
Preprocessing provides some useful functions to preprocess data before
training, such as pictures dataset building, sequence padding, etc...

Note: Those preprocessing functions are only meant to be directly applied to
data, they are not meant to be use with Tensors or Layers.
"""


# ------------------------------
# TARGETS (LABELS) PREPROCESSING
# ------------------------------


def to_categorical(y, nb_classes):
    """ to_categorical.

    Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.

    Arguments:
        y: `array`. Class vector to convert.
        nb_classes: `int`. Total number of classes.

    """
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


# -----------------------
# SEQUENCES PREPROCESSING
# -----------------------


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre',
                  truncating='pre', value=0.):
    """ pad_sequences.

    Pad each sequence to the same length: the length of the longest sequence.
    If maxlen is provided, any sequence longer than maxlen is truncated to
    maxlen. Truncation happens off either the beginning (default) or the
    end of the sequence. Supports post-padding and pre-padding (default).

    Arguments:
        sequences: list of lists where each element is a sequence.
        maxlen: int, maximum length.
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    Returns:
        x: `numpy array` with dimensions (number_of_sequences, maxlen)

    Credits: From Keras `pad_sequences` function.
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def string_to_semi_redundant_sequences(string, seq_maxlen=25, redun_step=3):
    """ string_to_semi_redundant_sequences.

    Vectorize a string and returns parsed sequences and targets, along with
    the associated dictionary.

    Arguments:
        string: `str`. Lower-case text from input text file.
        seq_maxlen: `int`. Maximum length of a sequence. Default: 25.
        redun_step: `int`. Redundancy step. Default: 3.

    Returns:
        `tuple`: (inputs, targets, dictionary)
    """

    print("Vectorizing text...")
    chars = set(string)
    char_idx = {c: i for i, c in enumerate(chars)}

    sequences = []
    next_chars = []
    for i in range(0, len(string) - seq_maxlen, redun_step):
        sequences.append(string[i: i + seq_maxlen])
        next_chars.append(string[i + seq_maxlen])

    X = np.zeros((len(sequences), seq_maxlen, len(chars)), dtype=np.bool)
    Y = np.zeros((len(sequences), len(chars)), dtype=np.bool)
    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            X[i, t, char_idx[char]] = 1
        Y[i, char_idx[next_chars[i]]] = 1

    print("Text total length: " + str(len(string)))
    print("Distinct chars: " + str(len(chars)))
    print("Total sequences: " + str(len(sequences)))

    return X, Y, char_idx


def textfile_to_semi_redundant_sequences(path, seq_maxlen=25, redun_step=3,
                                         to_lower_case=False):
    """ Vectorize Text file """
    text = open(path).read()
    if to_lower_case:
        text = text.lower()
    return string_to_semi_redundant_sequences(text, seq_maxlen, redun_step)


def random_sequence_from_string(string, seq_maxlen):
    rand_index = random.randint(0, len(string) - seq_maxlen - 1)
    return string[rand_index: rand_index + seq_maxlen]


def random_sequence_from_textfile(path, seq_maxlen):
    text = open(path).read()
    return random_sequence_from_string(text, seq_maxlen)


# --------------------
# IMAGES PREPROCESSING
# --------------------


def load_image(in_image):
    """ Load an image, returns PIL.Image. """
    img = Image.open(in_image)
    return img


def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    """ Resize an image.

    Arguments:
        in_image: `PIL.Image`. The image to resize.
        new_width: `int`. The image new width.
        new_height: `int`. The image new height.
        out_image: `str`. If specified, save the image to the given path.
        resize_mode: `PIL.Image.mode`. The resizing mode.

    Returns:
        `PIL.Image`. The resize image.

    """
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img


def convert_color(in_image, mode):
    """ Convert image color with provided `mode`. """
    return in_image.convert(mode)


def pil_to_nparray(pil_image):
    """ Convert a PIL.Image to numpy array. """
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")


def image_dirs_to_samples(directory, resize=None, convert_gray=None,
                          filetypes=None):
    print("Starting to parse images...")
    if filetypes:
        if filetypes not in [list, tuple]: filetypes = list(filetypes)
    samples, targets = directory_to_samples(directory, flags=filetypes)
    for i, s in enumerate(samples):
        samples[i] = load_image(s)
        if resize:
            samples[i] = resize_image(samples[i], resize[0], resize[1])
        if convert_gray:
            samples[i] = convert_color(samples[i], 'L')
        samples[i] = pil_to_nparray(samples[i])
        samples[i] /= 255.
    print("Parsing Done!")
    return samples, targets


def build_image_dataset_from_dir(directory,
                                 dataset_file="my_tflearn_dataset.pkl",
                                 resize=None, convert_gray=None,
                                 filetypes=None, shuffle_data=False,
                                 categorical_Y=False):
    try:
        X, Y = pickle.load(open(dataset_file, 'rb'))
    except Exception:
        X, Y = image_dirs_to_samples(directory, resize, convert_gray, filetypes)
        if categorical_Y:
            Y = to_categorical(Y, np.max(Y) + 1) # First class is '0'
        if shuffle_data:
            X, Y = shuffle(X, Y)
        pickle.dump((X, Y), open(dataset_file, 'wb'))
    return X, Y


# ------------------
# DATA PREPROCESSING
# ------------------


def shuffle(*arrs):
    """ shuffle.

    Shuffle given arrays at unison, along first axis.

    Arguments:
        *arrs: Each array to shuffle at unison as a parameter.

    Returns:
        Tuple of shuffled arrays.

    """
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)


def samplewise_zero_center(X):
    """ samplewise_zero_center.

    Zero center each sample by subtracting it by its mean.

    Arguments:
        X: `array`. The batch of samples to center.

    Returns:
        A numpy array with same shape as input.

    """
    for i in range(len(X)):
        X[i] -= np.mean(X[i], axis=1, keepdims=True)
    return X


def samplewise_std_normalization(X):
    """ samplewise_std_normalization.

    Scale each sample with its standard deviation.

    Arguments:
        X: `array`. The batch of samples to scale.

    Returns:
        A numpy array with same shape as input.

    """
    for i in range(len(X)):
        X[i] /= np.std(X[i], axis=1, keepdims=True)
    return X


def featurewise_zero_center(X, mean=None):
    """ featurewise_zero_center.

    Zero center every sample with specified mean. If not specified, the mean
    is evaluated over all samples.

    Arguments:
        X: `array`. The batch of samples to center.
        mean: `float`. The mean to use for zero centering. If not specified, it
            will be evaluated on provided data.

    Returns:
        A numpy array with same shape as input. Or a tuple (array, mean) if no
        mean value was specified.

    """
    if mean is None:
        mean = np.mean(X, axis=0)
        return X - mean, mean
    else:
        return X - mean


def featurewise_std_normalization(X, std=None):
    """ featurewise_std_normalization.

    Scale each sample by the specified standard deviation. If no std
    specified, std is evaluated over all samples data.

    Arguments:
        X: `array`. The batch of samples to scale.
        std: `float`. The std to use for scaling data. If not specified, it
            will be evaluated over the provided data.

    Returns:
        A numpy array with same shape as input. Or a tuple (array, std) if no
        std value was specified.

    """
    if std is None:
        std = np.std(X, axis=0)
        return X / std, std
    else:
        return X / std


def directory_to_samples(directory, flags=None):
    """ Read a directory, and list all subdirectories files as class sample """
    samples = []
    targets = []
    label = 0
    classes = sorted(os.walk(directory).next()[1])
    for c in classes:
        c_dir = os.path.join(directory, c)
        for sample in os.walk(c_dir).next()[2]:
            if not flags or any(flag in sample for flag in flags):
                    samples.append(os.path.join(c_dir, sample))
                    targets.append(label)
        label += 1
    return samples, targets


# ------------------
# OTHERS
# ------------------

def get_max(X):
    return np.max(X)


def get_mean(X):
    return np.mean(X)


def get_std(X):
    return np.std(X)
