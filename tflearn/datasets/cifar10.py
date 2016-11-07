""" CIFAR-10 Dataset

Credits: A. Krizhevsky. https://www.cs.toronto.edu/~kriz/cifar.html.

"""
from __future__ import absolute_import, print_function

import os
import sys
from six.moves import urllib
import tarfile

import numpy as np
import pickle

from ..data_utils import to_categorical


def load_data(dirname="cifar-10-batches-py", one_hot=False):
    tarpath = maybe_download("cifar-10-python.tar.gz",
                             "http://www.cs.toronto.edu/~kriz/",
                             dirname)
    X_train = []
    Y_train = []

    if dirname != './cifar-10-batches-py':
        dirname = os.path.join(dirname, 'cifar-10-batches-py')

    for i in range(1, 6):
        fpath = os.path.join(dirname, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        if i == 1:
            X_train = data
            Y_train = labels
        else:
            X_train = np.concatenate([X_train, data], axis=0)
            Y_train = np.concatenate([Y_train, labels], axis=0)

    fpath = os.path.join(dirname, 'test_batch')
    X_test, Y_test = load_batch(fpath)

    X_train = np.dstack((X_train[:, :1024], X_train[:, 1024:2048],
                         X_train[:, 2048:])) / 255.
    X_train = np.reshape(X_train, [-1, 32, 32, 3])
    X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048],
                        X_test[:, 2048:])) / 255.
    X_test = np.reshape(X_test, [-1, 32, 32, 3])

    if one_hot:
        Y_train = to_categorical(Y_train, 10)
        Y_test = to_categorical(Y_test, 10)

    return (X_train, Y_train), (X_test, Y_test)


def load_batch(fpath):
    with open(fpath, 'rb') as f:
        if sys.version_info > (3, 0):
            # Python3
            d = pickle.load(f, encoding='latin1')
        else:
            # Python2
            d = pickle.load(f)
    data = d["data"]
    labels = d["labels"]
    return data, labels


def maybe_download(filename, source_url, work_directory):
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        print("Downloading CIFAR 10, Please wait...")
        filepath, _ = urllib.request.urlretrieve(source_url + filename,
                                                 filepath, reporthook)
        statinfo = os.stat(filepath)
        print(('Succesfully downloaded', filename, statinfo.st_size, 'bytes.'))
        untar(filepath)
    return filepath

#reporthook from stackoverflow #13881092
def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))

def untar(fname):
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname)
        tar.extractall(path = '/'.join(fname.split('/')[:-1]))
        tar.close()
        print("File Extracted in Current Directory")
    else:
        print("Not a tar.gz file: '%s '" % sys.argv[0])
